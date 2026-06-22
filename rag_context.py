import hashlib
import json
import os
from datetime import datetime, timezone
from io import BytesIO

import chromadb
from pypdf import PdfReader

EMBED_MODEL = "text-embedding-3-small"
CHUNK_CHARS = 2400
CHUNK_OVERLAP = 400
TOP_K_PER_LINE = 4
MAX_TOTAL_CHUNKS = 24

FINANCE_CONTEXT_HEADER = """ADDITIONAL FINANCE CONTEXT:
Use the following retrieved context only to improve conceptual accuracy, visual metaphors, labels, UI states, and finance-specific details.
Do not rewrite the voiceover.
Do not add new narration.
Do not add facts that are not implied by the voiceover.
Use this context only to make the storyboard visually and conceptually sharper."""


def _paths(storage_dir):
    base = os.path.join(storage_dir, "vector_store")
    return {
        "base": base,
        "chroma": os.path.join(base, "varsity_context"),
        "registry": os.path.join(base, "indexed_files.json"),
    }


def _ensure_dirs(storage_dir):
    paths = _paths(storage_dir)
    os.makedirs(paths["chroma"], exist_ok=True)
    return paths


def _load_registry(registry_path):
    if not os.path.exists(registry_path):
        return {"files": []}
    try:
        with open(registry_path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict) and "files" in data:
                return data
    except Exception:
        pass
    return {"files": []}


def _save_registry(registry_path, registry):
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)


def _file_hash(data):
    return hashlib.sha256(data).hexdigest()


def _get_collection(storage_dir):
    paths = _ensure_dirs(storage_dir)
    client = chromadb.PersistentClient(path=paths["chroma"])
    return client.get_or_create_collection(name="varsity_context"), paths


def get_status(storage_dir):
    paths = _ensure_dirs(storage_dir)
    registry = _load_registry(paths["registry"])
    files = registry.get("files", [])
    try:
        collection, _ = _get_collection(storage_dir)
        chunk_count = collection.count()
    except Exception:
        chunk_count = 0
    return {
        "indexed_pdf_count": len(files),
        "total_chunks": chunk_count,
        "files": files,
        "ready": len(files) > 0 and chunk_count > 0,
    }


def has_indexed_context(storage_dir):
    status = get_status(storage_dir)
    return status["ready"]


def _extract_pdf_pages(data, filename):
    reader = PdfReader(BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page_number": i, "text": text, "filename": filename})
    return pages


def _chunk_text(text, filename, page_number):
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append({
                "text": piece,
                "filename": filename,
                "page_number": page_number,
                "chunk_index": idx,
                "source_label": "varsity",
            })
            idx += 1
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def _embed_texts(openai_client, texts):
    if not texts:
        return []
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def index_pdfs(openai_client, storage_dir, uploaded_files):
    collection, paths = _get_collection(storage_dir)
    registry = _load_registry(paths["registry"])
    known_hashes = {f["file_hash"] for f in registry.get("files", [])}

    already_indexed = 0
    newly_indexed = 0
    new_chunks_total = 0
    errors = []

    all_new_chunks = []
    new_registry_entries = []

    for f in uploaded_files:
        filename = f.filename or "document.pdf"
        if not filename.lower().endswith(".pdf"):
            errors.append(f"{filename}: not a PDF")
            continue
        try:
            data = f.read()
            digest = _file_hash(data)
            if digest in known_hashes:
                already_indexed += 1
                continue

            pages = _extract_pdf_pages(data, filename)
            if not pages:
                errors.append(f"{filename}: no extractable text")
                continue

            file_chunks = []
            for page in pages:
                file_chunks.extend(_chunk_text(page["text"], filename, page["page_number"]))

            if not file_chunks:
                errors.append(f"{filename}: no chunks created")
                continue

            for c in file_chunks:
                c["file_hash"] = digest
            all_new_chunks.extend(file_chunks)
            new_registry_entries.append({
                "filename": filename,
                "file_hash": digest,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
                "number_of_chunks": len(file_chunks),
            })
            known_hashes.add(digest)
            newly_indexed += 1
        except Exception as e:
            errors.append(f"{filename}: {e}")

    if all_new_chunks:
        texts = [c["text"] for c in all_new_chunks]
        embeddings = _embed_texts(openai_client, texts)
        ids = [f"{c['file_hash']}_{c['chunk_index']}_{c['page_number']}" for c in all_new_chunks]
        metadatas = [{
            "filename": c["filename"],
            "file_hash": c["file_hash"],
            "page_number": c["page_number"],
            "chunk_index": c["chunk_index"],
            "source_label": c["source_label"],
        } for c in all_new_chunks]
        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        new_chunks_total = len(all_new_chunks)
        registry.setdefault("files", []).extend(new_registry_entries)
        _save_registry(paths["registry"], registry)

    return {
        "already_indexed": already_indexed,
        "newly_indexed": newly_indexed,
        "new_chunks": new_chunks_total,
        "errors": errors,
        "ready": has_indexed_context(storage_dir),
    }


def _retrieve_for_query(openai_client, collection, query, top_k):
    if not query.strip():
        return []
    emb = _embed_texts(openai_client, [query])[0]
    results = collection.query(query_embeddings=[emb], n_results=top_k)
    out = []
    if not results or not results.get("documents"):
        return out
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    ids = results["ids"][0]
    for doc, meta, cid in zip(docs, metas, ids):
        out.append({"id": cid, "text": doc, "metadata": meta})
    return out


def retrieve_for_voiceover(openai_client, storage_dir, voiceover):
    collection, _ = _get_collection(storage_dir)
    if collection.count() == 0:
        return []

    lines = [l.strip() for l in voiceover.splitlines() if l.strip()]
    if not lines:
        lines = [voiceover.strip()]

    seen = {}
    for line in lines:
        hits = _retrieve_for_query(openai_client, collection, line, TOP_K_PER_LINE)
        for hit in hits:
            seen[hit["id"]] = hit
        if len(seen) >= MAX_TOTAL_CHUNKS:
            break

    return list(seen.values())[:MAX_TOTAL_CHUNKS]


def format_context_block(chunks):
    if not chunks:
        return ""
    parts = [FINANCE_CONTEXT_HEADER, ""]
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata") or {}
        src = meta.get("source_label", "varsity")
        fname = meta.get("filename", "unknown")
        page = meta.get("page_number", "?")
        parts.append(f"[Context {i} | source={src} | file={fname} | page={page}]")
        parts.append(chunk["text"])
        parts.append("")
    return "\n".join(parts).strip()
