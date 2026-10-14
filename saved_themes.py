import json
import os
import re
import shutil
from datetime import datetime, timezone
from io import BytesIO

from pypdf import PdfReader

BRAND_GUIDELINES_FILENAME = "brand_guidelines.txt"
BRAND_GUIDELINES_MAX_CHARS = 100_000

# Persisted under STORAGE_PATH (e.g. Railway volume /data/themes/) — shared across deploys/workers.
SAVED_DIR_NAME = "themes"


def saved_themes_root(storage_dir):
    root = os.path.join(storage_dir, SAVED_DIR_NAME)
    legacy = os.path.join(storage_dir, "saved_themes")
    if not os.path.isdir(root) and os.path.isdir(legacy):
        os.rename(legacy, root)
    os.makedirs(root, exist_ok=True)
    return root


def slugify(name):
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return s[:80] or "theme"


def unique_theme_id(storage_dir, display_name, exclude_id=None):
    base = slugify(display_name)
    root = saved_themes_root(storage_dir)
    candidate = base
    n = 2
    while os.path.isdir(os.path.join(root, candidate)) and candidate != exclude_id:
        candidate = f"{base}-{n}"
        n += 1
    return candidate


def list_saved(storage_dir):
    root = saved_themes_root(storage_dir)
    themes = []
    for name in sorted(os.listdir(root)):
        meta_path = os.path.join(root, name, "theme.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, "r") as f:
                data = json.load(f)
            ref_count = len(list_ref_files(storage_dir, name))
            brand_path = os.path.join(root, name, BRAND_GUIDELINES_FILENAME)
            themes.append({
                "theme_id": data.get("theme_id", name),
                "display_name": data.get("display_name", name),
                "source": "saved",
                "ref_count": ref_count,
                "updated_at": data.get("updated_at", ""),
                "has_brand_guidelines": bool(data.get("has_brand_guidelines"))
                or os.path.isfile(brand_path),
            })
        except Exception:
            continue
    return themes


def theme_dir(storage_dir, theme_id):
    return os.path.join(saved_themes_root(storage_dir), theme_id)


def list_ref_files(storage_dir, theme_id):
    ref_dir = os.path.join(theme_dir(storage_dir, theme_id), "refs")
    if not os.path.isdir(ref_dir):
        return []
    out = []
    for name in sorted(os.listdir(ref_dir)):
        if not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue
        path = os.path.join(ref_dir, name)
        if os.path.isfile(path):
            ext = os.path.splitext(name)[1].lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"
            if ext == ".webp":
                mime = "image/webp"
            out.append({"path": path, "filename": name, "mime": mime})
    return out


def load_saved(storage_dir, theme_id):
    td = theme_dir(storage_dir, theme_id)
    meta_path = os.path.join(td, "theme.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(theme_id)
    with open(meta_path, "r") as f:
        data = json.load(f)
    bp = os.path.join(td, "theme_breakdown.txt")
    fp = os.path.join(td, "style_fingerprint.txt")
    if os.path.isfile(bp):
        with open(bp, "r") as f:
            data["theme_breakdown"] = f.read()
    if os.path.isfile(fp):
        with open(fp, "r") as f:
            data["style_fingerprint"] = f.read()
    brand_path = os.path.join(td, BRAND_GUIDELINES_FILENAME)
    if os.path.isfile(brand_path):
        with open(brand_path, "r") as f:
            data["brand_guidelines"] = f.read()
    else:
        data["brand_guidelines"] = ""
    data["_theme_dir"] = td
    data["source"] = "saved"
    return data


def _write_refs(ref_dir, upload_files):
    if os.path.isdir(ref_dir):
        shutil.rmtree(ref_dir)
    os.makedirs(ref_dir, exist_ok=True)
    refs = []
    for i, f in enumerate(upload_files):
        raw = f.read()
        mime = f.content_type or "image/jpeg"
        ext = ".png" if mime == "image/png" else ".jpg"
        filename = f"{i:02d}{ext}"
        with open(os.path.join(ref_dir, filename), "wb") as out:
            out.write(raw)
        refs.append(filename)
    return refs


def extract_brand_pdf_text(upload_file):
    """Extract text from an uploaded brand-guidelines PDF."""
    if not upload_file or not getattr(upload_file, "filename", None):
        return None
    raw = upload_file.read()
    upload_file.seek(0)
    reader = PdfReader(BytesIO(raw))
    parts = []
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if text:
            parts.append(text)
    combined = "\n\n".join(parts).strip()
    if not combined:
        raise ValueError("Brand guidelines PDF has no extractable text.")
    if len(combined) > BRAND_GUIDELINES_MAX_CHARS:
        combined = combined[:BRAND_GUIDELINES_MAX_CHARS] + "\n…[truncated]"
    return combined


def save_analysis(
    storage_dir,
    theme_id,
    display_name,
    breakdown,
    fingerprint,
    upload_files,
    *,
    replace_refs=True,
    brand_guidelines_text=None,
):
    td = theme_dir(storage_dir, theme_id)
    os.makedirs(td, exist_ok=True)
    ref_dir = os.path.join(td, "refs")
    if replace_refs and upload_files:
        _write_refs(ref_dir, upload_files)
    with open(os.path.join(td, "theme_breakdown.txt"), "w") as f:
        f.write(breakdown)
    with open(os.path.join(td, "style_fingerprint.txt"), "w") as f:
        f.write(fingerprint)
    meta_path = os.path.join(td, "theme.json")
    meta = {}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    if brand_guidelines_text is not None:
        with open(os.path.join(td, BRAND_GUIDELINES_FILENAME), "w") as f:
            f.write(brand_guidelines_text)
        meta["has_brand_guidelines"] = bool(brand_guidelines_text.strip())
    meta.update({
        "theme_id": theme_id,
        "display_name": display_name,
        "source": "gpt",
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def resolve_theme_dir(storage_dir, theme_id):
    """Folder name on disk, or theme_id inside theme.json."""
    td = theme_dir(storage_dir, theme_id)
    if os.path.isdir(td):
        return td
    root = saved_themes_root(storage_dir)
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if name == theme_id:
            return path
        meta_path = os.path.join(path, "theme.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, "r") as f:
                data = json.load(f)
            if data.get("theme_id") == theme_id:
                return path
        except Exception:
            continue
    raise FileNotFoundError(theme_id)


def delete_theme(storage_dir, theme_id):
    td = resolve_theme_dir(storage_dir, theme_id)
    shutil.rmtree(td)


def delete_all_themes(storage_dir):
    root = saved_themes_root(storage_dir)
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
