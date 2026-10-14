import os
import json
import re
import base64
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from google import genai as google_genai
from google.genai import types as genai_types
from dotenv import load_dotenv
import rag_context
import saved_themes
import storyboard_kits

load_dotenv(override=True)

THEME_BREAKDOWN_PROMPT = """You are a senior visual systems architect, motion designer, and storyboard engineer. You will be given a batch of reference images that belong to a single visual theme.
Your task is NOT to describe the images individually.
Your task is to reverse-engineer the underlying VISUAL SYSTEM.
Do NOT do storyboarding.
Do NOT map to scripts.
Do NOT describe assets one by one.
You must extract and formalize the DESIGN GRAMMAR of the theme.
Your output must be a structured system with the following sections:
1) Core Visual Identity
2) Core Visual Philosophy
3) Background System
4) Color System
5) Typography System
6) Text Treatment Patterns
7) Motion Grammar
8) Composition Rules
9) Layout Patterns
10) Visual Metaphor Library
11) UI & Interface Motifs (if any)
12) Data Visualization Style
13) Human Presence System
14) Emotional Language
15) Narrative Visual Structure
16) Visual Rhythm
17) Spatial Design
18) Iconography Style
19) Asset Categories (Reusable Library)
20) Storyboard Grammar for this Theme
21) AI Visual Prompt Structure (theme-specific)
22) Differentiation from Other Visual Systems
23) Identity Summary (one-line definition)
Rules:
* Think in SYSTEMS, not scenes.
* Think in GRAMMAR, not assets.
* Think in PATTERNS, not examples.
* No storyboards.
* No script mapping.
* No narration.
* No opinions.
* No generic design words.
* No fluff language.
* No emojis.
* No bullet spam without structure.
* No marketing tone.
Your output must read like a DESIGN SYSTEM SPEC, not a description."""

STORYBOARD_PROMPT = """You are a professional storyboard artist, motion designer, and visual systems translator.
You will be given:
1) A voiceover script
2) A selected visual theme system (pre-defined visual grammar)
Your task is to convert the SCRIPT into a VISUAL STORYBOARD that follows the EXACT visual grammar of the selected theme.
CRITICAL RULES:
1) You are NOT a writer.
2) You must NOT rewrite, paraphrase, summarize, or improve the voiceover.
3) You must NOT add narration.
4) You must NOT explain the storyboard.
5) You must NOT add commentary.
6) You must NOT add extra lines.
7) You must NOT add new ideas.
8) Every non-empty voiceover line must appear exactly once across all beats (unchanged). When beats group multiple lines, join them with a single newline in voiceover.
Follow the SHOT GROUPING rules appended to this prompt for how many lines belong in each beat.
Each storyboard unit MUST follow this exact format:
[Storyboard instructions in a single paragraph, written as a visual description]
Exact corresponding voiceover line (unchanged)
Structure Rules:
* No bullet points outside brackets
* No section headings
* No numbering
* No emojis
* No summaries
* No labels like "Scene 1"
* No explanations
* No formatting
* No markdown sections
* No extra text
* No commentary
Storyboard rules:
* Use visual metaphors where the theme requires
* Use symbolic language if the theme is symbolic
* Use UI language if the theme is UI-based
* Use clean diagrams if the theme is documentary
* Use kinetic motion if the theme is motion-graphic
* Use editorial collage if the theme is surreal
You must assume the editor is skilled but needs clear visual direction.
Think like a director + designer + motion artist, not a writer.
End cleanly when the voiceover ends.
Return your output as a JSON object with a key "storyboard" containing an array. Each element must have:
- "line_number": integer (beat index, starting at 1)
- "voiceover": exact script line(s) for this beat, unchanged (newline between grouped lines)
- "visual_prompt": one storyboard instruction paragraph covering the whole beat
- "script_lines": optional string, e.g. "12-14", listing which script line numbers this beat covers

TECHNICAL UI RULES (infer from voiceover — no separate visual script is provided):
* Input is voiceover only. You must decide what appears on screen from each line's meaning.
* When a line explains product, finance, trading, settlement, platform features, or mechanics, write visual_prompt as instructional UI direction — not generic B-roll.
* Infer the correct visual treatment: icons, diagrams, product UI panels, labels, comparisons, state changes, and motion intent implied by what is being said.
* When the voiceover introduces a concept, show the representing element. When it contrasts options, show split/highlight/full-screen focus. When something is unsupported or expired, grey out or remove from frame. When something converts or delivers, show A-to-B transformation.
* For comparisons (two exchanges, two settlement types, before/after), specify layout and difference visually.
* For feature explanations (basket orders, order slicing, terminal layouts), infer multi-part UI compositions from the voiceover and reference theme.
* Preserve continuity across related lines: reuse the same metaphor objects and UI components throughout a sequence.
* Name exact on-screen text, widget names, and data fields when the voiceover implies them."""

STORYBOARD_GRANULARITY = {
    "smart": """--- SHOT GROUPING (smart beats) ---
Do NOT assign one image per voiceover line. Group consecutive lines into one beat when they share one visual idea, the same scene/build, or one motion build.
Use one beat for a single strong visual (chapter slide, map, chart, presenter stat). Combine short punchy lines (e.g. two questions, setup + payoff) into one frame when one illustration carries both.
Typical: 2–5 short lines per beat, or one dense line alone. Aim for clearly fewer beats than script lines (often ~35–55% of non-empty lines for explainers).
Consecutive beats may reuse the same base frame and add layers when the theme grammar says so.""",
    "1": """--- SHOT GROUPING (every line) ---
Create exactly one beat per non-empty script line, in order. Do not merge lines.""",
    "2": """--- SHOT GROUPING (~2 lines per frame) ---
Pair consecutive non-empty lines (1–2, 3–4, …). A single leftover line at the end is its own beat. Exact wording only.""",
    "3": """--- SHOT GROUPING (~3 lines per frame) ---
Group consecutive non-empty lines in threes; 1–2 trailing lines form the last beat(s). Exact wording only.""",
}


def _storyboard_granularity_block(mode):
    key = (mode or "smart").strip().lower()
    if key not in STORYBOARD_GRANULARITY:
        key = "smart"
    return STORYBOARD_GRANULARITY[key]


TECHNICAL_UI_RULES = """When the visual_prompt describes a technical or product explanation:
* Follow the visual_prompt's UI direction precisely — it was inferred from the voiceover for instructional accuracy.
* Match the reference toolkit's UI language, component shapes, typography, and icon style.
* Render the specific UI state specified (selected tab, highlighted row, disabled/greyed module, popup open, side-by-side panels).
* Show metaphor objects and transformations as directed (contract paper, cash, delivery, folders, bridges, sliced order chunks).
* Include legible on-screen labels for key terms when the prompt implies them.
* Do not simplify into generic stock finance imagery."""

IMAGE_GENERATION_PROMPT = """Your task is to analyze the provided set of reference images, which act as a visual toolkit for a specific creator or channel. Your goal is to reverse-engineer their unique visual style and then apply it to create a new image.
First, carefully study the reference images to identify the defining characteristics of their aesthetic. Pay close attention to:
* Color Palette: What are the dominant colors? Are they muted, vibrant, neon, or monochromatic?
* Textures & Materials: Does the style use paper-cutouts, digital gradients, hand-drawn sketches, 3D renders, or photographic elements?
* Lighting & Atmosphere: Is the lighting soft and flat, dramatic and moody, or glowing with digital light effects?
* Composition & Layout: Are elements centered, layered like a collage, or placed in a minimalist symmetrical way?
* Character/Object Design: Are figures realistic, silhouetted, cartoonish, or symbolic? How are objects rendered?
* Typography & Branding: Note the style of any text elements, labels, or logos present.
Once you have a solid understanding of this visual language, generate a single, detailed image generation prompt based on the user's visual prompt below. The final image must perfectly replicate the style, mood, and techniques found in the reference toolkit, making it look like an authentic piece from the same series.

If the scene is technical or product-focused, the prompt must specify exact UI elements, states, labels, icons, and visual transformations — not generic illustrations.

Return ONLY the image generation prompt as plain text. No explanation. No commentary. No extra text."""

app = Flask(__name__)


@app.before_request
def _reload_state_from_disk():
    """Shared session file on STORAGE_PATH — every worker/replica sees the same themes."""
    global state
    state = _load_state()

@app.errorhandler(404)
def handle_not_found(e):
    if request.path in ("/", "/index.html") or request.path.endswith(".html"):
        return send_from_directory(APP_ROOT, "index.html")
    return jsonify({"error": str(e)}), 404


@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e)}), 500
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
imagen_client = google_genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))


def _gemini_image_gen_config(modalities):
    kwargs = {"response_modalities": modalities}
    try:
        kwargs["image_config"] = genai_types.ImageConfig(aspect_ratio="16:9")
    except Exception:
        pass
    return genai_types.GenerateContentConfig(**kwargs)


def _ensure_landscape_16_9_png(raw_bytes):
    """Center-crop to 16:9 and resize to 1920×1080 so UI always gets widescreen frames."""
    from PIL import Image

    im = Image.open(BytesIO(raw_bytes)).convert("RGB")
    w, h = im.size
    target_w, target_h = 1920, 1080
    target_ratio = target_w / target_h
    if w <= 0 or h <= 0:
        return raw_bytes
    current_ratio = w / h
    if abs(current_ratio - target_ratio) > 0.02:
        if current_ratio > target_ratio:
            new_w = int(h * target_ratio)
            left = (w - new_w) // 2
            im = im.crop((left, 0, left + new_w, h))
        else:
            new_h = int(w / target_ratio)
            top = (h - new_h) // 2
            im = im.crop((0, top, w, top + new_h))
    if im.size != (target_w, target_h):
        im = im.resize((target_w, target_h), Image.Resampling.LANCZOS)
    out = BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()


def _run_gemini_image_gen(contents):
    """Try Gemini image generation models in order, return base64 PNG string."""
    models_to_try = [
        "gemini-2.5-flash-image",
        "gemini-3.1-flash-image",
        "gemini-3-pro-image",
    ]
    modality_combos = [["TEXT", "IMAGE"], ["IMAGE"]]
    errors = []

    for model in models_to_try:
        for modalities in modality_combos:
            try:
                result = imagen_client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=_gemini_image_gen_config(modalities),
                )
                candidates = result.candidates or []
                if not candidates:
                    errors.append(f"{model}/{modalities}: no candidates returned")
                    continue
                content = candidates[0].content
                if content is None:
                    reason = getattr(candidates[0], "finish_reason", "unknown")
                    errors.append(f"{model}/{modalities}: empty content (finish_reason={reason})")
                    continue
                for part in (content.parts or []):
                    if part.inline_data:
                        print(f"[image-gen] success: model={model} modalities={modalities}", flush=True)
                        return base64.b64encode(part.inline_data.data).decode("utf-8")
                errors.append(f"{model}/{modalities}: responded but no image part")
            except Exception as e:
                errors.append(f"{model}/{modalities}: {e}")
                continue

    raise Exception("No Gemini image model succeeded. Attempts: " + " | ".join(errors))

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.environ.get("STORAGE_PATH", APP_ROOT)
IMAGES_DIR  = os.path.join(STORAGE_DIR, "images")
THEME_IMAGES_DIR = os.path.join(STORAGE_DIR, "theme_images")
STATE_FILE  = os.path.join(STORAGE_DIR, ".session_state.json")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(THEME_IMAGES_DIR, exist_ok=True)

_default_state = {
    "theme_name": None,
    "theme_id": None,
    "theme_source": None,
    "theme_images": [],
    "theme_breakdown": None,
    "style_fingerprint": None,
    "brand_guidelines": None,
    "voiceover_script": None,
    "storyboard_data": None,
    "storyboard_format": None,
    "active_kit_id": None,
    "generated_images": {},
}

def _load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                saved = json.load(f)
                merged = {**_default_state, **saved}
                merged["theme_images"] = _normalize_theme_image_refs(merged.get("theme_images", []))
                return merged
        except Exception:
            pass
    return dict(_default_state)

def _save_state():
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

state = _load_state()


def _clear_theme_images_dir():
    if not os.path.isdir(THEME_IMAGES_DIR):
        return
    for name in os.listdir(THEME_IMAGES_DIR):
        path = os.path.join(THEME_IMAGES_DIR, name)
        if os.path.isfile(path):
            os.remove(path)


def _normalize_theme_image_refs(theme_images):
    """Session stores filenames only; drop legacy inline base64 entries."""
    if not theme_images:
        return []
    if isinstance(theme_images[0], dict) and theme_images[0].get("filename"):
        return theme_images
    return []


def _save_theme_images(files):
    _clear_theme_images_dir()
    refs = []
    for i, f in enumerate(files):
        raw = f.read()
        mime = f.content_type or "image/jpeg"
        ext = ".png" if mime == "image/png" else ".jpg"
        filename = f"{i}{ext}"
        with open(os.path.join(THEME_IMAGES_DIR, filename), "wb") as out:
            out.write(raw)
        refs.append({"filename": filename, "mime": mime})
    return refs


def _active_theme_id():
    return state.get("theme_id")


def _iter_theme_ref_paths():
    tid = _active_theme_id()
    if tid:
        for ref in saved_themes.list_ref_files(STORAGE_DIR, tid):
            yield ref["path"], ref["mime"]
        return
    for ref in state.get("theme_images") or []:
        path = os.path.join(THEME_IMAGES_DIR, ref["filename"])
        if os.path.isfile(path):
            yield path, ref["mime"]


def _activate_theme(theme_id, display_name, breakdown, fingerprint, brand_guidelines=None):
    state["theme_id"] = theme_id
    state["theme_source"] = "saved"
    state["theme_name"] = display_name
    state["theme_breakdown"] = breakdown
    state["style_fingerprint"] = fingerprint
    state["brand_guidelines"] = (brand_guidelines or "").strip() or None
    state["theme_images"] = []
    _clear_theme_images_dir()


def _brand_guidelines_block():
    bg = (state.get("brand_guidelines") or "").strip()
    if not bg:
        return ""
    cap = bg[: saved_themes.BRAND_GUIDELINES_MAX_CHARS]
    return (
        "\n\n--- BRAND GUIDELINES (mandatory — typography, colors, logo, layout rules) ---\n"
        f"{cap}\n"
    )


def _encode_saved_theme_refs(theme_id):
    encoded = []
    for ref in saved_themes.list_ref_files(STORAGE_DIR, theme_id):
        with open(ref["path"], "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        encoded.append({
            "type": "image_url",
            "image_url": {"url": f"data:{ref['mime']};base64,{data}"},
        })
    return encoded


def _brand_text_from_upload(file_storage):
    if not file_storage or not file_storage.filename:
        return None
    return saved_themes.extract_brand_pdf_text(file_storage)


def _load_theme_into_state(theme_id):
    try:
        theme = saved_themes.load_saved(STORAGE_DIR, theme_id)
    except FileNotFoundError:
        return False
    fp = (theme.get("style_fingerprint") or "").strip()
    breakdown = theme.get("theme_breakdown")
    if not breakdown:
        return False
    _activate_theme(
        theme_id,
        theme.get("display_name", theme_id),
        breakdown,
        fp,
        theme.get("brand_guidelines"),
    )
    _save_state()
    return True


def _ensure_theme_for_storyboard(body):
    """Standard storyboard needs theme_breakdown in session; load from request or disk."""
    if state.get("theme_breakdown"):
        return True
    theme_id = ((body or {}).get("theme_id") or state.get("theme_id") or "").strip()
    if theme_id:
        return _load_theme_into_state(theme_id)
    return False


def _gpt_analyze_theme(theme_name, image_content, brand_guidelines_text=""):
    brand_block = ""
    if (brand_guidelines_text or "").strip():
        brand_block = (
            "\n\n--- BRAND GUIDELINES DOCUMENT (merge into the visual system; enforce on every rule) ---\n"
            f"{brand_guidelines_text.strip()[:80000]}\n"
        )

    def call_breakdown():
        return openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{THEME_BREAKDOWN_PROMPT}\n\nTheme: {theme_name}"
                            f"{brand_block}"
                        ),
                    },
                    *image_content,
                ],
            }],
        )

    def call_fingerprint():
        return openai_client.chat.completions.create(
            model="gpt-4o",
            max_tokens=800,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert prompt engineer for image generation models.\n"
                        "Study the reference images carefully and write ONE detailed image generation prompt "
                        "that fully describes the visual style so precisely that any scene can be generated in this exact look.\n\n"
                        "Your prompt MUST include:\n"
                        "- On-screen talent (if any): exact physical description (appearance, clothing, accessories)\n"
                        "- Set/background: exact description (colors, textures, props, furniture, depth)\n"
                        "- Lighting: exact setup (direction, intensity, color temperature, shadows)\n"
                        "- Color palette: list 4-6 dominant hex-approximate colors (e.g. deep navy #1a2a4a)\n"
                        "- Visual style: photorealistic/cinematic/broadcast TV/motion-graphic/illustrated — be exact\n"
                        "- Camera: focal length feel, shot type (medium shot, MCU, wide), angle\n"
                        "- Post-processing look: color grade, contrast, saturation level\n"
                        "- Technical UI patterns: how product UI, icons, metaphors, and state changes are rendered\n"
                        "- Brand compliance: logo placement, primary/secondary colors, typography from brand guidelines when provided\n\n"
                        f"{TECHNICAL_UI_RULES}\n\n"
                        "Output ONLY the prompt string. No labels. No headers. No explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze these reference images and write the style fingerprint prompt:"
                                f"{brand_block}"
                            ),
                        },
                        *image_content,
                    ],
                },
            ],
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        breakdown_resp = executor.submit(call_breakdown).result()
        fingerprint_resp = executor.submit(call_fingerprint).result()
    breakdown = breakdown_resp.choices[0].message.content
    fingerprint = fingerprint_resp.choices[0].message.content.strip()
    return breakdown, fingerprint


def _theme_images_available():
    return any(True for _ in _iter_theme_ref_paths())


def _encode_theme_images_for_openai():
    encoded = []
    for path, mime in _iter_theme_ref_paths():
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        encoded.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{data}"},
        })
    return encoded


def _theme_images_for_gemini():
    parts = []
    for path, mime in _iter_theme_ref_paths():
        with open(path, "rb") as f:
            raw = f.read()
        parts.append(genai_types.Part(
            inline_data=genai_types.Blob(mime_type=mime, data=raw)
        ))
    return parts


def _generated_image_path(line_number):
    path = os.path.join(IMAGES_DIR, f"line_{line_number}.png")
    return path if os.path.isfile(path) else None


def _storyboard_continuity_anchor_line(exclude_line=None):
    """Lowest beat index that already has a generated still (series master frame)."""
    candidates = []
    for key in state.get("generated_images") or {}:
        try:
            ln = int(key)
        except (TypeError, ValueError):
            continue
        if exclude_line is not None and ln == exclude_line:
            continue
        if _generated_image_path(ln):
            candidates.append(ln)
    return min(candidates) if candidates else None


def _continuity_prompt_block(anchor_line, target_line, regenerate=False):
    if anchor_line is not None and target_line is not None and anchor_line != target_line:
        master = "Frame 1 (series master)" if anchor_line == 1 else f"Frame {anchor_line}"
        return (
            f"SERIES CONTINUITY (mandatory): The image attached immediately above is {master} "
            f"from this same storyboard session. You are generating Frame {target_line}.\n"
            f"Match {master} exactly for illustration style, line weight, textures, palette, "
            "background treatment, board/chapter chrome, logo lockup, typography, shadows, and spacing. "
            "Theme reference PNGs define the brand; the master frame defines how this run interprets them. "
            "Only scene content and on-screen copy change per the scene prompt below. "
            "Same IPO graphics package — not a new art direction.\n\n"
        )
    if target_line == 1:
        if regenerate:
            return (
                "Re-rendering Frame 1 — the SERIES MASTER for this storyboard. "
                "Every other frame must match this look; keep one coherent package.\n\n"
            )
        return (
            "This is Frame 1 — the SERIES MASTER for this storyboard run. "
            "Lock the production look here; all later frames will match this still.\n\n"
        )
    return ""


def encode_uploaded_files(files):
    encoded = []
    for f in files:
        data = base64.b64encode(f.read()).decode("utf-8")
        mime = f.content_type or "image/jpeg"
        encoded.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
        f.seek(0)
    return encoded


@app.route("/")
@app.route("/index.html")
def index():
    resp = send_from_directory(APP_ROOT, "index.html")
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


@app.route("/health", methods=["GET"])
def health():
    themes_root = saved_themes.saved_themes_root(STORAGE_DIR)
    writable = os.access(STORAGE_DIR, os.W_OK)
    try:
        theme_count = len(saved_themes.list_saved(STORAGE_DIR))
    except Exception as e:
        return jsonify({
            "ok": False,
            "storage_dir": STORAGE_DIR,
            "themes_root": themes_root,
            "writable": writable,
            "error": str(e),
        }), 500
    return jsonify({
        "ok": True,
        "storage_dir": STORAGE_DIR,
        "themes_root": themes_root,
        "writable": writable,
        "theme_count": theme_count,
    })


@app.route("/themes", methods=["GET"])
def themes_list():
    active_id = state.get("theme_id")
    active_name = state.get("theme_name")
    try:
        themes = saved_themes.list_saved(STORAGE_DIR)
    except Exception as e:
        print(f"[themes] list_saved failed: {e}", flush=True)
        return jsonify({"error": f"Could not read saved themes: {e}", "storage_dir": STORAGE_DIR}), 500
    return jsonify({
        "themes": themes,
        "themes_storage": saved_themes.saved_themes_root(STORAGE_DIR),
        "active_theme_id": active_id,
        "active_theme_name": active_name,
        "storage_dir": STORAGE_DIR,
    })


@app.route("/themes/create", methods=["POST"])
def themes_create():
    display_name = request.form.get("display_name", "").strip()
    images = request.files.getlist("images")
    if not display_name:
        return jsonify({"error": "Theme name required"}), 400
    if len(images) < 10 or len(images) > 20:
        return jsonify({"error": f"Upload between 10 and 20 images (got {len(images)})"}), 400

    try:
        brand_new = _brand_text_from_upload(request.files.get("brand_guidelines"))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    image_content = encode_uploaded_files(images)
    brand_for_gpt = brand_new or ""
    breakdown, fingerprint = _gpt_analyze_theme(display_name, image_content, brand_for_gpt)
    theme_id = saved_themes.unique_theme_id(STORAGE_DIR, display_name)
    saved_themes.save_analysis(
        STORAGE_DIR, theme_id, display_name, breakdown, fingerprint, images,
        brand_guidelines_text=brand_new if brand_new is not None else None,
    )
    loaded = saved_themes.load_saved(STORAGE_DIR, theme_id)
    _activate_theme(
        theme_id, display_name, breakdown, fingerprint.strip(),
        loaded.get("brand_guidelines"),
    )
    _save_state()
    ref_count = len(saved_themes.list_ref_files(STORAGE_DIR, theme_id))
    return jsonify({
        "success": True,
        "theme_id": theme_id,
        "display_name": display_name,
        "source": "saved",
        "ref_count": ref_count,
    })


@app.route("/themes/<theme_id>/reanalyze", methods=["POST"])
def themes_reanalyze(theme_id):
    display_name = request.form.get("display_name", "").strip()
    images = [f for f in request.files.getlist("images") if f.filename]
    try:
        brand_new = _brand_text_from_upload(request.files.get("brand_guidelines"))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    try:
        existing = saved_themes.load_saved(STORAGE_DIR, theme_id)
    except FileNotFoundError:
        return jsonify({"error": "Theme not found"}), 404
    name = display_name or existing.get("display_name") or theme_id
    ref_count = len(saved_themes.list_ref_files(STORAGE_DIR, theme_id))

    if len(images) >= 10:
        if len(images) > 20:
            return jsonify({"error": f"Upload between 10 and 20 images (got {len(images)})"}), 400
        image_content = encode_uploaded_files(images)
        replace_refs = True
        upload_for_save = images
    elif len(images) == 0 and ref_count >= 10:
        image_content = _encode_saved_theme_refs(theme_id)
        if len(image_content) < 10:
            return jsonify({"error": "Theme has too few saved reference images. Upload 10–20 PNGs."}), 400
        replace_refs = False
        upload_for_save = []
    else:
        return jsonify({
            "error": f"Upload 10–20 new reference PNGs, or leave images empty to re-run with saved refs ({ref_count} on disk).",
        }), 400

    brand_for_gpt = brand_new if brand_new is not None else (existing.get("brand_guidelines") or "")
    breakdown, fingerprint = _gpt_analyze_theme(name, image_content, brand_for_gpt)
    saved_themes.save_analysis(
        STORAGE_DIR, theme_id, name, breakdown, fingerprint, upload_for_save,
        replace_refs=replace_refs,
        brand_guidelines_text=brand_new,
    )
    loaded = saved_themes.load_saved(STORAGE_DIR, theme_id)
    if _active_theme_id() == theme_id:
        _activate_theme(
            theme_id, name, breakdown, fingerprint.strip(),
            loaded.get("brand_guidelines"),
        )
    _save_state()
    return jsonify({
        "success": True,
        "theme_id": theme_id,
        "display_name": name,
        "ref_count": len(saved_themes.list_ref_files(STORAGE_DIR, theme_id)),
        "has_brand_guidelines": bool((loaded.get("brand_guidelines") or "").strip()),
    })


def _themes_delete_impl(theme_id):
    try:
        saved_themes.delete_theme(STORAGE_DIR, theme_id)
    except FileNotFoundError:
        return jsonify({"error": "Theme not found"}), 404
    if state.get("theme_id") == theme_id:
        state["theme_id"] = None
        state["theme_name"] = None
        state["theme_breakdown"] = None
        state["style_fingerprint"] = None
        state["brand_guidelines"] = None
        state["theme_source"] = None
    _save_state()
    return jsonify({"success": True})


@app.route("/themes/<theme_id>", methods=["DELETE"])
@app.route("/themes/<theme_id>/delete", methods=["POST"])
def themes_delete(theme_id):
    return _themes_delete_impl(theme_id)


@app.route("/themes", methods=["DELETE"])
def themes_delete_all():
    saved_themes.delete_all_themes(STORAGE_DIR)
    state["theme_id"] = None
    state["theme_name"] = None
    state["theme_breakdown"] = None
    state["style_fingerprint"] = None
    state["brand_guidelines"] = None
    state["theme_source"] = None
    _save_state()
    return jsonify({"success": True})


@app.route("/themes/select", methods=["POST"])
def themes_select():
    body = request.get_json() or {}
    theme_id = (body.get("theme_id") or "").strip()
    if not theme_id:
        return jsonify({"error": "theme_id required"}), 400
    try:
        theme = saved_themes.load_saved(STORAGE_DIR, theme_id)
    except FileNotFoundError:
        return jsonify({"error": f"Unknown theme: {theme_id}"}), 404
    fp = (theme.get("style_fingerprint") or "").strip()
    _activate_theme(
        theme_id,
        theme.get("display_name", theme_id),
        theme["theme_breakdown"],
        fp,
        theme.get("brand_guidelines"),
    )
    ref_count = len(saved_themes.list_ref_files(STORAGE_DIR, theme_id))
    _save_state()
    return jsonify({
        "success": True,
        "theme_id": theme_id,
        "display_name": state["theme_name"],
        "ref_count": ref_count,
        "has_brand_guidelines": bool((theme.get("brand_guidelines") or "").strip()),
    })


@app.route("/context/status", methods=["GET"])
def context_status():
    try:
        return jsonify(rag_context.get_status(STORAGE_DIR))
    except Exception as e:
        return jsonify({"error": str(e), "ready": False, "indexed_pdf_count": 0}), 500


@app.route("/context/index", methods=["POST"])
def context_index():
    pdfs = request.files.getlist("pdfs")
    if not pdfs:
        return jsonify({"error": "Upload at least one PDF"}), 400
    try:
        result = rag_context.index_pdfs(openai_client, STORAGE_DIR, pdfs)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Indexing failed: {e}"}), 500


@app.route("/kits", methods=["GET"])
def kits_list():
    return jsonify({"kits": storyboard_kits.list_kits()})


@app.route("/kits/select", methods=["POST"])
def kits_select():
    body = request.get_json() or {}
    kit_id = (body.get("kit_id") or "").strip()
    if not kit_id:
        return jsonify({"error": "kit_id required"}), 400
    try:
        storyboard_kits.load_kit(kit_id)
    except FileNotFoundError:
        return jsonify({"error": f"Unknown kit: {kit_id}"}), 404
    state["active_kit_id"] = kit_id
    state["storyboard_format"] = "zo-ipo-lf"
    _save_state()
    return jsonify({"success": True, "kit_id": kit_id})


@app.route("/storyboard", methods=["POST"])
def storyboard():
    body = request.get_json()
    voiceover = (body or {}).get("voiceover", "").strip()
    use_context = bool((body or {}).get("use_context"))
    beat_granularity = (body or {}).get("beat_granularity") or "smart"
    script_scope = (body or {}).get("script_scope") or "full"
    selected_line_numbers = (body or {}).get("selected_line_numbers") or []
    kit_id = (body or {}).get("kit_id") or state.get("active_kit_id")

    if not voiceover:
        return jsonify({"error": "Voiceover script required"}), 400
    if kit_id:
        return _storyboard_zo_ipo_lf(body, voiceover, kit_id, use_context)
    if not _ensure_theme_for_storyboard(body):
        return jsonify({"error": "Select a saved theme, or use the ZO IPO LF kit generator"}), 400
    if use_context and not rag_context.has_indexed_context(STORAGE_DIR):
        return jsonify({"error": "Context mode enabled but no indexed PDFs found. Upload Varsity PDFs and click Index context first."}), 400

    # GPT-4o context window: 128K tokens input, 16K tokens output.
    # Reference images consume ~1000-2000 tokens each, theme breakdown ~3K.
    # Each storyboard line in output is ~100-150 tokens.
    # At 16K output tokens, ~100-130 lines max before truncation.
    # Rough guard: warn if script looks very large (>500 lines).
    line_count = len([l for l in voiceover.splitlines() if l.strip()])
    if line_count > 500:
        return jsonify({"error": f"Script has {line_count} lines. Keep under 500 lines to avoid output truncation."}), 400

    state["voiceover_script"] = voiceover
    state["script_scope"] = script_scope
    state["selected_line_numbers"] = selected_line_numbers
    state["beat_granularity"] = beat_granularity
    system_content = (
        f"{STORYBOARD_PROMPT}\n\n"
        f"{_storyboard_granularity_block(beat_granularity)}\n\n"
        f"--- VISUAL THEME SYSTEM (apply this grammar to every frame) ---\n"
        f"{state['theme_breakdown']}"
        f"{_brand_guidelines_block()}"
    )
    if use_context:
        try:
            chunks = rag_context.retrieve_for_voiceover(openai_client, STORAGE_DIR, voiceover)
            context_block = rag_context.format_context_block(chunks)
            if context_block:
                system_content += f"\n\n---\n{context_block}"
        except Exception as e:
            return jsonify({"error": f"Context retrieval failed: {e}"}), 500

    ref_images = _encode_theme_images_for_openai()
    ref_preamble = (
        "These are the reference images that define this visual theme.\n\n"
        if ref_images
        else "No reference images are attached; use the theme system description above.\n\n"
    )
    user_content = [
        {
            "type": "text",
            "text": (
                f"{ref_preamble}"
                "The input below is VOICEOVER ONLY. No separate visual direction is provided.\n"
                "For each beat, infer what must appear on screen from the grouped voiceover and the theme grammar.\n\n"
                "CRITICAL INSTRUCTIONS for every visual_prompt you write:\n"
                "1. If a presenter, host, or on-screen character appears in these references, "
                "describe that EXACT person in every frame they appear. Never replace them with a generic person.\n"
                "2. Replicate the exact background, set design, color grading, and lighting visible in these images.\n"
                "3. Replicate the exact graphic style, typography treatment, and motion language.\n"
                "4. Every visual_prompt must read as a specific shot direction, not a generic description.\n"
                "5. When the voiceover explains technical, product, or platform concepts, infer and write full "
                "instructional UI direction: widgets, icons, labels, states, comparisons, and transformations.\n\n"
                f"Voiceover Script:\n{voiceover}"
            )
        },
        *ref_images,
    ]
    # Attempt 1: JSON mode (fastest, cleanest)
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        max_tokens=16000,
    )
    choice = response.choices[0]
    content = choice.message.content

    # JSON mode + vision can trigger a refusal stored outside content
    if not content:
        refusal = getattr(choice.message, "refusal", None)
        print(f"[storyboard] JSON mode failed — refusal={refusal} finish={choice.finish_reason}", flush=True)
        # Attempt 2: plain text mode, parse JSON manually
        plain_system = system_content + "\n\nIMPORTANT: Return ONLY a raw JSON object with no markdown fences."
        response2 = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": plain_system},
                {"role": "user", "content": user_content},
            ],
            max_tokens=16000,
        )
        content = response2.choices[0].message.content
        if not content:
            raise ValueError("Model refused to generate the storyboard. Try rephrasing or shortening the voiceover script.")
        # Strip markdown fences if present
        content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    raw = json.loads(content)
    result = raw if isinstance(raw, list) else raw.get("storyboard", list(raw.values())[0])
    state["storyboard_data"] = result
    state["generated_images"] = {}
    _save_state()
    state["storyboard_format"] = "standard"
    state["active_kit_id"] = None
    _save_state()
    return jsonify({"format": "standard", "storyboard": result, "voiceover_script": voiceover})


def _norm_vo_line(s):
    s = (s or "").lower()
    s = re.sub(r"[^\w\s.%]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _vo_fragment_core(vo):
    vo = (vo or "").strip()
    vo = re.sub(r"^[\s.…]+", "", vo)
    vo = re.sub(r"[\s.…]+$", "", vo)
    vo = re.sub(r"\.+$", "", vo)
    return vo.strip()


def _enrich_graphics_vo(graphics, voiceover):
    """Replace truncated GPT vo snippets with full script lines."""
    lines = [l.strip() for l in voiceover.splitlines() if l.strip()]
    if not lines:
        return graphics
    used = set()
    for g in graphics:
        vo = (g.get("vo") or "").strip()
        if not vo:
            continue
        if vo in lines:
            used.add(lines.index(vo))
            continue
        core = _vo_fragment_core(vo)
        if not core:
            continue
        best_i = None
        best_len = 0
        for i, line in enumerate(lines):
            if i in used:
                continue
            if core in line or line in core:
                if len(line) > best_len:
                    best_i, best_len = i, len(line)
                    continue
            nc, nl = _norm_vo_line(core), _norm_vo_line(line)
            if len(nc) >= 8 and (nc in nl or nl in nc):
                if len(line) > best_len:
                    best_i, best_len = i, len(line)
        if best_i is not None:
            g["vo"] = lines[best_i]
            used.add(best_i)
            continue
        for line in lines:
            if core in line or _norm_vo_line(core) in _norm_vo_line(line):
                g["vo"] = line
                break
    return graphics


def _storyboard_zo_ipo_lf(body, voiceover, kit_id, use_context):
    try:
        kit = storyboard_kits.load_kit(kit_id)
    except FileNotFoundError:
        return jsonify({"error": f"Unknown kit: {kit_id}"}), 404

    company = (body or {}).get("company") or {}
    ipo = (body or {}).get("ipo") or {}
    image_only = (body or {}).get("image_only", True)
    if image_only:
        user_text = (
            "Generate storyboard JSON for STATIC IMAGE generation only (this tool does not render video).\n"
            "Rules:\n"
            "- Include ONLY beats that need a generated board/OST still (BOARD, CHAPTER, OST, SPLIT).\n"
            "- Do NOT include PRESENTER rows in graphics[] — presenter is real footage.\n"
            "- One still frame per graphic; omit build[] or leave it empty (no animation planning).\n"
            "- Each row must have: id (G01…), shot, graphic_type, vo (full exact script line(s), never truncate or use ellipsis), hero, "
            "on_screen_text, assets_to_create[].spec describing the still for an image model.\n"
            "- Skip timecodes unless useful; focus on visual content tied to the script.\n\n"
        )
    else:
        user_text = (
            "Generate a complete storyboard JSON for this IPO video.\n"
            "Segment the script into beats, assign shot types, hero objects, build steps, and continuity.\n"
            "Include PRESENTER beats (P01, P02, …) for timing continuity where the references do.\n\n"
        )
    if company.get("legal_name"):
        user_text += f"Company legal name: {company['legal_name']}\n"
    if company.get("short_name"):
        user_text += f"Short name: {company['short_name']}\n"
    if ipo.get("open_date"):
        user_text += f"IPO open date: {ipo['open_date']}\n"
    if ipo.get("close_date"):
        user_text += f"IPO close date: {ipo['close_date']}\n"
    user_text += f"\nVoiceover script:\n{voiceover}"

    system_content = kit["system_prompt"]
    if use_context and rag_context.has_indexed_context(STORAGE_DIR):
        try:
            chunks = rag_context.retrieve_for_voiceover(openai_client, STORAGE_DIR, voiceover)
            context_block = rag_context.format_context_block(chunks)
            if context_block:
                system_content += f"\n\n---\n{context_block}"
        except Exception as e:
            return jsonify({"error": f"Context retrieval failed: {e}"}), 500

    line_count = len([l for l in voiceover.splitlines() if l.strip()])
    if line_count > 200:
        return jsonify({"error": f"Script has {line_count} lines. For ZO IPO LF kit, try a section (≤200 lines) per run."}), 400

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_text},
        ],
        response_format={"type": "json_object"},
        max_tokens=16000,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Model refused to generate the storyboard. Try a shorter script section.")
    raw = json.loads(content)
    graphics = raw.get("graphics") if isinstance(raw, dict) else None
    if not graphics:
        return jsonify({"error": "Output missing graphics[] array. Retry or shorten the script."}), 500
    if image_only:
        graphics = [g for g in graphics if g.get("shot") != "PRESENTER"]
        raw["graphics"] = graphics

    graphics = _enrich_graphics_vo(graphics, voiceover)
    raw["graphics"] = graphics

    state["voiceover_script"] = voiceover
    state["storyboard_data"] = raw
    state["storyboard_format"] = "zo-ipo-lf"
    state["active_kit_id"] = kit_id
    state["generated_images"] = {}
    _save_state()
    return jsonify({
        "format": "zo-ipo-lf",
        "storyboard": raw,
        "graphics": graphics,
        "voiceover_script": voiceover,
    })


def _tweak_visual_prompt_for_regen(visual_prompt):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You refine storyboard visual prompts for image generation. "
                    "Keep the same scene, storyboard intent, UI elements, labels, and metaphors. "
                    "Make small improvements only: clearer composition, lighting, framing, or visual specificity. "
                    "Do not add new story beats, narration, or elements not in the original."
                ),
            },
            {
                "role": "user",
                "content": f"Refine this visual prompt for a fresh generation attempt:\n\n{visual_prompt}",
            },
        ],
        max_tokens=600,
    )
    content = response.choices[0].message.content
    return content.strip() if content else visual_prompt


@app.route("/generate-image", methods=["POST"])
def generate_image():
    body = request.get_json()
    line_number = body.get("line_number")
    visual_prompt = body.get("visual_prompt", "").strip()
    regenerate = bool((body or {}).get("regenerate"))

    if not _theme_images_available():
        tid = _active_theme_id() or "your-theme"
        hint = f"Re-save the theme with reference images (stored under {saved_themes.saved_themes_root(STORAGE_DIR)}/{tid}/refs/)."
        return jsonify({"error": f"No theme reference images available. {hint}"}), 400
    if not visual_prompt:
        return jsonify({"error": "Visual prompt required"}), 400

    graphic = (body or {}).get("graphic")
    use_kit = state.get("storyboard_format") == "zo-ipo-lf" and state.get("active_kit_id")
    kit_ref_priority = ""
    if use_kit:
        kit_id = state["active_kit_id"]
        if graphic and graphic.get("shot") == "PRESENTER":
            return jsonify({"error": "Presenter beats use footage — generate images only for BOARD/CHAPTER/OST/SPLIT graphics."}), 400
        kit_prompt = storyboard_kits.build_image_prompt(graphic) if graphic else None
        if kit_prompt:
            visual_prompt = kit_prompt if not visual_prompt else f"{kit_prompt}\n\nEditor notes:\n{visual_prompt}"
        fingerprint = storyboard_kits.style_anchor_for_images(kit_id)
        theme_fp = (state.get("style_fingerprint") or "").strip()
        if theme_fp:
            fingerprint += (
                "\n\n--- Saved theme fingerprint (match the reference PNGs above; kit JSON defines layout) ---\n"
                f"{theme_fp[:6000]}\n"
            )
        kit_ref_priority = (
            "ZO IPO LF JSON kit mode: scene content comes from the kit prompt below. "
            "Visual execution MUST match the attached reference PNGs from the user's saved theme "
            "(palette, flat-vector weight, logo placement, board texture). Reference images override generic stock looks.\n\n"
        )
    else:
        fingerprint = state.get("style_fingerprint") or ""

    if regenerate:
        visual_prompt = _tweak_visual_prompt_for_regen(visual_prompt)

    try:
        target_line = int(line_number)
    except (TypeError, ValueError):
        target_line = None

    anchor_line = None
    if target_line is not None and not (regenerate and target_line == 1):
        if _generated_image_path(1) and target_line != 1:
            anchor_line = 1
        else:
            anchor_line = _storyboard_continuity_anchor_line(
                exclude_line=target_line if regenerate else None,
            )
            if anchor_line == target_line:
                anchor_line = None

    parts = _theme_images_for_gemini()
    if anchor_line is not None and target_line is not None and anchor_line != target_line:
        with open(_generated_image_path(anchor_line), "rb") as f:
            parts.append(genai_types.Part(
                inline_data=genai_types.Blob(mime_type="image/png", data=f.read())
            ))

    continuity_block = _continuity_prompt_block(anchor_line, target_line, regenerate=regenerate)
    regen_note = (
        "Produce a fresh variation of this scene. Same storyboard intent and instructions, "
        "with refined composition and visual execution. Keep series continuity with the master frame.\n\n"
        if regenerate else ""
    )
    parts.append(genai_types.Part(text=(
        "These are the reference images that define the visual style. "
        "Generate a new image that is visually faithful to this exact style, "
        "replicating set design, color palette, lighting, and composition. "
        "MANDATORY CANVAS: landscape 16:9 only (1920×1080). Never square, never portrait, never 1:1. "
        "Compose the full scene for widescreen broadcast — same aspect ratio as the reference frames. "
        "Output a single static 16:9 frame — no motion, no animation sequence. "
        "Ignore any PDF-style title bars or documentation chrome in the references; "
        "match the production frame style only.\n\n"
        f"{continuity_block}"
        f"{kit_ref_priority}"
        f"{regen_note}"
        f"Style anchor:\n{fingerprint}\n"
        f"{_brand_guidelines_block()}\n"
        f"{TECHNICAL_UI_RULES}\n\n"
        f"Scene to generate: {visual_prompt}"
    )))
    contents = [genai_types.Content(parts=parts, role="user")]
    img_b64 = _run_gemini_image_gen(contents)

    img_path = os.path.join(IMAGES_DIR, f"line_{line_number}.png")
    png_bytes = _ensure_landscape_16_9_png(base64.b64decode(img_b64))
    with open(img_path, "wb") as f:
        f.write(png_bytes)

    state["generated_images"][str(line_number)] = True
    _save_state()
    return jsonify({"image_url": f"/image/{line_number}"})


@app.route("/image/<int:line_number>")
def serve_image(line_number):
    return send_from_directory(IMAGES_DIR, f"line_{line_number}.png")


@app.route("/download/<int:line_number>")
def download_image(line_number):
    return send_from_directory(
        IMAGES_DIR,
        f"line_{line_number}.png",
        as_attachment=True,
        download_name=f"storyboard_line_{line_number}.png",
    )


@app.route("/test-gen")
def test_gen_page():
    return """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Image Gen Test</title>
<style>
  body{background:#0e0e0e;color:#d4d4d4;font-family:system-ui,sans-serif;max-width:700px;margin:40px auto;padding:0 20px}
  h2{color:#e0e0e0;margin-bottom:24px}
  label{display:block;font-size:12px;color:#666;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;margin-top:18px}
  input,textarea{width:100%;background:#181818;border:1px solid #2a2a2a;border-radius:8px;color:#e0e0e0;font-size:14px;padding:10px 12px;box-sizing:border-box}
  textarea{resize:vertical;height:100px}
  button{margin-top:18px;background:#e0e0e0;color:#0e0e0e;border:none;border-radius:8px;font-size:13px;font-weight:600;padding:11px 22px;cursor:pointer}
  #result{margin-top:24px}
  #result img{width:100%;border-radius:10px;border:1px solid #2a2a2a}
  #err{color:#e05252;background:#1f0e0e;border:1px solid #3a1e1e;border-radius:8px;padding:12px;margin-top:16px}
  #log{color:#666;font-size:12px;margin-top:10px}
</style></head>
<body>
<h2>Image Generation Test</h2>
<label>Reference Images (up to 10)</label>
<input type="file" id="imgs" accept="image/*" multiple />
<label>Scene / Prompt</label>
<textarea id="prompt" placeholder="Describe the scene to generate..."></textarea>
<button onclick="run()">Generate</button>
<div id="log"></div>
<div id="result"></div>
<script>
async function run(){
  const files = Array.from(document.getElementById('imgs').files).slice(0,10);
  const prompt = document.getElementById('prompt').value.trim();
  const log = document.getElementById('log');
  const result = document.getElementById('result');
  result.innerHTML = ''; log.textContent = 'Generating…';
  const fd = new FormData();
  files.forEach(f => fd.append('images', f));
  fd.append('prompt', prompt);
  try{
    const res = await fetch('/test-gen/run', {method:'POST', body:fd});
    const data = await res.json();
    if(!res.ok) throw new Error(data.error);
    log.textContent = 'Done.';
    result.innerHTML = '<img src="data:image/png;base64,' + data.image + '" />';
  } catch(e){
    log.textContent = '';
    result.innerHTML = '<div id="err">' + e.message + '</div>';
  }
}
</script>
</body></html>"""


@app.route("/test-gen/run", methods=["POST"])
def test_gen_run():
    images = request.files.getlist("images")
    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt required"}), 400

    parts = []
    for f in images:
        raw = f.read()
        mime = f.content_type or "image/jpeg"
        parts.append(genai_types.Part(
            inline_data=genai_types.Blob(mime_type=mime, data=raw)
        ))
    parts.append(genai_types.Part(text=(
        "The images above are visual style references. Generate a new image that replicates "
        "their exact visual style, color palette, lighting, character design, and composition.\n\n"
        f"Scene to generate: {prompt}"
    )))
    contents = [genai_types.Content(parts=parts, role="user")]
    img_b64 = _run_gemini_image_gen(contents)
    return jsonify({"image": img_b64})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=port)
