import json
import os

KITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kits")

KIT_FILES = [
    "01_storyboard_instructions.md",
    "02_visual_guidelines.md",
    "03_brand_guidelines.md",
    "04_technical_qc.md",
    "05_reference_breakdowns.md",
]


def list_kits():
    if not os.path.isdir(KITS_DIR):
        return []
    kits = []
    for name in sorted(os.listdir(KITS_DIR)):
        path = os.path.join(KITS_DIR, name)
        readme = os.path.join(path, "00_README.md")
        if not os.path.isdir(path) or not os.path.isfile(readme):
            continue
        with open(readme, "r") as f:
            first = f.read(400)
        kits.append({
            "kit_id": name,
            "display_name": "ZO IPO LF Storyboard Kit" if name == "zo-ipo-lf" else name,
            "description": "IPO long-form graphics from script (G01→Gnn JSON)" if name == "zo-ipo-lf" else "",
        })
    return kits


def load_kit(kit_id):
    kit_dir = os.path.join(KITS_DIR, kit_id)
    if not os.path.isdir(kit_dir):
        raise FileNotFoundError(kit_id)
    parts = []
    for fname in KIT_FILES:
        fp = os.path.join(kit_dir, fname)
        if os.path.isfile(fp):
            with open(fp, "r") as f:
                parts.append(f"--- {fname} ---\n{f.read()}")
    schema_path = os.path.join(kit_dir, "storyboard_schema.json")
    schema = None
    if os.path.isfile(schema_path):
        with open(schema_path, "r") as f:
            schema = json.load(f)
    system_prompt = (
        "You are the ZO IPO LF storyboard generator.\n"
        "Follow the kit documents below exactly.\n"
        "Output ONE JSON object that validates against the provided JSON Schema.\n"
        "Graphics must appear in order G01, G02, … with continuous continuity.\n\n"
        + "\n\n".join(parts)
    )
    if schema:
        system_prompt += (
            "\n\n--- storyboard_schema.json ---\n"
            + json.dumps(schema, indent=2)
        )
    return {
        "kit_id": kit_id,
        "system_prompt": system_prompt,
        "schema": schema,
    }


def style_anchor_for_images(kit_id, max_chars=12000):
    kit_dir = os.path.join(KITS_DIR, kit_id)
    chunks = []
    for fname in ("02_visual_guidelines.md", "03_brand_guidelines.md"):
        fp = os.path.join(kit_dir, fname)
        if os.path.isfile(fp):
            with open(fp, "r") as f:
                chunks.append(f.read())
    text = "\n\n".join(chunks)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…[truncated]"
    return text


def build_image_prompt(graphic):
    """Turn one zo-ipo-lf graphic object into a Gemini scene prompt."""
    if not graphic or graphic.get("shot") == "PRESENTER":
        return None
    hero = graphic.get("hero") or {}
    texts = graphic.get("on_screen_text") or []
    builds = graphic.get("build") or []
    assets = graphic.get("assets_to_create") or []
    decor = graphic.get("decor") or []
    data = graphic.get("data")

    text_lines = []
    for t in texts:
        line = t.get("text", "")
        role = t.get("role", "")
        tok = t.get("colour_token", "")
        hl = t.get("highlight_phrase", "")
        if hl:
            line += f" (highlight '{hl}' in Zerodha blue)"
        text_lines.append(f"- [{role}] {line} {tok}".strip())

    asset_lines = [f"- {a.get('asset_id')}: {a.get('spec') or a.get('kind')}" for a in assets]

    data_block = ""
    if data:
        data_block = f"\nChart/data (exact values): {json.dumps(data, ensure_ascii=False)}\n"

    hero_block = ""
    if hero:
        hero_block = (
            f"Hero object: {hero.get('description') or hero.get('asset_id')}\n"
            f"Position ~{hero.get('position_px')}, width ~{hero.get('width_px')}px, "
            f"grounding: {hero.get('grounding', 'ground_line_trees')}.\n"
        )

    return (
        f"Generate ONE complete 16:9 storyboard frame (1920×1080) for the ZO IPO LF Zerodha Online series.\n"
        f"Graphic id: {graphic.get('id')} | Shot: {graphic.get('shot')} | Type: {graphic.get('graphic_type')}\n"
        f"Voiceover covered: {graphic.get('vo', '')}\n\n"
        f"{hero_block}"
        f"On-screen text (legible, Inter-style, exact wording):\n"
        + ("\n".join(text_lines) if text_lines else "- (infer from VO, sentence case, key phrase in #387ED1)\n")
        + data_block
        + "\nIMPORTANT: One finished STATIC frame only — no motion blur, no multi-step animation, "
        "no before/after sequence. Show the final composed board as a single still.\n"
        + f"\nDecor (edges only, 2–4): {', '.join(decor) if decor else 'sparkle_amber, sparkle_blue, cloud_outline, green accent dash bottom-left'}\n"
        + f"Accent dash: {graphic.get('accent_dash', 'green')}. "
        f"Include Zerodha logo top-left (blue on white board). "
        f"Include 'Source: RHP' bottom-left when facts are shown. "
        f"White dot-grid board background fading at centre; no black PDF title bars.\n"
        + (f"\nAsset specs:\n" + "\n".join(asset_lines) if asset_lines else "")
        + "\n\nIllustration style: semi-realistic flat vector, soft cel-shading, thin grey outlines, "
        "Zerodha blue glass/buildings where relevant, peach/blue sun circle behind hero, ground line + trees.\n"
        "Canvas: strict 16:9 landscape (1920×1080). Do not compose as a square or vertical card.\n"
    )
