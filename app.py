import os
import json
import base64
from io import BytesIO
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from google import genai as google_genai
from google.genai import types as genai_types
from dotenv import load_dotenv

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
7) You must NOT skip lines.
8) You must NOT merge lines.
9) You must NOT add new ideas.
You must visualize the script line-by-line or idea-by-idea.
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
- "line_number": integer
- "voiceover": the exact voiceover line unchanged
- "visual_prompt": the storyboard instruction paragraph for that line"""

IMAGE_GENERATION_PROMPT = """Your task is to analyze the provided set of reference images, which act as a visual toolkit for a specific creator or channel. Your goal is to reverse-engineer their unique visual style and then apply it to create a new image.
First, carefully study the reference images to identify the defining characteristics of their aesthetic. Pay close attention to:
* Color Palette: What are the dominant colors? Are they muted, vibrant, neon, or monochromatic?
* Textures & Materials: Does the style use paper-cutouts, digital gradients, hand-drawn sketches, 3D renders, or photographic elements?
* Lighting & Atmosphere: Is the lighting soft and flat, dramatic and moody, or glowing with digital light effects?
* Composition & Layout: Are elements centered, layered like a collage, or placed in a minimalist symmetrical way?
* Character/Object Design: Are figures realistic, silhouetted, cartoonish, or symbolic? How are objects rendered?
* Typography & Branding: Note the style of any text elements, labels, or logos present.
Once you have a solid understanding of this visual language, generate a single, detailed image generation prompt based on the user's visual prompt below. The final image must perfectly replicate the style, mood, and techniques found in the reference toolkit, making it look like an authentic piece from the same series.
Return ONLY the image generation prompt as plain text. No explanation. No commentary. No extra text."""

app = Flask(__name__)

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e)}), 500
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
imagen_client = google_genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))


def _run_gemini_image_gen(contents):
    """Try Gemini image generation models in order, return base64 PNG string."""
    models_to_try = [
        "gemini-2.0-flash-exp-image-generation",
        "gemini-2.5-flash-image",
    ]
    modality_combos = [["IMAGE"], ["TEXT", "IMAGE"]]

    for model in models_to_try:
        for modalities in modality_combos:
            try:
                result = imagen_client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=genai_types.GenerateContentConfig(
                        response_modalities=modalities,
                    ),
                )
                for part in result.candidates[0].content.parts:
                    if part.inline_data:
                        return base64.b64encode(part.inline_data.data).decode("utf-8")
            except Exception:
                continue

    raise Exception("No Gemini image model succeeded. Check your Google API key permissions.")

STORAGE_DIR = os.environ.get("STORAGE_PATH", os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR  = os.path.join(STORAGE_DIR, "images")
STATE_FILE  = os.path.join(STORAGE_DIR, ".session_state.json")
os.makedirs(IMAGES_DIR, exist_ok=True)

_default_state = {
    "theme_name": None,
    "theme_images": [],
    "theme_breakdown": None,
    "voiceover_script": None,
    "storyboard_data": None,
    "generated_images": {},
}

def _load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                saved = json.load(f)
                return {**_default_state, **saved}
        except Exception:
            pass
    return dict(_default_state)

def _save_state():
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

state = _load_state()


def encode_uploaded_files(files):
    encoded = []
    for f in files:
        data = base64.b64encode(f.read()).decode("utf-8")
        mime = f.content_type or "image/jpeg"
        encoded.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
    return encoded


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    theme_name = request.form.get("theme_name", "").strip()
    images = request.files.getlist("images")

    if not theme_name:
        return jsonify({"error": "Theme name required"}), 400
    if len(images) == 0 or len(images) > 10:
        return jsonify({"error": f"Upload between 1 and 10 images (got {len(images)})"}), 400

    image_content = encode_uploaded_files(images)
    state["theme_name"] = theme_name
    state["theme_images"] = image_content

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": f"{THEME_BREAKDOWN_PROMPT}\n\nTheme: {theme_name}"},
            *image_content,
        ],
    }]
    response = openai_client.chat.completions.create(model="gpt-4o", messages=messages)
    state["theme_breakdown"] = response.choices[0].message.content
    _save_state()
    return jsonify({"success": True})


@app.route("/storyboard", methods=["POST"])
def storyboard():
    body = request.get_json()
    voiceover = (body or {}).get("voiceover", "").strip()

    if not voiceover:
        return jsonify({"error": "Voiceover script required"}), 400
    if not state["theme_breakdown"]:
        return jsonify({"error": "Analyze theme first"}), 400

    state["voiceover_script"] = voiceover
    system_content = (
        f"{STORYBOARD_PROMPT}\n\n"
        f"--- VISUAL THEME SYSTEM (apply this grammar to every frame) ---\n"
        f"{state['theme_breakdown']}"
    )
    user_content = [
        {
            "type": "text",
            "text": (
                "These are the reference images that define this visual theme.\n\n"
                "CRITICAL INSTRUCTIONS for every visual_prompt you write:\n"
                "1. If a presenter, host, or on-screen character appears in these references, "
                "describe that EXACT person — their appearance, clothing, hair, skin tone, expression — "
                "in every frame they appear. Never replace them with a generic person.\n"
                "2. Replicate the exact background, set design, color grading, and lighting visible in these images.\n"
                "3. Replicate the exact graphic style, typography treatment, and motion language.\n"
                "4. Every visual_prompt must read as a specific shot direction, not a generic description.\n\n"
                f"Voiceover Script:\n{voiceover}"
            )
        },
        *state["theme_images"],
    ]
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        max_tokens=4096,
    )
    choice = response.choices[0]
    content = choice.message.content
    if not content:
        reason = choice.finish_reason or "unknown"
        raise ValueError(f"Model returned no content (finish_reason: {reason}). Try shortening the voiceover script.")
    raw = json.loads(content)
    result = raw if isinstance(raw, list) else raw.get("storyboard", list(raw.values())[0])
    state["storyboard_data"] = result
    state["generated_images"] = {}
    _save_state()
    return jsonify({"storyboard": result})


@app.route("/generate-image", methods=["POST"])
def generate_image():
    body = request.get_json()
    line_number = body.get("line_number")
    visual_prompt = body.get("visual_prompt", "").strip()

    if not state["theme_images"]:
        return jsonify({"error": "No theme images in session"}), 400
    if not visual_prompt:
        return jsonify({"error": "Visual prompt required"}), 400

    # Step 1: GPT-4o vision → ultra-detailed style fingerprint prompt
    # GPT-4o sees all reference images and produces an Imagen-ready prompt that encodes
    # exact character appearance, colors, lighting, composition, and style keywords.
    style_analysis = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert at writing image generation prompts that faithfully replicate a visual style.\n"
                    "You will be shown reference images that define a visual theme.\n"
                    "Your output must be a single image generation prompt (no headers, no explanation) that:\n"
                    "1. Describes the EXACT presenter/host character — face, hair, skin tone, expression, clothing, accessories\n"
                    "2. Describes the EXACT set/background — color, texture, props, lighting setup\n"
                    "3. Describes the EXACT color palette — dominant hues, contrast level, color temperature\n"
                    "4. Describes the EXACT visual style — photorealistic/illustrated/motion-graphic/cinematic, etc.\n"
                    "5. Describes the EXACT camera framing — wide shot, medium shot, close-up, angle\n"
                    "6. Appends the specific scene action at the end\n"
                    "Be hyper-specific. Never use vague words. Write as if describing to someone who cannot see the references.\n\n"
                    f"--- VISUAL THEME SYSTEM ---\n{state['theme_breakdown']}"
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "These are the reference images. Extract every visual detail and write a single "
                            "image generation prompt that would make Imagen reproduce this exact style.\n\n"
                            f"Scene to depict: {visual_prompt}"
                        ),
                    },
                    *state["theme_images"],
                ],
            },
        ],
        max_tokens=1024,
    )
    imagen_prompt = style_analysis.choices[0].message.content.strip()

    # Step 2: Imagen 4 generates from the style fingerprint prompt
    result = imagen_client.models.generate_images(
        model="imagen-4.0-generate-001",
        prompt=imagen_prompt,
        config=genai_types.GenerateImagesConfig(number_of_images=1),
    )
    img_bytes = result.generated_images[0].image.image_bytes
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    img_path = os.path.join(IMAGES_DIR, f"line_{line_number}.png")
    with open(img_path, "wb") as f:
        f.write(base64.b64decode(img_b64))

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
