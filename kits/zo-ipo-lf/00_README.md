# Zerodha Online — IPO Long-Format (ZO IPO LF) Storyboard Kit

This kit tells a graphics generator how to turn **a new IPO script + raw presenter footage** into an
**ordered storyboard**, where each graphic follows from the one before, starting at G01, in the
house style of the ZO IPO series.

The rules come from the first 60 seconds of the four approved videos in this series:

| # | Video | Res | First-minute shape |
|---|-------|-----|--------------------|
| 1 | Hero Motors IPO (LF final, 2nd draft) | 1920×1080, 25fps | OST title on footage → product scene → engine + promoter → OST numbered lists → pie |
| 2 | Jindal Supreme India Ltd IPO (V2) | 1920×1080, 25fps | Chapter slide + calendar → split-screen factory hub → OST map → production-chain sequence |
| 3 | SS Retail IPO (1 min) | 3840×2160, 25fps | Chapter slide + calendar → OST category cards → India map → store-format carousel (COFO/COCO/FOFO) |
| 4 | Sonaselection India Ltd IPO (1 min) | 3840×2160, 25fps | OST cold-open title → chapter slide + calendar → process flow → donut that changes over time → state map → factory + capacity ribbon |

The kit also follows the "ZO IPO LF Theme" sheet (colours, buildings, maps, pointers, bar/pie, numbers,
sequence, scene, basic FSS, chapter slide, payment UI, text-with-icon, icons, elements, single character).

## Files, and the order to feed them

| Order | File | What your code uses it for |
|---|---|---|
| 1 | `01_storyboard_instructions.md` | **The system prompt.** How a video is structured, the hero-object rule, the graphic-type library, script → graphic mapping, how graphics are sequenced and kept continuous. |
| 2 | `02_visual_guidelines.md` | Canvas, layout zones and pixel positions, type scale, illustration style and prompt template, decor kit, component specs, motion and transitions. |
| 3 | `03_brand_guidelines.md` | Logo, palette tokens, typeface, Source and SEBI disclaimer lines, naming and casing, copy rules, do's and don'ts. |
| 4 | `04_technical_qc.md` | Export specs, a pre-flight checklist, automatable checks, and the actual defects found in the four reference videos. |
| 5 | `05_reference_breakdowns.md` | Beat-by-beat timelines of all four first minutes. Use them as few-shot examples. |
| 6 | `storyboard_schema.json` | JSON Schema for the generator's output: one ordered `graphics[]` array, G01 → Gnn. |

## Pipeline contract (suggested)

```
INPUT
  script.txt            – full VO script (English)
  footage_index.json    – optional: list of A-roll takes with in/out timecodes
  rhp_facts.json        – optional: numbers pulled from the RHP (dates, %, counts, locations)

STEP 1  Segment the script into beats (one beat ≈ one spoken sentence or clause that carries ONE fact).
STEP 2  For each beat decide: PRESENTER (A-roll only) | OST (graphic over A-roll) | BOARD (full-screen graphic) | SPLIT (presenter cut-out + board).
STEP 3  Group consecutive board beats into SCENES that share one HERO OBJECT.
STEP 4  Emit graphics G01…Gnn in order, each with: type, hero, build steps synced to VO words, on-screen text,
        transition in/out, continuity (what carries over from the previous graphic), assets to make.
STEP 5  Run the QC rules from 04_technical_qc.md on the JSON before rendering.

OUTPUT  storyboard.json (validates against storyboard_schema.json) + storyboard.md (human-readable table)
```

## A note on how this was derived
- Frames were sampled at 2fps (Hero Motors) or 1fps (the other three), with 12.5fps checks on every transition.
  Colours were measured from frames and checked against the theme sheet.
- The audio was not transcribed (the speech-to-text model could not be downloaded here). VO wording was
  inferred from the on-screen text, which in this series tracks the VO almost word-for-word.
