# 02 — Visual Guidelines (ZO IPO LF)

All coordinates are for a **1920×1080** design canvas. For 4K masters, multiply by 2 (see 04_technical_qc.md).

---

## 1. Two worlds

| World | Background | Share of runtime |
|---|---|---|
| **Board (light)** | White + dot grid + soft white clearing in the content area | ~60–70% |
| **Studio (footage)** | Presenter at a desk; dark vertical wood slats; warm lamp at left; bookshelf at right | ~30–40% |

The board should feel like a clean white page that the footage "drops" onto. No coloured or gradient boards.
The only full-colour exception is the dark vignette behind OST maps and icon cards on footage.

---

## 2. Board background

- Base `#FFFFFF`.
- **Dot grid:** light grey dots (`#D9DEE3`–`#E3E6EA`), Ø≈4px, ~24px pitch. Full strength at the top edge and the top-right and right edges.
  Fades to clear white over the content area (radial/linear mask, ~60% of the canvas clear). Never a square line grid (that belongs to the product-video theme, not IPO).
- Optional **brand watermark motif** at 8–12% opacity, large and partly off-frame: a gear for auto/engineering (Hero Motors).
  Pick a motif from the company's domain (pipe end, fabric weave, store awning).

---

## 3. Layout zones (1080p)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ [ZERODHA logo] x24–238, y26–54                                           │
│                                                                          │
│   TITLE ZONE   y 120–330 (heading)     — centred, or left col x≈105      │
│                                                                          │
│   HERO ZONE    y 300–900               — centre, or right col x 1000–1880│
│                                                                          │
│   DATA / CAPTION ZONE  y 760–980       — bars, labels, counts            │
│ [accent dash] x15–300, y≈950 (5–10px tall)                               │
│ Source: RHP   x38, cap-height y≈990–1015                                 │
│ SEBI disclaimer x28, y≈1030–1050 (chapter slide only)                    │
└──────────────────────────────────────────────────────────────────────────┘
```

- **Title-safe:** 90px from the left/right edges, 100px from the top (below the logo), 140px from the bottom (above Source).
- **Chapter slide:** left text column x 105–950 (title + calendar). Right hero column x 1000–1880, hero centre ≈ (1450, 620).
- **Centred board:** heading centred at y≈130–260; hero centred at (960, 560); data under the ground line.
- **SPLIT:** board content in x 0–1200 (the hero top-left, y 90–470; sub-info y 500–1010). Presenter cut-out x 1080–1920, y 420–1080, desk edge at y≈1010. Keep labels left of x≈1100, or they collide with the presenter (see QC).
- **OST on footage:** lists in x 120–830, y 150–640 (the face is at x≈850–1150). Maps, numbers and cards in x 1150–1880 with a right-side dark vignette.

---

## 4. Typography

The typeface is **Inter** (a neutral grotesque that matches Zerodha UI). Tracking −1% on headings. No italics.
Uppercase only for acronyms (`COFO`, `IPO`, `FY`).

| Role | Size (1080p) | Weight | Colour |
|---|---|---|---|
| Chapter company name | 110–120px | Bold 700 | Zerodha Blue `#387ED1` |
| Chapter suffix ("India Limited IPO") | 96–104px | Regular 400 | Ink `#222222` / Deep Grey `#424242` |
| OST title on footage | 120–150px | Bold 700 | Blue name + White `#FFFFFF` second line/word |
| Board heading | 64–80px | Medium 500 (ink), or Bold 700 (blue headings) | Ink, with the key phrase in Blue |
| Big number | 80–140px | Regular/Medium (ink or blue); Bold if white inside a segment | Ink / Blue / White |
| Card / stage label | 36–48px | Medium 500–SemiBold 600 | Ink |
| Leader-line chart label | 40–46px | SemiBold 600 | Ink |
| Chart % | 70–90px | SemiBold/Bold | White inside a pie; segment colour beside a donut |
| OST list item | 44–50px | Bold 700 | Blue on white box |
| Source | 28px | Regular | Ink on board, White on footage |
| SEBI disclaimer | 18–20px | Regular | Ink 80% on board, White 70% on footage |

**Mixed-colour headings** (signature): neutral words in ink, the key concept in blue.
"Executing the **entire production** chain", "1.2-megawatt rooftop **solar installation**",
"**Hero** Motors" (Hero blue, Motors ink/white).

**Type-on:** every text element types on character by character (15–30 chars/s). Finish within 0.6–1.2s.
Chapter titles show a blinking text cursor while typing. Long labels may reveal each character with a small
scale/opacity ramp (letters "settle" in, as in "Franchise O w n e d").

---

## 5. Illustration style (hero objects)

- **Look:** semi-realistic flat vector with soft cel-shading, thin dark-grey outlines (1.5–2.5px at 1080p),
  clean highlights. Objects read clearly at thumbnail size.
- **Palette inside illustrations:** Zerodha blue family (`#387ED1`, `#5094E5`, navy `#1F3D7A`), white/light greys,
  metallic greys for steel, accents in sunshine yellow `#FFA412` / gold `#F6C343`. Small greens only for trees and plants.
- **Branding inside the art** is allowed and expected: the company name on the store fascia, factory wall or tag
  ("SS Retail", "Sonaselection India", "Hero" on the powertrain). Spell it exactly as in the RHP.
- **Grounding:** each hero sits on (a) a 2px ink ground line with two round green trees at the ends, (b) a soft
  pale-blue elliptical floor shadow, or (c) a curved light-grey floor with diagonal hatching.
- **Sun circle:** a large tinted circle behind the upper part of the hero. Blue-tint `#E5F1FD` for product heroes,
  amber-tint `#FEF3DD`–`#FDEFD3` for buildings and stores.
- **People:** the promoter as a real grayscale cut-out photo (never illustrated), cropped at the frame edge.
  Generic people use the line-art avatar set (theme sheet: "Single Character").
- **Icon set:** small flat icons in the same palette (store with a striped amber/blue awning, factory, pin, rupee
  bag, etc.; theme sheet: "Icons").

**Prompt template for generating a hero illustration**
```
Clean semi-realistic flat vector illustration of {OBJECT} for a fintech explainer video,
soft cel shading, thin dark grey outlines, Zerodha blue (#387ED1) and white colour scheme with
small sunshine-yellow (#FFA412) accents, metallic light greys, {BRAND TEXT on object if any},
isolated on pure white background, three-quarter front view, centred, no text other than the brand name,
no drop shadow except a soft pale-blue floor shadow, 4K, consistent with an industrial/retail icon set.
```
Export the hero as a transparent PNG at 2× the size it will be used. Keep each object's
perspective consistent across a video (all ¾-front, or all straight-front for carousel heroes).

---

## 6. Decor kit

Scatter 2–4 per board, at the edges, never touching content or the hero.

| Element | Spec | Where seen |
|---|---|---|
| 4-point sparkle | Amber `#F6A720`, 40–80px, with slight rotation; small blue version `#387ED1` | Every board, top-left and bottom-right corners; on the "o" of a title |
| Outline cloud | 2px blue `#387ED1` outline, white fill, 60–100px | Chapter slides, maps, carousel |
| Triangle | Solid, ~40–60px, rotated; peach `#F7B9A0`, purple `#A964FF`, amber | Hero Motors scenes |
| Squiggle | Hand-drawn loop "ℓℓℓ", 3px grey stroke | Beside titles |
| Ring | Hollow circle, 2px light grey | Scene corners |
| Accent dash | Rounded bar, 8–10px × 100–210px; green `#4CAF52` (chapter/scene) or amber `#F6A825` (maps/carousel) | Bottom-left above Source; sometimes bottom-right |
| Gear | Light blue-grey `#D5DEE3` outline gear, partly off-frame | Engineering companies |
| Curved arrow | Thin 2–3px ink hand-drawn arrow with a small head | Annotations: from a label to a part of the hero |
| Dashed curved arrow | 2px dashed ink | From a progress bar to its caption |

---

## 7. Component specs

**Calendar strip (chapter slide)**
- A card ~800×250, radius 16, fill white, 2px border `#DEEAF8`, header band `#DEEAF8` with 8 navy spiral rings.
- 7 cells: date (ink, 44px SemiBold) over month (ink, 26px). Thin grey vertical dividers.
- Selected date: cell fill `#FBE1B3`, amber chip `#F5A319` (radius 20) with `Open`/`Closed` in ink 28px Medium, and the weekday below the card in grey 24px.

**OST numbered list**
- Circle Ø120, fill `#0F7CFC`→`#387ED1`, white numeral Bold 90px, outer glow ring (2px, 30% white, +10px).
- Label box: height 76, padding 14/18, fill white, 2px border `#387ED1` at 60%, text Bold 46px blue.

**Product card (on board)**
- Radius 24, fill `#E5F1FD` (primary) or `#E4E4E4` (secondary). Label at the top of the card (ink 44px Medium), object overlapping the bottom edge of the card.

**Progress / share bar**
- 690×80, radius 4. Blue `#4B87D0`/`#387ED1`, remainder `#CFCFCF` (white board) or `#888888` (split board).
- Inside label: % in white Bold ~70px, then "of total sales" in white Regular 40px.

**Acronym chip**
- Outlined rounded rect, radius 18, 3px blue border, white fill, ink Regular 80px, padding 24/32.

**Map tag**
- Amber fill `#F6B24A`, 1.5px dark border, ink Medium 34px, a white dot with a grey ring at the anchor point.

**Pie/donut**
- Pie: Ø≈640, 1px darker outline. Donut: outer Ø≈560, ring width ~80.
- Colours in order: Blue `#387ED1` → Sunshine `#FFA412` → Green `#4CAF52` → Orange `#F07335` → Purple `#A964FF` (secondary palette; only when more than 2 segments).

**Dark OST card (on footage)**
- Fill `#1B242B` at ~85%, 2–3px border `#2A5BA8`, radius 2–4, white label 30–34px. Product on a blue/yellow podium.

---

## 8. Motion language

| Behaviour | Spec |
|---|---|
| Ease | Fast-out / slow-in (cubic-bezier 0.2, 0.8, 0.2, 1). Small overshoot on pops (scale 0.9 → 1.04 → 1, 250–350ms). |
| Object entry | Drop from above with a slight squash on landing, or scale-up from 0.8 with a 6-frame motion blur. One object per VO item. |
| Text | Type-on (see §4). Never fade a whole sentence in. |
| Lines/connectors | Draw on (stroke trim) 300–500ms, starting at the hero end. |
| Charts | Pie wipes clockwise 400–600ms. % counters roll 500–800ms. Bars fill left→right 500ms. |
| Morph | Same element, interpolated values. The year label rolls vertically (old out up, new in from below). |
| Carousel | Active → side: translate ±650px, scale 0.7, opacity 0.25, 500ms. New hero enters from the opposite side. |
| Camera | Board pans/zooms on maps (2–3s, ease in-out). Digital punch-in on the presenter for OST cards (~1.25×). |
| Idle | Decor drifts ≤10px. Smoke or wheels may loop subtly. The hero stays still once landed. |
| Exit | Elements exit only at a scene end or when the subject changes. Sub-info can wipe out while the hero stays. |
| Hold | Each completed state holds until the VO line ends (minimum 1.2s after the last reveal). |
