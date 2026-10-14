# 04 — Technical QC & Technical Detailing (ZO IPO LF)

## 1. Delivery specs

| Item | Spec |
|---|---|
| Frame rate | **25 fps** (all four references). Animate on whole frames: 1 frame = 40ms. |
| Master resolution | **Standardise on one.** The references are mixed: Hero Motors and Jindal are 1920×1080, SS Retail and Sona are 3840×2160. Recommended: design at 1920×1080 with vector/2× raster assets, and render a 3840×2160 master. |
| Aspect | 16:9. Safe areas: title-safe 90px left/right, 100px top, 140px bottom (1080p values). |
| Graphics export | Transparent PNG sequences or ProRes 4444 for OST/SPLIT layers. ProRes 422 HQ / H.264 high-bitrate for full boards. |
| Colour | sRGB / Rec.709. Check that Zerodha Blue renders `#387ED1` ±3 per channel after encode. |
| Audio | Music bed + VO. Graphics never carry their own SFX louder than the music (the references use subtle whooshes on wipes). |
| Asset resolution | Raster hero illustrations ≥ 2× their on-screen size; maps and charts as vectors. |

## 2. Timing rules (machine-checkable)

| Rule | Value |
|---|---|
| Reveal ↔ VO word sync | A reveal starts 0–4 frames **after** its trigger word begins; never before. |
| Type-on speed | 15–30 chars/s; a line completes in ≤1.2s. |
| Minimum hold of a completed state | ≥ 1.2s (30 frames) after the last reveal. |
| Board wipe / push / zoom-reveal | 8–10 frames. |
| Presenter break length | 2–6s. |
| Longest board without a break | ≤ 20s (Jindal's 33s run works only because it is SPLIT with the presenter visible). |
| OST list exit | 8 frames, slide left + motion blur. |
| Counter roll | 12–20 frames. |

## 3. Pre-render checklist (run on storyboard.json)

**Content accuracy**
- [ ] Every number matches `rhp_facts.json` (value, unit, FY).
- [ ] Parts sum correctly: store counts = total (316 + 84 + 103 = 503), pie % = 100 (54 + 46), donut % = 100 for each FY.
- [ ] Chart geometry matches the data (a 54% slice is 194.4°; bar fill width = % × bar width).
- [ ] Dates and weekdays agree (16 Sep 2026 = Wednesday, etc.), and the open date is before the close date.
- [ ] Company and third-party names are spelt as in the RHP or the official brand (flag likely typos: "Duccati").

**Text**
- [ ] Heading ≤6 words, label ≤4 words a line and ≤2 lines.
- [ ] Casing is consistent for the company name across all graphics.
- [ ] One spelling system (British/Indian) throughout: galvanised/galvanisation, metres.
- [ ] No stray punctuation (e.g. "Rajasthan." at the end of a label) and consistent spacing ("metres/year", not "meters/ year").
- [ ] Mixed-colour headings: only the key phrase is blue.

**Layout**
- [ ] The logo bbox (x24–238, y26–54) is never overlapped.
- [ ] The Source bbox (x30–320, y980–1020) is never overlapped by accent dashes, bars, cut-outs or decor.
      *(Failed in Jindal 52–53s and SS Retail 3–8s, where the green/amber dash sits on "Source: RHP".)*
- [ ] On SPLIT, no text extends past x≈1100 behind the presenter cut-out.
      *(Failed in Jindal 37–42s: "Galvanised tubular street lighting poles" is clipped to "stree".)*
- [ ] On OST, no element overlaps the presenter's face box (≈x 820–1180, y 220–620) or the mic.
- [ ] Decor elements keep ≥40px clearance from text and the hero.
- [ ] Accent dashes sit fully inside the frame (Sona's bottom-right green dash is cut at the frame edge).

**Continuity**
- [ ] `same_hero`/`resume` graphics reuse identical hero position, scale and asset.
- [ ] Elements from the next graphic don't pre-roll into the current shot.
      *(Failed in Hero Motors 46.5–48.5s: scooter/motorcycle icons appear bottom-left over the presenter and Source before the cut.)*
- [ ] "Source: RHP" never half-fades on a hard cut (Hero Motors 42s shows a partially faded Source).
- [ ] The SEBI disclaimer is on the first title graphic.

## 4. Post-render checks (automatable with ffmpeg + OCR)

1. `ffprobe`: fps = 25, resolution = master spec, duration matches the edit.
2. Scene-cut detection (`select='gt(scene,0.25)'`) → compare cut times to the storyboard in/out (±2 frames).
3. Sample 1 frame per second → OCR → diff against the storyboard's `on_screen_text` (catches typos and casing).
4. Pixel probes on every board frame: logo region blue `#387ED1`±10; Source region present.
5. Colour check: sample the blue in titles, bars and pies; flag any blue more than 12 ΔE from `#387ED1`
   (the OST number circle `#0F7CFC` is an allowed exception).
6. Contact sheet (4×3 tiles, 1fps) for human review of the hero rule: one dominant object per board tile.

## 5. Defects found in the four reference videos (fix list / regression tests)

| Video | Time | Issue | Category |
|---|---|---|---|
| Hero Motors | 13.5–19s | "Hero motors" (lower-case m) vs "Hero Motors" elsewhere | Casing consistency |
| Hero Motors | 1.5s vs 9s vs 35s | Three treatments of the company name: blue + white two-tone, all blue, blue + ink two-tone | Brand consistency |
| Hero Motors | 28–30s | "Duccati" should be "Ducati" | Spelling (third-party brand) |
| Hero Motors | 42.0s | "Source: RHP" partly faded on the cut | Continuity |
| Hero Motors | 46.5–48.5s | Vehicle icons from the next/previous graphic visible bottom-left over the presenter and Source | Stray element / pre-roll |
| Jindal | 37–42s | "Galvanised tubular street lighting poles" clipped behind the presenter cut-out | Safe-area collision |
| Jindal | 25s, 41s, 52–53s | Green accent dash touching or overlapping the progress bar and "Source: RHP" | Overlap |
| Jindal | 51–55s | "Galvanised" (UK) vs "galvanization" (US) in the same video | Spelling system |
| SS Retail | 3–8s | Green dash overlaps "Source: RHP" | Overlap |
| SS Retail | 0–2s | SEBI disclaimer over footage at small size, low contrast | Legibility |
| Sona | 43–44s | "Hamirgarh, Bhilwara Rajasthan." trailing full stop; label clipped during the zoom | Punctuation / clipping |
| Sona | 45–55s | "82.4 million meters/ year": "meters" (US) and a space after the slash | Spelling and spacing |
| Sona | 2–18s | Green accent dash cropped at the right frame edge | Safe area |
| All | — | Mixed master resolutions (1080p vs 2160p) | Delivery spec |

## 6. Technical detailing for the generator (what each graphic record must carry)

- `id`, `in_tc`, `out_tc` (25fps timecode `HH:MM:SS:FF`), `duration_frames`
- `shot` (PRESENTER/OST/BOARD/SPLIT/CHAPTER) and `graphic_type`
- `hero` {asset_id, position_px, scale, ground: line | floor | sun_circle_blue | sun_circle_amber}
- `build[]` — ordered steps: {trigger_word, start_offset_frames, element, animation, duration_frames}
- `on_screen_text[]` — {text, role, colour_token, size_px, weight, position}
- `decor[]`, `accent` (green/amber), `source_line` (bool), `disclaimer` (bool)
- `transition_in`, `transition_out`, `continuity_from`
- `assets_to_create[]` — {asset_id, kind: illustration | map | icon | photo_cutout | chart, prompt_or_spec}
- `qc_flags[]` — the output of the §3 checks
