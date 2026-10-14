# 03 — Brand Guidelines (ZO IPO LF)

## 1. Brand marks (always on screen)

| Element | Rule |
|---|---|
| **Zerodha logo** | Top-left on **every** frame (board and footage). Mark + wordmark "ZERODHA", ~214×28px at x24, y26 (1080p). **Blue `#387ED1` on board, white on footage.** Never animated, never moved, never covered. |
| **Source line** | `Source: RHP` bottom-left, x38, ~28px Inter Regular. Ink on board, white on footage. Show it whenever an RHP fact is on screen, and keep it through the presenter shots of that section. |
| **SEBI disclaimer** | `Zerodha Broking Ltd.: Member of NSE, BSE & MCX – SEBI Registration no.: INZ000031633`. Bottom-left under Source, 18–20px. Required on the **opening title/chapter graphic** (the first ~8s) and on any cold-open footage carrying the title. Use the en-dash (–) exactly as written. |
| **Presenter wardrobe** | Black "invest, india!" tee (the series identity). Keep graphics off the chest text where possible. |

## 2. Colour tokens (from the "ZO IPO LF Theme" sheet, verified against frames)

```css
:root {
  /* primary palette — the brand signature */
  --zo-white:        #FFFFFF;  /* "The clean, timeless base" — every board */
  --zo-blue:         #387ED1;  /* "Zerodha Blue — the sacred core": company names, key phrases, primary segment, bars, states */
  --zo-sunshine:     #FFA412;  /* "Sunshine Yellow — warm balancing accent": second segment, chips, sparkles */
  --zo-deep-grey:    #424242;  /* "The bridge between warm and cool": body ink */

  /* secondary palette — use minimally, mainly for chart proportions */
  --zo-green:        #4CAF52;  /* accent dash, trees, positive */
  --zo-gold:         #FFC02E;
  --zo-orange:       #F07335;  /* risk, negative, third segment */
  --zo-purple:       #A964FF;  /* triangles, fourth segment */

  /* working tints measured in the approved videos */
  --zo-ink:          #222222;  /* headings (#2E2E2E measured) */
  --zo-blue-tint:    #E5F1FD;  /* sun circles, map land, primary cards */
  --zo-blue-cal:     #DEEAF8;  /* calendar header/border */
  --zo-amber-tint:   #FEF3DD;  /* building sun circles */
  --zo-date-hl:      #FBE1B3;  /* selected calendar cell */
  --zo-chip:         #F5A319;  /* Open/Closed chip */
  --zo-state:        #589AE6;  /* filled states on maps */
  --zo-bar-rest:     #CFCFCF;  /* bar remainder (white board) */
  --zo-bar-rest-dk:  #888888;  /* bar remainder (split board) */
  --zo-card-grey:    #E4E4E4;
  --zo-dot:          #DDE1E5;  /* dot grid */
  --zo-ost-blue:     #0F7CFC;  /* OST number circles (brighter, glows on footage) */
  --zo-ost-card:     #1B242B;  /* dark glass OST card */
  --zo-ost-border:   #2A5BA8;
}
```

**Ratio:** white space dominates, then blue as the hero colour, sunshine yellow as the single warm accent,
and secondary colours used sparingly. A frame should never show more than **blue + one accent** besides neutrals
(chart segments excepted).

## 3. Typeface
Inter (Regular 400, Medium 500, SemiBold 600, Bold 700). Load from Google Fonts. No other faces.
Exception: brand names inside illustrations may use the company's own lettering.

## 4. Naming and copy rules

- **Company name exactly as in the RHP** on its first appearance: "Jindal Supreme India Limited IPO",
  "Sonaselection India Limited IPO". After that, use the short form consistently ("Hero Motors", not "Hero motors").
- **Title Case** for company names and headings that are names. **Sentence case** for headings that are statements
  ("Manufactures and processes", "Executing the entire production chain", "Revenue diversification").
- Pick **one** colour treatment for the company name per video (e.g. full blue, or "**Hero** Motors" two-tone) and
  keep it everywhere.
- **Third-party brand names spelt correctly** (Ducati, not "Duccati"). Check them against official spellings.
- **Numbers:** exactly as in the RHP. `%` without a space. Approximations use `~` ("~70%").
  Indian units (₹ crore, lakh). Fiscal years as `FY 2024` or `FY2024`; pick one per video.
  Units with a hyphen when used as an adjective ("16-acre", "1.2-megawatt").
- **Spelling:** Indian/British English ("galvanised", "metres"). Don't mix in "galvanization"/"meters"
  within one video.
- **Brevity:** heading ≤6 words; label ≤4 words a line, ≤2 lines; no full sentences on the board except
  annotation lines (≤8 words).
- **No investment advice language** on screen ("buy", "must subscribe", "guaranteed"). Facts and sources only.
- **Dates:** `16 Sep` + weekday (`Wednesday`), with chips `Open` / `Closed`.

## 5. Do / Don't

| Do | Don't |
|---|---|
| One hero per board, information attached to it | Float text with no object |
| Blue for the company and key phrase | Colour whole sentences blue |
| Real grayscale photo for named promoters | Illustrate a real person |
| Keep logo + Source in their fixed places | Let graphics, dashes or presenter cut-outs overlap Source or the logo |
| Ghost previous items in a carousel | Delete context the viewer still needs |
| Use the theme's decor kit sparingly | Add emojis, stock icons or drop shadows outside the style |
