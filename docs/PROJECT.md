# Project record

## Objective

OCR pipeline core: decode PNM and PNG to grayscale, median, Otsu and deskew preprocessing, an injected OCR engine port and line assembly.

## Scope

- Pure-Python PNM and PNG decoding to 8-bit grayscale.
- Image preprocessing pipeline (invert, median, Otsu, deskew).
- Injected OCR engine protocol and result assembly (bounding box filtering, line reordering).

## Out-of-scope

- Shipping concrete engine adapters (e.g., Tesseract).
- Layout analysis (tables, images, etc.).
- Color output, PDF input, and multi-language support.

## Factory run u197 (measured from API usage)

| Phase | Model | Batch | Input | Output | Cache write | Cache read | Cost (USD) |
|---|---|---|---|---|---|---|---|
| generation | gemini-3.8-flash | yes | 4451 | 14332 | 0 | 0 | 0.0285 |
| fix-1 | gemini-3.8-flash | no | 23421 | 532 | 0 | 51242 | 0.0234 |
| fix-2 | gemini-3.8-flash | no | 41521 | 554 | 0 | 159245 | 0.0452 |
| audit | gemini-3.1-flash-lite | no | 47023 | 278 | 0 | 116223 | 0.0151 |
| docs | gemini-3.1-flash-lite | no | 140185 | 1414 | 0 | 0 | 0.0372 |

Total: $0.1494. Prices: Anthropic list prices.
