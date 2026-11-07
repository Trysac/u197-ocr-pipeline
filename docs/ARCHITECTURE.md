# Architecture

`ocr_pipeline` implements an extensible, pure-Python OCR preprocessing and execution pipeline without third-party dependencies.

## Components

1. **Decoders (`load_image`)**:
   - PNM parser (P2 ASCII grayscale, P5 binary grayscale, P3 ASCII RGB, P6 binary RGB).
   - PNG parser supporting 8-bit Gray, Gray+Alpha, RGB, and RGBA with all 5 filter types (None, Sub, Up, Average, Paeth) and alpha compositing over white.
   - Hard limit enforcement on byte length and maximum pixel count before memory allocation.

2. **Preprocessing (`preprocess`, `rotate`)**:
   - `invert_if_dark`: Inverts images if overall luminosity mean < 128.
   - `median3`: 3x3 median filter with border edge replication.
   - `otsu`: Fast histogram-based thresholding calculating optimal between-class variance.
   - `deskew`: Angular scan using row projection variance on dark pixels with deterministic tie-breaking.

3. **Line Assembly & Execution (`run_ocr`)**:
   - Calls injected `OcrEngine` implementing `recognize(GrayImage) -> Sequence[Word]`.
   - Filters low-confidence words (< `min_confidence`).
   - Groups words into reading lines based on vertical overlap against the line's median height.
   - Orders lines vertically and words horizontally within each line.
   - Dispatches step events without leaking sensitive recognized text.
