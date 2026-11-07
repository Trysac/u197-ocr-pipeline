# Testing

## Running

```sh
uv run pytest --cov=ocr_pipeline
```

## Coverage

Tests are located in `tests/test_ocr_pipeline.py`. They cover:

- **PNM Decoding**: Validation of P2, P5, P3, and P6 formats, handling of comments, and scaling of maxval < 255.
- **PNG Decoding**: Support for 8-bit Gray, Gray+Alpha, RGB, and RGBA; verification of filter types 0-4 (Paeth); detection of interlaced PNGs.
- **Preprocessing**: Functional verification of `invert_if_dark`, `median3` (including edge replication), `otsu` (histogram maximization), and `deskew` (variance of row sums).
- **Line Assembly**: Sorting by bbox top, grouping by median height overlap, and reordering within lines by left coordinate.
- **Engine Integration**: Verification of `FakeOcr` call recording, error wrapping for engine failures, and confidence filtering/event emission.