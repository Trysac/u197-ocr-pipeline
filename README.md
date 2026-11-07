# ocr-pipeline

[![CI](https://github.com/Trysac/u197-ocr-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u197-ocr-pipeline/actions/workflows/ci.yml)

OCR pipeline core: pure-Python decoding of PNM and PNG to grayscale, median filtering, Otsu binarization, deskewing, line assembly, and an injected OCR engine port.

## The Problem

OCR quality depends more on preprocessing than on the engine: skewed, noisy or dark scans produce garbage, and pipelines glue ad hoc image steps to an engine call with no tests. Engines such as Tesseract are external binaries that make tests slow and platform-dependent. This library implements the preprocessing steps as tested pure functions on grayscale images and calls the engine through a small injected port.

## Installation

```bash
pip install trysac-ocr-pipeline
```

## Usage

```python
from ocr_pipeline import GrayImage, Word, FakeOcr, run_ocr

img = GrayImage(width=100, height=20, pixels=bytes([255] * 2000))
engine = FakeOcr([Word("Hello", 98.0, (0, 0, 50, 18)), Word("World", 95.0, (55, 0, 40, 18))])
result = run_ocr(img, engine, steps=("invert_if_dark", "otsu"))
print(result.text)
```

## API Summary

- `load_image(data: bytes, *, max_pixels: int = 2_000_000, max_bytes: int = 50_000_000) -> GrayImage`: Decodes PNM (P2, P3, P5, P6) or PNG (8-bit Gray, RGB, RGBA) images. Raises `unsupported_image` for unsupported types or `too_large` if limits are exceeded.
- `rotate(img: GrayImage, degrees: float) -> GrayImage`: Rotates an image about its center (nearest neighbor, positive = counter-clockwise).
- `preprocess(img: GrayImage, steps: Sequence[Step] = (...), *, max_angle: float = 5.0, angle_step: float = 0.5) -> Preprocessed`: Runs an ordered sequence of preprocessing steps. Raises `invalid_option` for bad steps or parameters.
- `OcrEngine(Protocol)`: Interface for custom engine adapters.
- `run_ocr(...) -> OcrResult`: Orchestrates preprocessing, engine recognition, confidence filtering, and line assembly. Raises `engine_failed` if the engine raises an exception.

## Why not Tesseract directly?

This library provides tested, isolated, pure-Python preprocessing steps and an engine-agnostic port, avoiding platform-dependent binary dependencies and providing deterministic control over the image stream.

See [docs/SECURITY.md](docs/SECURITY.md) for vulnerability reporting.