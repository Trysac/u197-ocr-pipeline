# ocr-pipeline

[![PyPI version](https://img.shields.io/pypi/v/trysac-ocr-pipeline)](https://pypi.org/project/trysac-ocr-pipeline/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-ocr-pipeline)](LICENSE)
[![CI](https://github.com/Trysac/u197-ocr-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u197-ocr-pipeline/actions/workflows/ci.yml)

OCR pipeline core: decode PNM and PNG to grayscale, median, Otsu and deskew preprocessing, an injected OCR engine port and line assembly

## Problem

Describe in two or three sentences the concrete problem this utility solves, who hits it, and what
they do today without it. Replace this paragraph when the first real feature lands.

## Install

```sh
pip install trysac-ocr-pipeline
# or
uv add trysac-ocr-pipeline
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
import ocr_pipeline

# normalize() trims surrounding whitespace and lower-cases the text.
print(ocr_pipeline.normalize("  Hello World "))  # -> "hello world"

# Blank input is rejected explicitly instead of returning an empty string.
try:
    ocr_pipeline.normalize("   ")
except ValueError as exc:
    print(f"rejected: {exc}")
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
