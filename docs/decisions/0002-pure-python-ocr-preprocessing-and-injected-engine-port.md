# Pure-Python OCR Preprocessing and Injected Engine Port

Status: accepted (generated, pending human review)

## Context

OCR preprocessing pipelines often require heavy native dependencies such as OpenCV or PIL, making installation fragile and testing difficult. Furthermore, coupling directly to external binaries makes pipeline steps hard to test in isolation.

## Decision

Implement decoding for PNM and non-interlaced PNG and image transforms (median filter, Otsu binarization, rotation, deskew) purely in standard library Python, wrapping OCR engines behind a structural Protocol.

## Consequences

Zero external runtime dependencies, 100% deterministic test coverage with FakeOcr, and seamless integration with external OCR engines without environment locks.
