# Security notes

## Threat model

### Defended

- **Memory Exhaustion**: The library enforces strict `max_pixels` and `max_bytes` limits during decoding of PNM and PNG data before large allocations occur. 
- **Arbitrary Code Execution**: No `eval` or deserialization (e.g., `pickle`) is used. Data is parsed via structured byte manipulation.
- **Input Validation**: All pixel buffer dimensions, color depths, and confidence scores are validated. Invalid inputs raise `OcrError` with machine-readable codes.

### Not defended

- **Engine-Side Vulnerabilities**: If an `OcrEngine` implementation is wrapping an external binary (like Tesseract), the security of that specific external process is the responsibility of that implementation; the library simply delegates the `recognize` call.

To report a vulnerability, please contact the maintainers.