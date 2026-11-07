"""OCR pipeline core: pure-Python image decoding, preprocessing, engine port, and line assembly."""

from __future__ import annotations

import logging
import math
import struct
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = [
    "ErrorCode",
    "FakeOcr",
    "GrayImage",
    "OcrEngine",
    "OcrError",
    "OcrResult",
    "Preprocessed",
    "Step",
    "StepEvent",
    "Word",
    "__version__",
    "load_image",
    "preprocess",
    "rotate",
    "run_ocr",
]

__version__ = "0.1.0"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

ErrorCode = Literal["unsupported_image", "too_large", "corrupt", "invalid_option", "engine_failed"]
Step = Literal["invert_if_dark", "median3", "otsu", "deskew"]
VALID_STEPS: frozenset[str] = frozenset({"invert_if_dark", "median3", "otsu", "deskew"})


class OcrError(Exception):
    """Base exception for all ocr_pipeline errors with a stable machine-readable code."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        # Ensure error messages never contain excessive data
        trimmed_msg = message[:200] if len(message) > 200 else message
        super().__init__(f"[{code}] {trimmed_msg}")
        self.code: ErrorCode = code
        self.message: str = trimmed_msg


@dataclass(frozen=True, slots=True)
class GrayImage:
    """Immutable 8-bit grayscale image in row-major order."""

    width: int
    height: int
    pixels: bytes

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise OcrError(
                "invalid_option", f"Invalid image dimensions: {self.width}x{self.height}"
            )
        expected_len = self.width * self.height
        if len(self.pixels) != expected_len:
            raise OcrError(
                "corrupt",
                f"Pixel buffer mismatch: expected {expected_len} bytes, got {len(self.pixels)}",
            )


@dataclass(frozen=True, slots=True)
class Word:
    """Recognized word bounding box, confidence, and text."""

    text: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (left, top, width, height)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 100.0):
            raise OcrError(
                "invalid_option", f"Confidence {self.confidence} out of range [0.0, 100.0]"
            )


@dataclass(frozen=True, slots=True)
class Preprocessed:
    """Result of image preprocessing pipeline."""

    image: GrayImage
    threshold: int | None = None
    angle: float | None = None


@dataclass(frozen=True, slots=True)
class StepEvent:
    """Notification event emitted during OCR processing."""

    step: str
    details: dict[str, Any]


@dataclass(frozen=True, slots=True)
class OcrResult:
    """Complete OCR run output including assembled text, words, and metadata."""

    text: str
    lines: tuple[str, ...]
    words: tuple[Word, ...]
    dropped: int
    preprocessing: Preprocessed


@runtime_checkable
class OcrEngine(Protocol):
    """Protocol defining the engine interface for optical character recognition."""

    def recognize(self, image: GrayImage) -> Sequence[Word]:
        """Recognize words in the given grayscale image."""
        ...


class FakeOcr:
    """Test double recording recognition requests and returning pre-set words."""

    def __init__(self, words: Sequence[Word] | None = None) -> None:
        self._words: list[Word] = list(words) if words is not None else []
        self.calls: list[GrayImage] = []

    def recognize(self, image: GrayImage) -> Sequence[Word]:
        self.calls.append(image)
        return self._words


def _rgb_to_gray(r: int, g: int, b: int) -> int:
    return (299 * r + 587 * g + 114 * b + 500) // 1000


def _decode_pnm(data: bytes, max_pixels: int) -> GrayImage:
    idx = 0
    n = len(data)

    def next_token() -> bytes:
        nonlocal idx
        while idx < n:
            b = data[idx]
            if b in b" \t\r\n":
                idx += 1
                continue
            if b == ord(b"#"):
                while idx < n and data[idx] not in b"\r\n":
                    idx += 1
                continue
            break
        if idx >= n:
            raise OcrError("corrupt", "Unexpected end of PNM header")
        start = idx
        while idx < n and data[idx] not in b" \t\r\n#":
            idx += 1
        return data[start:idx]

    magic = next_token()
    if magic not in (b"P2", b"P3", b"P5", b"P6"):
        raise OcrError(
            "unsupported_image", f"Unsupported PNM magic {magic.decode('latin1', 'replace')}"
        )

    try:
        width = int(next_token().decode("ascii"))
        height = int(next_token().decode("ascii"))
        maxval = int(next_token().decode("ascii"))
    except ValueError as err:
        raise OcrError("corrupt", "Malformed PNM header integers") from err

    if width <= 0 or height <= 0:
        raise OcrError("corrupt", f"Invalid dimensions {width}x{height}")
    if width * height > max_pixels:
        raise OcrError(
            "too_large", f"Image pixel count {width * height} exceeds limit {max_pixels}"
        )
    if maxval < 1 or maxval > 255:
        raise OcrError("unsupported_image", f"PNM maxval {maxval} not supported (must be 1-255)")

    pixel_count = width * height

    def scale(val: int) -> int:
        return (val * 255 + (maxval // 2)) // maxval if maxval != 255 else val

    if magic == b"P2":
        pixels_out = bytearray(pixel_count)
        for i in range(pixel_count):
            val = int(next_token().decode("ascii"))
            pixels_out[i] = scale(val)
        return GrayImage(width, height, bytes(pixels_out))

    if magic == b"P3":
        pixels_out = bytearray(pixel_count)
        for i in range(pixel_count):
            r = scale(int(next_token().decode("ascii")))
            g = scale(int(next_token().decode("ascii")))
            b = scale(int(next_token().decode("ascii")))
            pixels_out[i] = _rgb_to_gray(r, g, b)
        return GrayImage(width, height, bytes(pixels_out))

    # Binary formats P5, P6: skip single whitespace char after maxval
    if idx < n and data[idx] in b" \t\r\n":
        idx += 1

    if magic == b"P5":
        raw_data = data[idx:]
        if len(raw_data) < pixel_count:
            raise OcrError("corrupt", f"Expected {pixel_count} bytes, got {len(raw_data)}")
        raw_pixels = raw_data[:pixel_count]
        if maxval == 255:
            return GrayImage(width, height, raw_pixels)
        out = bytes(scale(b) for b in raw_pixels)
        return GrayImage(width, height, out)

    # magic == b"P6"
    expected_raw = pixel_count * 3
    raw_data = data[idx:]
    if len(raw_data) < expected_raw:
        raise OcrError("corrupt", f"Expected {expected_raw} bytes for P6, got {len(raw_data)}")
    pixels_out = bytearray(pixel_count)
    p_idx = 0
    for i in range(pixel_count):
        r = scale(raw_data[p_idx])
        g = scale(raw_data[p_idx + 1])
        b = scale(raw_data[p_idx + 2])
        pixels_out[i] = _rgb_to_gray(r, g, b)
        p_idx += 3
    return GrayImage(width, height, bytes(pixels_out))


def _paeth_predict(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _decode_png(data: bytes, max_pixels: int) -> GrayImage:
    if len(data) < 8 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise OcrError("corrupt", "Invalid PNG signature")

    idx = 8
    n = len(data)
    ihdr_found = False
    width = 0
    height = 0
    color_type = 0
    idat_chunks: list[bytes] = []

    while idx < n:
        if idx + 8 > n:
            raise OcrError("corrupt", "Truncated PNG chunk header")
        length, chunk_type = struct.unpack(">I4s", data[idx : idx + 8])
        idx += 8
        if idx + length + 4 > n:
            raise OcrError("corrupt", "Truncated PNG chunk data")
        chunk_data = data[idx : idx + length]
        idx += length + 4  # skip data + 4 bytes CRC

        if chunk_type == b"IHDR":
            if length != 13:
                raise OcrError("corrupt", "Invalid IHDR length")
            width, height, bit_depth, color_type, comp_m, filter_m, interlace_m = struct.unpack(
                ">IIBBBBB", chunk_data
            )
            ihdr_found = True
            if width <= 0 or height <= 0:
                raise OcrError("corrupt", f"Invalid PNG dimensions: {width}x{height}")
            if width * height > max_pixels:
                raise OcrError(
                    "too_large", f"Image pixel count {width * height} exceeds limit {max_pixels}"
                )
            if interlace_m != 0:
                raise OcrError("unsupported_image", "Interlaced PNG is not supported")
            if bit_depth != 8:
                raise OcrError("unsupported_image", f"PNG bit depth {bit_depth} not supported")
            if comp_m != 0 or filter_m != 0:
                raise OcrError("corrupt", "Unsupported compression or filter method in PNG")
            # color types: 0 (grayscale), 2 (RGB), 4 (gray+alpha), 6 (RGBA)
            if color_type not in (0, 2, 4, 6):
                raise OcrError("unsupported_image", f"PNG color type {color_type} not supported")
        elif chunk_type == b"IDAT":
            if not ihdr_found:
                raise OcrError("corrupt", "IDAT chunk before IHDR")
            idat_chunks.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    if not ihdr_found:
        raise OcrError("corrupt", "PNG missing IHDR chunk")
    if not idat_chunks:
        raise OcrError("corrupt", "PNG missing IDAT data")

    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    row_bytes = width * channels
    expected_decomp_size = height * (1 + row_bytes)

    compressed_stream = b"".join(idat_chunks)
    try:
        # max_length cap of expected size + height per spec
        decomp_obj = zlib.decompressobj()
        decomp = decomp_obj.decompress(compressed_stream, max_length=expected_decomp_size + height)
    except Exception as err:
        raise OcrError("corrupt", "Failed to decompress PNG IDAT data") from err

    if len(decomp) != expected_decomp_size:
        raise OcrError(
            "corrupt", f"Decompressed PNG data length {len(decomp)} != {expected_decomp_size}"
        )

    # Unfilter scanlines
    unfiltered = bytearray(height * row_bytes)
    stride = 1 + row_bytes
    bpp = channels

    for y in range(height):
        row_start = y * stride
        filter_type = decomp[row_start]
        curr_row_offset = y * row_bytes
        prev_row_offset = (y - 1) * row_bytes if y > 0 else None

        for x in range(row_bytes):
            raw_val = decomp[row_start + 1 + x]
            left_val = unfiltered[curr_row_offset + x - bpp] if x >= bpp else 0
            up_val = unfiltered[prev_row_offset + x] if prev_row_offset is not None else 0
            up_left_val = (
                unfiltered[prev_row_offset + x - bpp]
                if (prev_row_offset is not None and x >= bpp)
                else 0
            )

            if filter_type == 0:  # None
                val = raw_val
            elif filter_type == 1:  # Sub
                val = (raw_val + left_val) & 0xFF
            elif filter_type == 2:  # Up
                val = (raw_val + up_val) & 0xFF
            elif filter_type == 3:  # Average
                val = (raw_val + ((left_val + up_val) >> 1)) & 0xFF
            elif filter_type == 4:  # Paeth
                val = (raw_val + _paeth_predict(left_val, up_val, up_left_val)) & 0xFF
            else:
                raise OcrError("corrupt", f"Unknown PNG filter type: {filter_type}")

            unfiltered[curr_row_offset + x] = val

    # Convert channels to Grayscale
    pixel_count = width * height
    pixels_out = bytearray(pixel_count)
    p_idx = 0
    unf_idx = 0

    if color_type == 0:  # Gray
        return GrayImage(width, height, bytes(unfiltered))
    elif color_type == 2:  # RGB
        for _ in range(pixel_count):
            r = unfiltered[unf_idx]
            g = unfiltered[unf_idx + 1]
            b = unfiltered[unf_idx + 2]
            pixels_out[p_idx] = _rgb_to_gray(r, g, b)
            p_idx += 1
            unf_idx += 3
    elif color_type == 4:  # Gray + Alpha (composite on white background 255)
        for _ in range(pixel_count):
            gray = unfiltered[unf_idx]
            alpha = unfiltered[unf_idx + 1]
            # out = (gray * alpha + 255 * (255 - alpha) + 127) // 255
            comp = (gray * alpha + 255 * (255 - alpha) + 127) // 255
            pixels_out[p_idx] = comp
            p_idx += 1
            unf_idx += 2
    elif color_type == 6:  # RGBA (composite on white background 255)
        for _ in range(pixel_count):
            r = unfiltered[unf_idx]
            g = unfiltered[unf_idx + 1]
            b = unfiltered[unf_idx + 2]
            alpha = unfiltered[unf_idx + 3]
            # composite RGB over 255
            cr = (r * alpha + 255 * (255 - alpha) + 127) // 255
            cg = (g * alpha + 255 * (255 - alpha) + 127) // 255
            cb = (b * alpha + 255 * (255 - alpha) + 127) // 255
            pixels_out[p_idx] = _rgb_to_gray(cr, cg, cb)
            p_idx += 1
            unf_idx += 4

    return GrayImage(width, height, bytes(pixels_out))


def load_image(
    data: bytes, *, max_pixels: int = 2_000_000, max_bytes: int = 50_000_000
) -> GrayImage:
    """Load an image from raw PNM or PNG bytes into a GrayImage."""
    if len(data) > max_bytes:
        raise OcrError("too_large", f"Input byte length {len(data)} exceeds max_bytes {max_bytes}")
    if not data:
        raise OcrError("corrupt", "Input data is empty")

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return _decode_png(data, max_pixels)
    if data[:2] in (b"P2", b"P3", b"P5", b"P6"):
        return _decode_pnm(data, max_pixels)

    raise OcrError("unsupported_image", "Image format not recognized or unsupported")


def rotate(img: GrayImage, degrees: float) -> GrayImage:
    """Rotate img about its centre by degrees (positive = counter-clockwise)."""
    if degrees == 0.0 or degrees == -0.0:
        return img

    w = img.width
    h = img.height
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    # Positive angle = counter-clockwise rotation of the image.
    # To map dest pixel (x, y) back to source pixel (sx, sy), rotate by -theta.
    # Using standard math coord where y points up, screen y points down, so angle sign:
    rad = math.radians(degrees)
    cos_theta = math.cos(rad)
    sin_theta = math.sin(rad)

    src_bytes = img.pixels
    out_bytes = bytearray(w * h)

    out_idx = 0
    for y in range(h):
        dy = y - cy
        for x in range(w):
            dx = x - cx
            # Inverse transformation (counter-clockwise rotate image by theta means
            # forward map: (x', y') = (cx + dx*cos - dy*sin, cy + dx*sin + dy*cos)
            # inverse map: (sx, sy) = (cx + dx*cos + dy*sin, cy - dx*sin + dy*cos))
            sx = cx + dx * cos_theta + dy * sin_theta
            sy = cy - dx * sin_theta + dy * cos_theta

            # Nearest neighbor sampling
            ix = round(sx)
            iy = round(sy)

            if 0 <= ix < w and 0 <= iy < h:
                out_bytes[out_idx] = src_bytes[iy * w + ix]
            else:
                out_bytes[out_idx] = 255  # background 255
            out_idx += 1

    return GrayImage(w, h, bytes(out_bytes))


def _otsu_threshold(img: GrayImage) -> int:
    hist = [0] * 256
    for b in img.pixels:
        hist[b] += 1

    total_pixels = img.width * img.height
    if total_pixels == 0:
        return 0

    # If image is uniform or all white
    if hist[255] == total_pixels:
        return 0

    sum_total = 0
    for t in range(256):
        sum_total += t * hist[t]

    sum_b = 0
    w_b = 0
    max_variance = -1.0
    best_threshold = 0

    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total_pixels - w_b
        if w_f == 0:
            break

        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f

        diff = m_b - m_f
        variance = w_b * w_f * (diff * diff)

        # First maximum on ties
        if variance > max_variance:
            max_variance = variance
            best_threshold = t

    return best_threshold


def _deskew_image(img: GrayImage, max_angle: float, angle_step: float) -> tuple[GrayImage, float]:
    # Check for uniform / all white image
    if all(p >= 128 for p in img.pixels):
        return img, 0.0

    # Candidate angles from -max_angle to +max_angle by angle_step
    angles: list[float] = []
    curr = -max_angle
    while curr <= max_angle + 1e-9:
        angles.append(round(curr, 6))
        curr += angle_step

    best_variance = -1.0
    best_angle = 0.0

    w = img.width
    h = img.height

    for a in angles:
        candidate_img = rotate(img, a)
        pix = candidate_img.pixels

        # Variance of row sums of dark pixels (< 128)
        row_sums = [0] * h
        for r in range(h):
            row_start = r * w
            count = 0
            for c in range(w):
                if pix[row_start + c] < 128:
                    count += 1
            row_sums[r] = count

        mean_row = sum(row_sums) / h
        var = sum((x - mean_row) ** 2 for x in row_sums) / h

        # Tie breaking: maximum variance, then smaller abs(angle), then negative first
        if var > best_variance + 1e-9:
            best_variance = var
            best_angle = a
        elif abs(var - best_variance) <= 1e-9:
            # Compare abs(angle)
            if abs(a) < abs(best_angle) - 1e-9:
                best_variance = var
                best_angle = a
            elif abs(abs(a) - abs(best_angle)) <= 1e-9 and a < best_angle:
                # Negative first
                best_variance = var
                best_angle = a

    return rotate(img, best_angle), best_angle


def preprocess(
    img: GrayImage,
    steps: Sequence[Step] = ("invert_if_dark", "median3", "otsu", "deskew"),
    *,
    max_angle: float = 5.0,
    angle_step: float = 0.5,
    on_step: Callable[[StepEvent], None] | None = None,
) -> Preprocessed:
    """Run ordered preprocessing steps on a GrayImage."""
    for s in steps:
        if s not in VALID_STEPS:
            raise OcrError("invalid_option", f"Invalid preprocessing step: {s}")
    if max_angle < 0 or angle_step <= 0:
        raise OcrError("invalid_option", "max_angle must be >= 0 and angle_step > 0")

    current = img
    threshold_result: int | None = None
    angle_result: float | None = None

    for step in steps:
        if step == "invert_if_dark":
            total_sum = sum(current.pixels)
            mean = total_sum / len(current.pixels)
            inverted = False
            if mean < 128:
                current = GrayImage(
                    current.width, current.height, bytes(255 - b for b in current.pixels)
                )
                inverted = True
            if on_step is not None:
                on_step(StepEvent("invert_if_dark", {"mean": mean, "inverted": inverted}))

        elif step == "median3":
            w = current.width
            h = current.height
            pix = current.pixels
            out = bytearray(w * h)
            for y in range(h):
                for x in range(w):
                    window: list[int] = []
                    for dy in (-1, 0, 1):
                        ny = min(max(y + dy, 0), h - 1)
                        row_idx = ny * w
                        for dx in (-1, 0, 1):
                            nx = min(max(x + dx, 0), w - 1)
                            window.append(pix[row_idx + nx])
                    window.sort()
                    out[y * w + x] = window[4]
            current = GrayImage(w, h, bytes(out))
            if on_step is not None:
                on_step(StepEvent("median3", {"kernel_size": 3}))

        elif step == "otsu":
            th = _otsu_threshold(current)
            threshold_result = th
            current = GrayImage(
                current.width,
                current.height,
                bytes(0 if b <= th else 255 for b in current.pixels),
            )
            if on_step is not None:
                on_step(StepEvent("otsu", {"threshold": th}))

        elif step == "deskew":
            current, ang = _deskew_image(current, max_angle, angle_step)
            angle_result = ang
            if on_step is not None:
                on_step(StepEvent("deskew", {"angle": ang}))

    return Preprocessed(image=current, threshold=threshold_result, angle=angle_result)


def _assemble_lines(words: Sequence[Word]) -> tuple[str, tuple[str, ...]]:
    if not words:
        return "", ()

    # Sort words by vertical coordinate (top) then left
    sorted_words = sorted(words, key=lambda w: (w.bbox[1], w.bbox[0]))

    # Structure to keep lines: list of list of Word
    lines: list[list[Word]] = []

    for word in sorted_words:
        w_top = word.bbox[1]
        w_height = word.bbox[3]
        w_center = w_top + w_height / 2.0

        matched_line: list[Word] | None = None
        for line in lines:
            # Line center and median height
            line_heights = sorted(lw.bbox[3] for lw in line)
            med_height = line_heights[len(line_heights) // 2]
            line_centers = [lw.bbox[1] + lw.bbox[3] / 2.0 for lw in line]
            line_center = sum(line_centers) / len(line_centers)

            if abs(w_center - line_center) <= (med_height / 2.0):
                matched_line = line
                break

        if matched_line is not None:
            matched_line.append(word)
        else:
            lines.append([word])

    # Sort lines by average vertical center
    lines.sort(key=lambda line: sum(lw.bbox[1] + lw.bbox[3] / 2.0 for lw in line) / len(line))

    text_lines: list[str] = []
    for line in lines:
        line.sort(key=lambda w: w.bbox[0])
        text_lines.append(" ".join(w.text for w in line))

    full_text = "\n".join(text_lines)
    return full_text, tuple(text_lines)


def run_ocr(
    img: GrayImage,
    engine: OcrEngine,
    *,
    steps: Sequence[Step] = ("invert_if_dark", "median3", "otsu", "deskew"),
    min_confidence: float = 60.0,
    on_event: Callable[[StepEvent], None] | None = None,
) -> OcrResult:
    """Preprocess an image, run OCR engine, filter confidence, and assemble text lines."""
    if not (0.0 <= min_confidence <= 100.0):
        raise OcrError("invalid_option", f"min_confidence {min_confidence} must be in [0.0, 100.0]")

    preprocessed = preprocess(img, steps=steps, on_step=on_event)

    try:
        words_seq = engine.recognize(preprocessed.image)
    except Exception as err:
        raise OcrError("engine_failed", f"Engine recognition failed: {err}") from err

    accepted_words: list[Word] = []
    dropped = 0
    for w in words_seq:
        if w.confidence >= min_confidence:
            accepted_words.append(w)
        else:
            dropped += 1

    if on_event is not None:
        on_event(
            StepEvent(
                "ocr",
                {
                    "total_words": len(words_seq),
                    "accepted_words": len(accepted_words),
                    "dropped_words": dropped,
                },
            )
        )

    full_text, text_lines = _assemble_lines(accepted_words)
    return OcrResult(
        text=full_text,
        lines=text_lines,
        words=tuple(accepted_words),
        dropped=dropped,
        preprocessing=preprocessed,
    )
