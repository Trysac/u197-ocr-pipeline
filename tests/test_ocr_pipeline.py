"""Tests for the ocr_pipeline package."""

from __future__ import annotations

import struct
import zlib

import pytest

from ocr_pipeline import (
    FakeOcr,
    GrayImage,
    OcrError,
    StepEvent,
    Word,
    load_image,
    preprocess,
    rotate,
    run_ocr,
)


def make_png(
    width: int, height: int, color_type: int, raw_rows: list[bytes], filter_types: list[int]
) -> bytes:
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    crc_ihdr = zlib.crc32(b"IHDR" + header)
    ihdr_chunk = struct.pack(">I", 13) + b"IHDR" + header + struct.pack(">I", crc_ihdr)

    uncompressed = bytearray()
    for f_type, row in zip(filter_types, raw_rows, strict=True):
        uncompressed.append(f_type)
        uncompressed.extend(row)

    compressed = zlib.compress(bytes(uncompressed))
    crc_idat = zlib.crc32(b"IDAT" + compressed)
    idat_chunk = (
        struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", crc_idat)
    )

    iend_crc = zlib.crc32(b"IEND")
    iend_chunk = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)

    return b"\x89PNG\r\n\x1a\n" + ihdr_chunk + idat_chunk + iend_chunk


def test_pnm_load_and_gray_formula() -> None:
    img_p2 = load_image(b"P2\n2 1\n255\n0 255\n")
    assert img_p2 == GrayImage(width=2, height=1, pixels=b"\x00\xff")

    # P6: (255, 0, 0) -> (299*255 + 500)//1000 = 76
    p6_data = b"P6\n1 1\n255\n\xff\x00\x00"
    img_p6 = load_image(p6_data)
    assert img_p6 == GrayImage(width=1, height=1, pixels=b"\x4c")
    assert img_p6.pixels[0] == 76


def test_png_rgb_filters_match_p6() -> None:
    # 2x1 RGB pixels: (255, 0, 0) -> 76, (0, 255, 0) -> (587*255 + 500)//1000 = 150
    # Test PNG filter types 0-4
    p6_equiv = load_image(b"P6\n2 1\n255\n\xff\x00\x00\x00\xff\x00")
    assert p6_equiv.pixels == bytes([76, 150])

    target_bytes = b"\xff\x00\x00\x00\xff\x00"
    # filter 0: raw
    png0 = make_png(2, 1, 2, [target_bytes], [0])
    assert load_image(png0) == p6_equiv

    # filter 1: Sub
    # Byte differences with left pixel:
    # b0..2: 255, 0, 0
    # b3..5: (0-255)&0xff = 1, (255-0)&0xff = 255, (0-0)&0xff = 0
    f1_raw = bytes([255, 0, 0, (0 - 255) & 0xFF, (255 - 0) & 0xFF, 0])
    png1 = make_png(2, 1, 2, [f1_raw], [1])
    assert load_image(png1) == p6_equiv

    # filter 2: Up (height 1, up is 0)
    png2 = make_png(2, 1, 2, [target_bytes], [2])
    assert load_image(png2) == p6_equiv

    # filter 3: Average
    # Height 1: up is 0, so left // 2
    # b0..2: 255, 0, 0
    # b3: 0 - (255>>1) = (0 - 127) & 0xff = 129
    # b4: 255 - (0>>1) = 255
    # b5: 0 - (0>>1) = 0
    f3_raw = bytes([255, 0, 0, (0 - 127) & 0xFF, 255, 0])
    png3 = make_png(2, 1, 2, [f3_raw], [3])
    assert load_image(png3) == p6_equiv

    # filter 4: Paeth
    # Height 1: up=0, up_left=0. Paeth(left, 0, 0) == left
    # Same as Sub
    png4 = make_png(2, 1, 2, [f1_raw], [4])
    assert load_image(png4) == p6_equiv


def test_png_interlaced_and_too_large() -> None:
    # Interlaced PNG
    header_interlaced = struct.pack(">IIBBBBB", 2, 1, 8, 2, 0, 0, 1)
    crc_ihdr = zlib.crc32(b"IHDR" + header_interlaced)
    ihdr_chunk = struct.pack(">I", 13) + b"IHDR" + header_interlaced + struct.pack(">I", crc_ihdr)
    interlaced_png = b"\x89PNG\r\n\x1a\n" + ihdr_chunk
    with pytest.raises(OcrError) as exc_info:
        load_image(interlaced_png)
    assert exc_info.value.code == "unsupported_image"

    # 2_000_001 pixels raises too_large before allocating
    header_big = struct.pack(">IIBBBBB", 2000001, 1, 8, 0, 0, 0, 0)
    crc_big = zlib.crc32(b"IHDR" + header_big)
    ihdr_big = struct.pack(">I", 13) + b"IHDR" + header_big + struct.pack(">I", crc_big)
    big_png = b"\x89PNG\r\n\x1a\n" + ihdr_big
    with pytest.raises(OcrError) as exc_info:
        load_image(big_png)
    assert exc_info.value.code == "too_large"


def test_otsu_bimodal() -> None:
    pixels = bytes([10] * 50 + [200] * 50)
    img = GrayImage(10, 10, pixels)
    prep = preprocess(img, steps=("otsu",))
    assert prep.threshold == 10
    assert prep.image.pixels == bytes([0] * 50 + [255] * 50)


def test_median3_salt_and_pepper() -> None:
    # 5x5 image of 255 with a single 0 at the centre (index 12)
    pix = bytearray([255] * 25)
    pix[12] = 0
    img = GrayImage(5, 5, bytes(pix))
    prep = preprocess(img, steps=("median3",))
    assert prep.image.pixels == bytes([255] * 25)


def test_deskew_and_invert_if_dark() -> None:
    # 200x100 image with 3 2-pixel black horizontal bars
    w, h = 200, 100
    canvas = bytearray([255] * (w * h))
    for bar_y in (20, 21, 50, 51, 80, 81):
        for x in range(w):
            canvas[bar_y * w + x] = 0
    img = GrayImage(w, h, bytes(canvas))

    # Angle == 0.0
    prep0 = preprocess(img, steps=("deskew",))
    assert prep0.angle == 0.0

    # rotate(that_image, 2.0)
    rot_img = rotate(img, 2.0)
    prep_rot = preprocess(rot_img, steps=("deskew",))
    assert prep_rot.angle is not None
    assert abs(prep_rot.angle + 2.0) <= 0.5

    # invert_if_dark on mean < 128
    dark_img = GrayImage(2, 2, bytes([10, 20, 30, 40]))
    prep_inv = preprocess(dark_img, steps=("invert_if_dark",))
    assert prep_inv.image.pixels == bytes([245, 235, 225, 215])

    bright_img = GrayImage(2, 2, bytes([200, 210, 220, 230]))
    prep_no_inv = preprocess(bright_img, steps=("invert_if_dark",))
    assert prep_no_inv.image.pixels == bright_img.pixels


def test_run_ocr_and_line_assembly() -> None:
    dummy_img = GrayImage(100, 100, bytes([255] * 10000))
    fake = FakeOcr(
        [
            Word("world", 90.0, (60, 10, 40, 10)),
            Word("Hello", 95.0, (0, 12, 50, 10)),
            Word("noise", 20.0, (0, 40, 20, 10)),
        ]
    )
    res = run_ocr(dummy_img, fake)
    assert res.text == "Hello world"
    assert res.dropped == 1
    assert len(fake.calls) == 1
    assert fake.calls[0] == res.preprocessing.image


def test_error_handling_and_validation() -> None:
    dummy_img = GrayImage(2, 2, bytes([255] * 4))

    class FailingEngine:
        def recognize(self, image: GrayImage):
            raise RuntimeError("Crash in engine")

    with pytest.raises(OcrError) as exc_info:
        run_ocr(dummy_img, FailingEngine())
    assert exc_info.value.code == "engine_failed"
    assert isinstance(exc_info.value.__cause__, RuntimeError)

    # invalid option: steps=('blur',)
    with pytest.raises(OcrError) as exc_info:
        preprocess(dummy_img, steps=("blur",))  # type: ignore[arg-type]
    assert exc_info.value.code == "invalid_option"

    # invalid option: min_confidence=101
    with pytest.raises(OcrError) as exc_info:
        run_ocr(dummy_img, FakeOcr([]), min_confidence=101.0)
    assert exc_info.value.code == "invalid_option"


def test_events_emission() -> None:
    events: list[StepEvent] = []
    dummy_img = GrayImage(2, 2, bytes([10, 10, 200, 200]))
    fake = FakeOcr([Word("Test", 80.0, (0, 0, 10, 10))])
    run_ocr(dummy_img, fake, on_event=events.append)

    step_names = [e.step for e in events]
    assert "otsu" in step_names
    assert "ocr" in step_names

    otsu_ev = next(e for e in events if e.step == "otsu")
    assert "threshold" in otsu_ev.details

    ocr_ev = next(e for e in events if e.step == "ocr")
    assert ocr_ev.details == {"total_words": 1, "accepted_words": 1, "dropped_words": 0}
    for ev in events:
        # Ensure no payload leaks recognized text
        assert "Test" not in str(ev.details)


def test_edge_cases() -> None:
    # 1x1 image
    img1 = GrayImage(1, 1, bytes([100]))
    prep1 = preprocess(img1, steps=("invert_if_dark", "otsu", "deskew"))
    assert prep1.image.width == 1 and prep1.image.height == 1

    # all-white image: otsu threshold 0, deskew 0.0
    img_white = GrayImage(10, 10, bytes([255] * 100))
    prep_white = preprocess(img_white, steps=("otsu", "deskew"))
    assert prep_white.threshold == 0
    assert prep_white.angle == 0.0

    # empty word list: text ''
    fake_empty = FakeOcr([])
    res_empty = run_ocr(img_white, fake_empty)
    assert res_empty.text == ""
    assert res_empty.lines == ()
    assert res_empty.words == ()
    assert res_empty.dropped == 0


def test_additional_decoders_and_errors() -> None:
    # PNM with comments
    p2_comment = b"P2\n# comment\n1 1 # another\n255\n128\n"
    assert load_image(p2_comment).pixels == bytes([128])

    # PNM P3 (RGB ascii)
    p3 = b"P3\n1 1\n255\n255 0 0\n"
    assert load_image(p3).pixels == bytes([76])

    # PNM P5
    p5 = b"P5\n2 1\n255\n\x10\x20"
    assert load_image(p5).pixels == bytes([0x10, 0x20])

    # PNM scaling maxval != 255
    p2_scale = b"P2\n1 1\n100\n50\n"
    # 50 * 255 + 50 // 100 = 128
    assert load_image(p2_scale).pixels == bytes([128])

    # PNG Gray+Alpha and RGBA
    # Gray+alpha: 2x1, gray=100, alpha=255 -> 100; gray=0, alpha=0 -> 255
    png_ga = make_png(2, 1, 4, [bytes([100, 255, 0, 0])], [0])
    res_ga = load_image(png_ga)
    assert res_ga.pixels[0] == 100
    assert res_ga.pixels[1] == 255

    # RGBA: 1x1, (255, 0, 0, 255) -> gray 76
    png_rgba = make_png(1, 1, 6, [bytes([255, 0, 0, 255])], [0])
    assert load_image(png_rgba).pixels == bytes([76])

    # Errors
    with pytest.raises(OcrError) as ex:
        load_image(b"")
    assert ex.value.code == "corrupt"

    with pytest.raises(OcrError) as ex:
        load_image(b"x" * 100, max_bytes=50)
    assert ex.value.code == "too_large"

    with pytest.raises(OcrError) as ex:
        load_image(b"INVALID_HEADER_DATA")
    assert ex.value.code == "unsupported_image"

    # GrayImage validation
    with pytest.raises(OcrError) as ex:
        GrayImage(0, 5, b"")
    assert ex.value.code == "invalid_option"

    with pytest.raises(OcrError) as ex:
        GrayImage(2, 2, b"\x00")
    assert ex.value.code == "corrupt"

    # Word confidence validation
    with pytest.raises(OcrError) as ex:
        Word("test", 150.0, (0, 0, 1, 1))
    assert ex.value.code == "invalid_option"
