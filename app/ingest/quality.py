"""Per-page extraction quality detectors.

Each detector is a small pure function (measurement in, warning string or None out) so it can be unit tested
without Docling, a GPU or a database. A warning is one string, "<code>: <detail>", with " (page N)" appended when
the page is known. quality_flags on a chunk stay coarse tags; these strings explain what looked wrong and where.
numpy is imported lazily so the module loads in environments without it (only the image detectors need it).
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Iterable, Sequence

BLURRY_IMAGE = "blurry_image"
LOW_OCR_CONFIDENCE = "low_ocr_confidence"
GARBLED_TEXT = "garbled_text"
EMPTY_PAGE = "empty_page"
OCR_TEXT_LAYER_MISMATCH = "ocr_text_layer_mismatch"
DOCLING_ERROR = "docling_error"
DOCLING_WARNING = "docling_warning"
NO_TEXT_EXTRACTED = "no_text_extracted"

WARNING_CODES = (
    BLURRY_IMAGE,
    LOW_OCR_CONFIDENCE,
    GARBLED_TEXT,
    EMPTY_PAGE,
    OCR_TEXT_LAYER_MISMATCH,
    DOCLING_ERROR,
    DOCLING_WARNING,
    NO_TEXT_EXTRACTED,
)

RENDER_DPI = 150
BLUR_RATIO = 0.35  # flagged when sharpness < BLUR_RATIO x the document's median page sharpness
LOW_MEAN_CONFIDENCE = 0.85
LOW_CELL_CONFIDENCE = 0.6
LOW_CELL_SHARE = 0.10
MIN_PAGE_CHARS = 20
MIN_INK_FRACTION = 0.002  # share of non-white pixels above which an empty extraction is suspicious
MIN_LAYER_TOKENS = 20
MISMATCH_SIMILARITY = 0.8
GARBLED_MIN_TOKENS = 5
GARBLED_LINE_SHARE = 0.6
GARBLED_PAGE_SHARE = 0.05
GARBLED_MIN_LINES = 2
MESSAGE_LIMIT = 200
GARBLED_FORMULA_SHARE = 0.4

_TOKEN = re.compile(r"[A-Za-z]+")
_WORD = re.compile(r"[a-z0-9]+")
_VOWELS = frozenset("aeiouyAEIOUY")


def format_warning(code: str, detail: str, page: int | None = None) -> str:
    suffix = f" (page {page})" if page is not None else ""
    return f"{code}: {detail}{suffix}"


def warning_code(warning: str) -> str:
    return warning.partition(":")[0]


def warning_page(warning: str) -> int | None:
    match = re.search(r" \(page (\d+)\)$", warning)
    return int(match.group(1)) if match else None


def truncate(text: str, limit: int = MESSAGE_LIMIT) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def merge_warnings(*groups: Iterable[str]) -> list[str]:
    """Union of warning strings, de-duplicated, first-seen order preserved."""
    return list(dict.fromkeys(warning for group in groups for warning in group))


# ---------------------------------------------------------------- image measurements


def laplacian_variance(gray, tile: int = 32, min_contrast: int = 100) -> float:
    """Variance of the 4-neighbour Laplacian over the 32x32 tiles that contain a strong edge (max-min >= 100).

    Low means blurry, high means sharp. Tiles of white margin or flat fill are ignored so sparse and
    graphic pages are compared on their edges only; an image with no strong edge uses every pixel.
    """
    import numpy

    image = numpy.asarray(gray, dtype=numpy.float32)
    if image.ndim != 2 or image.shape[0] < 3 or image.shape[1] < 3:
        raise ValueError("laplacian_variance needs a 2-D image of at least 3x3 pixels")
    interior = image[1:-1, 1:-1]
    laplacian = 4.0 * interior - image[:-2, 1:-1] - image[2:, 1:-1] - image[1:-1, :-2] - image[1:-1, 2:]
    rows, cols = laplacian.shape[0] // tile, laplacian.shape[1] // tile
    if rows == 0 or cols == 0:
        return float(laplacian.var())
    laplacian = laplacian[: rows * tile, : cols * tile].reshape(rows, tile, cols, tile)
    contrast = interior[: rows * tile, : cols * tile].reshape(rows, tile, cols, tile)
    edged = (contrast.max(axis=(1, 3)) - contrast.min(axis=(1, 3))) >= min_contrast
    if not edged.any():
        return float(laplacian.var())
    return float(laplacian[numpy.broadcast_to(edged[:, None, :, None], laplacian.shape)].var())


def ink_fraction(gray, white_level: int = 235) -> float:
    """Share of pixels darker than near-white; ~0 for a blank page."""
    import numpy

    image = numpy.asarray(gray)
    return float((image < white_level).mean())


def blurry_image_warning(
    sharpness: float, document_median: float, page: int | None = None, ratio: float = BLUR_RATIO
) -> str | None:
    if document_median <= 0 or sharpness >= ratio * document_median:
        return None
    return format_warning(
        BLURRY_IMAGE,
        f"Laplacian variance {sharpness:.1f} is {sharpness / document_median:.2f}x the document median "
        f"{document_median:.1f} (threshold {ratio:.2f}x)",
        page,
    )


# ---------------------------------------------------------------- OCR confidence


def confidence_stats(confidences: Sequence[float]) -> dict[str, float | int] | None:
    if not confidences:
        return None
    low = sum(1 for value in confidences if value < LOW_CELL_CONFIDENCE)
    return {
        "mean": sum(confidences) / len(confidences),
        "min": min(confidences),
        "cells": len(confidences),
        "low_share": low / len(confidences),
    }


def low_confidence_warning(confidences: Sequence[float], page: int | None = None) -> str | None:
    stats = confidence_stats(confidences)
    if stats is None:
        return None
    if stats["mean"] >= LOW_MEAN_CONFIDENCE and stats["low_share"] <= LOW_CELL_SHARE:
        return None
    return format_warning(
        LOW_OCR_CONFIDENCE,
        f"mean recognition confidence {stats['mean']:.2f}, min {stats['min']:.2f}, "
        f"{stats['low_share'] * 100:.0f}% of {stats['cells']} lines below {LOW_CELL_CONFIDENCE:.2f}",
        page,
    )


# ---------------------------------------------------------------- garbled text


_COMMON_SHORT_WORDS = frozenset(
    "a i am an as at be by do go he if in is it me my no of on or so to up us we oh ok ye".split()
)
_SYMBOLS = frozenset("=*/+-^()<>%\u00d7\u00b1")


def _is_bad_token(token: str) -> bool:
    if len(token) <= 2:
        return token.lower() not in _COMMON_SHORT_WORDS
    return not any(char in _VOWELS for char in token)


def garbled_line(line: str) -> bool:
    """True when most lower/mixed-case alphabetic tokens of a line are stray 1-2 letter pieces or vowel-free.

    Skipped: table markdown, formula-like lines (over 40% digits/operators) and ALL-CAPS tokens (acronyms,
    headings), which would otherwise look like fragments. Example hit: "'ssns sd u od h s th t".
    """
    if line.lstrip().startswith("|"):
        return False
    compact = [char for char in line if not char.isspace()]
    if (
        not compact
        or sum(char.isdigit() or char in _SYMBOLS for char in compact) / len(compact) > GARBLED_FORMULA_SHARE
    ):
        return False
    tokens = [token for token in _TOKEN.findall(line) if not token.isupper()]
    if len(tokens) < GARBLED_MIN_TOKENS:
        return False
    return sum(_is_bad_token(token) for token in tokens) / len(tokens) > GARBLED_LINE_SHARE


def garbled_text_warning(text: str, page: int | None = None) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bad = [line for line in lines if garbled_line(line)]
    if len(bad) < GARBLED_MIN_LINES or len(bad) / len(lines) < GARBLED_PAGE_SHARE:
        return None
    return format_warning(
        GARBLED_TEXT,
        f'{len(bad)} of {len(lines)} lines look garbled, first: "{truncate(bad[0], 60)}"',
        page,
    )


# ---------------------------------------------------------------- empty extraction


def empty_page_warning(char_count: int, page: int | None = None, ink: float | None = None) -> str | None:
    """A page that produced fewer than MIN_PAGE_CHARS characters.

    Zero text on a page that visibly has content is no_text_extracted; otherwise it is an empty_page.
    """
    if char_count >= MIN_PAGE_CHARS:
        return None
    if char_count == 0 and ink is not None and ink >= MIN_INK_FRACTION:
        return format_warning(
            NO_TEXT_EXTRACTED, f"no text extracted although {ink * 100:.1f}% of the page is non-white", page
        )
    return format_warning(EMPTY_PAGE, f"only {char_count} characters extracted", page)


# ---------------------------------------------------------------- OCR vs embedded text layer


def token_similarity(first: str, second: str) -> float:
    """Jaccard similarity of the normalized (lower-case alphanumeric) token sets."""
    a, b = set(_WORD.findall(first.lower())), set(_WORD.findall(second.lower()))
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def text_layer_mismatch_warning(layer_text: str, ocr_text: str, page: int | None = None) -> str | None:
    """Compare the embedded text layer with OCR output of the same page; skipped when the layer is (nearly) empty."""
    if len(_WORD.findall(layer_text.lower())) < MIN_LAYER_TOKENS:
        return None
    similarity = token_similarity(layer_text, ocr_text)
    if similarity >= MISMATCH_SIMILARITY:
        return None
    return format_warning(
        OCR_TEXT_LAYER_MISMATCH,
        f"OCR and embedded text layer share only {similarity:.2f} of their tokens (threshold {MISMATCH_SIMILARITY:.2f})",
        page,
    )


# ---------------------------------------------------------------- Docling errors and log records


class LogCapture(logging.Handler):
    """Collects WARNING/ERROR records of the Docling and OCR loggers while attached."""

    LOGGERS = ("docling", "docling_core", "RapidOCR", "rapidocr")

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[tuple[int, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append((record.levelno, record.getMessage()))

    def __enter__(self) -> "LogCapture":
        for name in self.LOGGERS:
            logging.getLogger(name).addHandler(self)
        return self

    def __exit__(self, *exc) -> None:
        for name in self.LOGGERS:
            logging.getLogger(name).removeHandler(self)


def log_record_warnings(records: Iterable[tuple[int, str]]) -> list[str]:
    """One docling_warning/docling_error string per distinct message (repeat count in the detail), no page."""
    counts: Counter[tuple[str, str]] = Counter()
    for level, message in records:
        code = DOCLING_ERROR if level >= logging.ERROR else DOCLING_WARNING
        counts[(code, truncate(message))] += 1
    return [
        format_warning(code, message if count == 1 else f"{message} [x{count}]")
        for (code, message), count in counts.items()
    ]


def error_item_warning(message: str, page: int | None = None) -> str:
    return format_warning(DOCLING_ERROR, truncate(message), page)


# ---------------------------------------------------------------- aggregation


def summarize(page_warnings: dict[int, list[str]], pages_total: int, worst: int = 20) -> dict:
    """Counts per warning code and the pages with the most warnings."""
    counts = Counter(warning_code(w) for warnings in page_warnings.values() for w in warnings)
    ranked = sorted(((page, ws) for page, ws in page_warnings.items() if ws), key=lambda item: (-len(item[1]), item[0]))
    return {
        "pages_total": pages_total,
        "pages_with_warnings": len(ranked),
        "counts": {code: counts.get(code, 0) for code in WARNING_CODES},
        "worst_pages": [{"page": page, "warnings": ws} for page, ws in ranked[:worst]],
    }
