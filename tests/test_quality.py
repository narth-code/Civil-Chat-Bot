"""Extraction quality detectors on synthetic inputs (no GPU, Docling or database).

The image detectors need numpy: they are skipped in the API environment (.venv) and run in .venv-ingest.
"""

import logging

import pytest

from app.ingest import quality
from app.ingest.structure import Block, build_structure


def _image(sharp: bool):
    numpy = pytest.importorskip("numpy")
    rng = numpy.random.default_rng(7)
    image = numpy.full((256, 256), 255.0, dtype=numpy.float32)
    # Black vertical bars 3 px wide, like text strokes.
    for column in range(8, 248, 12):
        image[16:240, column : column + 3] = 0.0
    if not sharp:
        # Box blur, 9 px wide in both directions.
        kernel = numpy.ones(9, dtype=numpy.float32) / 9
        for axis in (0, 1):
            image = numpy.apply_along_axis(lambda line: numpy.convolve(line, kernel, mode="same"), axis, image)
    return image + rng.normal(0, 0.0, image.shape).astype(numpy.float32)


def test_blurred_image_scores_far_below_the_sharp_one():
    sharp, blurred = quality.laplacian_variance(_image(True)), quality.laplacian_variance(_image(False))
    assert sharp > 0 and blurred < 0.35 * sharp


def test_blur_warning_is_relative_to_the_document_median():
    warning = quality.blurry_image_warning(10.0, 100.0, page=7)
    assert warning is not None and warning.startswith("blurry_image: ") and warning.endswith(" (page 7)")
    assert "10.0" in warning and "100.0" in warning
    assert quality.blurry_image_warning(40.0, 100.0, page=7) is None
    assert quality.blurry_image_warning(0.0, 0.0, page=7) is None  # no reference, no judgement


def test_blank_page_has_no_ink_and_a_page_with_content_does():
    numpy = pytest.importorskip("numpy")
    assert quality.ink_fraction(numpy.full((50, 50), 255, dtype=numpy.uint8)) == 0.0
    assert quality.ink_fraction(_image(True)) > quality.MIN_INK_FRACTION


def test_flat_fill_without_edges_does_not_depend_on_page_whiteness():
    numpy = pytest.importorskip("numpy")
    sparse = numpy.full((256, 256), 255.0, dtype=numpy.float32)
    sparse[100:110, 100:110] = 0.0
    assert quality.laplacian_variance(sparse) > 0


def test_low_ocr_confidence_by_mean_and_by_share_of_weak_cells():
    low_mean = quality.low_confidence_warning([0.7, 0.75, 0.8, 0.72], page=2)
    assert low_mean is not None and low_mean.startswith("low_ocr_confidence: mean recognition confidence 0.74")
    assert low_mean.endswith("(page 2)")
    many_weak = quality.low_confidence_warning([0.99] * 8 + [0.3, 0.2], page=2)  # mean 0.91 but 20% below 0.6
    assert many_weak is not None and "20% of 10 lines" in many_weak
    assert quality.low_confidence_warning([0.95, 0.9, 0.97], page=2) is None
    assert quality.low_confidence_warning([], page=2) is None


def test_garbled_line_detected_and_named_in_the_warning():
    garbled = "'ssns sd u od h s th t"
    text = "\n".join(
        [
            "The design storm shall be selected by the engineer of record.",
            garbled,
            "Runoff coefficients are given in the table below for each land use.",
            "ssns sd u od h s th t",
            "Inlets shall be spaced so that the ponded width does not exceed limits.",
        ]
    )
    assert quality.garbled_line(garbled)
    warning = quality.garbled_text_warning(text, page=9)
    assert warning is not None and warning.startswith("garbled_text: 2 of 5 lines")
    assert "'ssns sd u od h s th t" in warning and warning.endswith("(page 9)")


def test_ordinary_short_word_lines_formulas_and_acronyms_are_not_garbled():
    assert not quality.garbled_line("If vegetation is to be temporary, it is to be in a pot")
    assert not quality.garbled_line("WQV = 1.75 ac * 43,560 ft2/ac * 1.16 in * 1 ft/12 in = 7,350 ft3")
    assert not quality.garbled_line("E.C.S.D. INSPECTOR INFORMATION:")
    assert not quality.garbled_line("| a | b | c | d | e | f |")
    clean = "The design storm shall be selected.\nRunoff is computed by the rational method.\n"
    assert quality.garbled_text_warning(clean, page=1) is None
    one_bad_line = clean + "ssns sd u od h s th t"
    assert quality.garbled_text_warning(one_bad_line, page=1) is None  # a single stray line is not a page problem


def test_empty_page_and_no_text_extracted():
    empty = quality.empty_page_warning(3, page=4, ink=0.0)
    assert empty == "empty_page: only 3 characters extracted (page 4)"
    nothing = quality.empty_page_warning(0, page=5, ink=0.2)
    assert nothing is not None and nothing.startswith("no_text_extracted: ") and nothing.endswith("(page 5)")
    assert quality.empty_page_warning(0, page=6, ink=0.0).startswith("empty_page: ")
    assert quality.empty_page_warning(20, page=6, ink=0.5) is None


def test_text_layer_mismatch_reports_the_similarity():
    layer = "the design storm shall be selected by the engineer of record using the rainfall tables " * 2
    assert quality.text_layer_mismatch_warning(layer, layer.upper(), page=3) is None
    garbage = "zz qq xx ww vv uu tt ss rr pp oo nn mm ll kk jj ii hh gg ff ee dd cc bb aa"
    warning = quality.text_layer_mismatch_warning(layer, garbage, page=3)
    assert warning is not None and warning.startswith("ocr_text_layer_mismatch: ")
    assert "0.00" in warning and warning.endswith("(page 3)")
    assert quality.text_layer_mismatch_warning("short layer", garbage, page=3) is None  # layer too short to compare
    assert quality.token_similarity("a b c", "a b d") == pytest.approx(0.5)


def test_docling_log_records_become_deduplicated_warnings_with_truncated_text():
    records = [(logging.WARNING, "RapidOCR returned empty result!")] * 3 + [(logging.ERROR, "x" * 500)]
    warnings = quality.log_record_warnings(records)
    assert warnings[0] == "docling_warning: RapidOCR returned empty result! [x3]"
    assert (
        warnings[1].startswith("docling_error: ") and len(warnings[1]) <= len("docling_error: ") + quality.MESSAGE_LIMIT
    )
    assert quality.error_item_warning("boom", 12) == "docling_error: boom (page 12)"


def test_log_capture_collects_only_docling_and_ocr_loggers_while_attached():
    with quality.LogCapture() as capture:
        logging.getLogger("docling.models.thing").warning("layout model is slow")
        logging.getLogger("RapidOCR").error("recognition failed")
        logging.getLogger("unrelated").warning("ignore me")
    logging.getLogger("docling.models.thing").warning("after detach")
    assert capture.records == [(logging.WARNING, "layout model is slow"), (logging.ERROR, "recognition failed")]


def test_warning_helpers_and_summary_ranking():
    warning = quality.format_warning(quality.EMPTY_PAGE, "only 0 characters extracted", 8)
    assert (quality.warning_code(warning), quality.warning_page(warning)) == ("empty_page", 8)
    assert quality.warning_page("docling_warning: no page") is None
    page_warnings = {1: [], 2: [warning], 3: [warning, "garbled_text: x (page 3)"]}
    summary = quality.summarize(page_warnings, pages_total=3, worst=1)
    assert summary["counts"]["empty_page"] == 2 and summary["counts"]["garbled_text"] == 1
    assert summary["counts"]["blurry_image"] == 0
    assert summary["pages_with_warnings"] == 2
    assert [entry["page"] for entry in summary["worst_pages"]] == [3]


def test_page_warnings_aggregate_onto_the_chunks_built_from_those_pages():
    blurry = quality.format_warning(quality.BLURRY_IMAGE, "Laplacian variance 1.0 ...", 2)
    garbled = quality.format_warning(quality.GARBLED_TEXT, "1 of 3 lines look garbled", 3)
    blocks = [
        Block("heading", "1.1.0 - GENERAL", 1),
        Block("paragraph", "Page one.", 1),
        Block("paragraph", "Page two.", 2, warnings=(blurry,)),
        Block("paragraph", "Page three.", 3, warnings=(blurry, garbled)),
    ]
    chunk = build_structure(blocks).chunks[0]
    assert chunk.warnings == [blurry, garbled]
    assert quality.merge_warnings([blurry], [blurry, garbled]) == [blurry, garbled]


def _result(errors=(), status="success", cells=None):
    from types import SimpleNamespace

    pages = [
        SimpleNamespace(
            page_no=number,
            cells=[SimpleNamespace(from_ocr=True, confidence=value, text="text") for value in values],
        )
        for number, values in (cells or {}).items()
    ]
    return SimpleNamespace(status=SimpleNamespace(value=status), errors=list(errors), pages=pages)


def _analyse(result, blocks, log_records, pages=(1, 2, 3)):
    from app.ingest import convert

    scan = {page: {"sharpness": 100.0, "ink": 0.1, "layer_chars": 0} for page in pages}
    return convert._analyse_run(result, list(pages), convert.OCR, blocks, scan, 100.0, {}, log_records, "ocr-label")


def _block(text, page):
    return {"kind": "paragraph", "text": text, "page": page, "flags": ["ocr"]}


def test_log_derived_warnings_stay_run_level_and_never_reach_pages_or_chunks():
    blocks = [_block("Sentence about inlets and the design storm. " * 2, 1), _block("More ordinary text here. " * 3, 3)]
    records = [
        (logging.WARNING, "RapidOCR returned empty result!"),
        (logging.WARNING, "The text detection result is empty"),
    ]
    info, run_level = _analyse(_result(cells={1: [0.99], 2: [], 3: [0.99]}), blocks, records)
    assert [quality.warning_code(w) for w in run_level] == ["docling_warning", "docling_warning"]
    for entry in info.values():
        assert not any(quality.warning_code(w).startswith("docling") for w in entry["warnings"])
    assert info[1]["warnings"] == [] and info[3]["warnings"] == []


def test_page_attributed_docling_errors_go_to_that_page_and_pageless_ones_stay_run_level():
    from types import SimpleNamespace

    errors = [
        SimpleNamespace(error_message="table model failed", page_no=2),
        SimpleNamespace(error_message="boom", page_no=None),
    ]
    blocks = [_block("Ordinary prose about drainage criteria for the site. " * 2, page) for page in (1, 2, 3)]
    info, run_level = _analyse(_result(errors=errors, status="partial_success"), blocks, [])
    assert info[2]["warnings"] == ["docling_error: table model failed (page 2)"]
    assert info[1]["warnings"] == [] and info[3]["warnings"] == []
    assert run_level == ["docling_warning: conversion status partial_success", "docling_error: boom"]


def test_empty_page_warning_reaches_only_chunks_that_include_that_page():
    blocks = [_block("Ordinary prose about drainage criteria for the site. " * 2, page) for page in (1, 3)]
    info, _run_level = _analyse(_result(cells={1: [0.99], 2: [], 3: [0.99]}), blocks, [])
    assert [quality.warning_code(w) for w in info[2]["warnings"]] == ["no_text_extracted"]
    assert info[1]["warnings"] == [] and info[3]["warnings"] == []

    page_one, page_three = info[1]["warnings"], info[3]["warnings"]
    empty = info[2]["warnings"]
    stream = [
        Block("heading", "1.1.0 - GENERAL", 1),
        Block("paragraph", "Page one text.", 1, warnings=tuple(page_one)),
        Block("heading", "1.2.0 - NEXT", 3),
        Block("paragraph", "Page three text.", 3, warnings=tuple(page_three) + tuple(empty)),
    ]
    first, second = build_structure(stream).chunks
    assert first.warnings == [] and second.warnings == empty


def test_chunk_spanning_pages_carries_only_the_warnings_of_the_pages_it_spans():
    on_two = quality.format_warning(quality.GARBLED_TEXT, "1 of 3 lines look garbled", 2)
    on_four = quality.format_warning(quality.EMPTY_PAGE, "only 0 characters extracted", 4)
    stream = [
        Block("heading", "1.1.0 - GENERAL", 1),
        Block("paragraph", "Page one.", 1),
        Block("paragraph", "Page two.", 2, warnings=(on_two,)),
        Block("heading", "1.2.0 - NEXT", 3),
        Block("paragraph", "Page four.", 4, warnings=(on_four,)),
    ]
    first, second = build_structure(stream).chunks
    assert (first.page_start, first.page_end, first.warnings) == (1, 2, [on_two])
    assert (second.page_start, second.warnings) == (4, [on_four])
