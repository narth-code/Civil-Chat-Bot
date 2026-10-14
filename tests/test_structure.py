"""Content-aware structure builder against heading shapes printed in the Environmental Criteria Manual."""

import pytest

from app.ingest.structure import Block, build_structure, find_parent, parse_heading, provision_path


def heading(text, page=1):
    return Block("heading", text, page)


def paragraph(text, page=1):
    return Block("paragraph", text, page)


def identifiers(structure):
    return [p.full_identifier for p in structure.provisions]


def parent_identifier(structure, identifier):
    provision = next(p for p in structure.provisions if p.full_identifier == identifier)
    return None if provision.parent_index is None else structure.provisions[provision.parent_index].full_identifier


def test_parse_heading_reads_printed_identifier_and_title():
    assert parse_heading("SECTION 1 - WATER QUALITY MANAGEMENT") == ((1,), "SECTION 1", "WATER QUALITY MANAGEMENT")
    assert parse_heading("1.1.0 - GENERAL") == ((1, 1, 0), "1.1.0", "GENERAL")
    assert parse_heading("1.2.1.1 - Fiscal Surety Calculations.") == (
        (1, 2, 1, 1),
        "1.2.1.1",
        "Fiscal Surety Calculations",
    )
    key, identifier, title = parse_heading("1.2.2.1 - Submittal Requirements for Projects in the Barton Springs Zone")
    assert (key, identifier) == ((1, 2, 2, 1), "1.2.2.1")
    assert title == "Submittal Requirements for Projects in the Barton Springs Zone"
    assert parse_heading("APPENDIX A - Forms") == (("APPENDIX A",), "APPENDIX A", "Forms")


def test_prose_with_a_leading_number_is_not_a_heading():
    assert parse_heading("10.5 miles of pipe were inspected.") is None
    assert parse_heading("2.5 times the design flow") is None
    assert parse_heading("Fiscal surety is required.") is None


def test_numbered_body_sentence_stays_body_text():
    blocks = [
        heading("1.2.1 - Fiscal Surety"),
        paragraph("1.2.1 The applicant shall post fiscal surety before the permit is issued."),
        paragraph("10.5 miles of pipe were inspected."),
    ]
    structure = build_structure(blocks)
    assert identifiers(structure) == ["1.2.1"]
    assert "10.5 miles of pipe were inspected." in structure.chunks[0].raw_text


def test_printed_heading_labelled_as_paragraph_is_accepted_only_in_explicit_form():
    structure = build_structure([paragraph("1.2.1 - Fiscal Surety"), paragraph("1.3 The applicant shall comply.")])
    assert identifiers(structure) == ["1.2.1"]
    assert structure.chunks[0].raw_text == "1.3 The applicant shall comply."


def test_hierarchy_follows_identifier_prefixes():
    structure = build_structure(
        [
            heading("SECTION 1 - WATER QUALITY MANAGEMENT"),
            heading("1.1.0 - GENERAL"),
            heading("1.2.0 - ADMINISTRATIVE"),
            heading("1.2.1 - Fiscal Surety"),
            heading("1.2.1.1 - Fiscal Surety Calculations."),
            heading("1.2.2.1 - Submittal Requirements for Projects in the Barton Springs Zone"),
        ]
    )
    assert parent_identifier(structure, "SECTION 1") is None
    assert parent_identifier(structure, "1.1.0") == "SECTION 1"
    assert parent_identifier(structure, "1.2.0") == "SECTION 1"
    assert parent_identifier(structure, "1.2.1") == "1.2.0"
    assert parent_identifier(structure, "1.2.1.1") == "1.2.1"
    # 1.2.2 is never printed, so 1.2.2.1 attaches to the nearest printed ancestor
    assert parent_identifier(structure, "1.2.2.1") == "1.2.0"


def test_sibling_order_counts_per_parent():
    structure = build_structure(
        [heading("1.2.0 - ADMINISTRATIVE"), heading("1.2.1 - A"), heading("1.2.2 - B"), heading("1.2.3 - C")]
    )
    assert [p.sibling_order for p in structure.provisions[1:]] == [0, 1, 2]


def test_find_parent_attaches_a_skipped_level_to_nearest_ancestor():
    seen = {(1,): 0, (1, 2, 0): 1}
    assert find_parent((1, 2, 1, 1), seen) == 1
    assert find_parent((1, 3, 4), seen) == 0
    assert find_parent((1,), seen) is None
    assert find_parent((7, 1), seen) is None


def test_duplicate_identifier_is_flagged_not_dropped():
    structure = build_structure(
        [
            heading("1.2.1 - Fiscal Surety"),
            paragraph("First text."),
            heading("1.2.1 - Fiscal Surety"),
            paragraph("Second text."),
        ]
    )
    assert len(structure.provisions) == 2
    first, second = structure.provisions
    assert not first.duplicate and second.duplicate
    assert first.full_identifier != second.full_identifier and second.full_identifier.startswith("1.2.1")
    assert "duplicate_identifier" in structure.chunks[1].flags
    assert "duplicate_identifier" not in structure.chunks[0].flags
    assert [c.raw_text for c in structure.chunks] == ["First text.", "Second text."]


def test_table_becomes_its_own_chunk_with_page_numbers():
    table = "| Zone | Depth |\n|---|---|\n| 1 | 2 |"
    structure = build_structure(
        [
            heading("1.2.1 - Fiscal Surety", page=3),
            paragraph("Before the table.", page=3),
            Block("table", table, 4),
            paragraph("After the table.", page=5),
        ]
    )
    assert [c.table_markdown for c in structure.chunks] == [None, table, None]
    table_chunk = structure.chunks[1]
    assert (table_chunk.raw_text, table_chunk.page_start, table_chunk.page_end) == (table, 4, 4)
    assert "table" in table_chunk.flags
    assert (structure.chunks[0].page_start, structure.chunks[2].page_start) == (3, 5)
    assert [c.ordinal for c in structure.chunks] == [0, 1, 2]


def test_chunks_never_exceed_the_size_limit_and_keep_text_exactly():
    sentences = [f"Sentence number {n} describes the water quality requirement." for n in range(120)]
    paragraphs = [paragraph(" ".join(sentences[i : i + 30]), page=1 + i // 30) for i in range(0, 120, 30)]
    limit = 500
    structure = build_structure([heading("1.1.0 - GENERAL"), *paragraphs], max_chars=limit)
    assert len(structure.chunks) > 1
    assert all(len(c.raw_text) <= limit for c in structure.chunks)
    rebuilt = " ".join(" ".join(c.raw_text.split()) for c in structure.chunks)
    assert rebuilt == " ".join(sentences)
    assert structure.chunks[0].page_start == 1 and structure.chunks[-1].page_end == 4


def test_text_before_the_first_heading_goes_to_an_unnumbered_preamble():
    structure = build_structure([paragraph("Front matter."), heading("SECTION 1 - WATER QUALITY MANAGEMENT")])
    assert identifiers(structure) == ["(unnumbered text)", "SECTION 1"]
    assert structure.provisions[0].parent_index is None


def test_exact_text_is_preserved_inside_a_chunk():
    quote = "The 25-year storm (2.3.0) shall be 10.5 in. per hour; see Table 2."
    structure = build_structure([heading("1.1.0 - GENERAL"), paragraph(f"  {quote}  ")])
    assert structure.chunks[0].raw_text == quote


def test_provision_path_gives_breadcrumb_from_root():
    structure = build_structure(
        [
            heading("SECTION 1 - WATER QUALITY MANAGEMENT"),
            heading("1.2.0 - ADMINISTRATIVE"),
            heading("1.2.1 - Fiscal Surety"),
        ]
    )
    assert (
        provision_path(structure.provisions, 2)
        == "SECTION 1 WATER QUALITY MANAGEMENT > 1.2.0 ADMINISTRATIVE > 1.2.1 Fiscal Surety"
    )


def test_chunk_warnings_are_the_deduplicated_union_of_its_blocks_in_order():
    first = "blurry_image: Laplacian variance 10.0 is 0.20x the document median 50.0 (threshold 0.35x) (page 3)"
    second = 'garbled_text: 2 of 10 lines look garbled, first: "x" (page 4)'
    structure = build_structure(
        [
            heading("1.1.0 - GENERAL", page=3),
            Block("paragraph", "On page three.", 3, warnings=(first,)),
            Block("paragraph", "On page four.", 4, warnings=(first, second)),
            Block("paragraph", "Clean page.", 5),
        ]
    )
    assert len(structure.chunks) == 1
    assert structure.chunks[0].warnings == [first, second]


def test_chunk_without_warnings_has_an_empty_list_and_split_pieces_keep_block_warnings():
    warning = "empty_page: only 3 characters extracted (page 1)"
    sentences = " ".join(f"Sentence number {n} describes the requirement." for n in range(40))
    structure = build_structure(
        [
            heading("1.1.0 - GENERAL"),
            paragraph("Clean text. " * 22),
            Block("paragraph", sentences, 2, warnings=(warning,)),
        ],
        max_chars=300,
    )
    assert structure.chunks[0].warnings == []
    assert len(structure.chunks) > 2
    assert all(warning in chunk.warnings for chunk in structure.chunks[1:])


def test_table_chunk_carries_its_block_warnings():
    warning = "low_ocr_confidence: mean recognition confidence 0.70, min 0.20, 30% of 10 lines below 0.60 (page 4)"
    structure = build_structure(
        [heading("1.2.1 - Fiscal Surety"), Block("table", "| a | b |\n|---|---|\n| 1 | 2 |", 4, warnings=(warning,))]
    )
    assert structure.chunks[0].warnings == [warning]


def test_chunk_method_is_combined_when_its_blocks_come_from_different_sources_and_confidence_is_the_page_mean():
    layer, ocr = "docling:pdf-text-layer", "docling:rapidocr-torch-medium:full_page"
    structure = build_structure(
        [
            heading("1.1.0 - GENERAL"),
            Block("paragraph", "From the layer.", 1, method=layer),
            Block("paragraph", "From OCR, first.", 2, method=ocr, confidence=0.9),
            Block("paragraph", "From OCR, second.", 2, method=ocr, confidence=0.9),
            Block("paragraph", "From OCR, other page.", 3, method=ocr, confidence=0.7),
        ]
    )
    chunk = structure.chunks[0]
    assert chunk.extraction_method == f"{layer} + {ocr}"
    assert chunk.extraction_confidence == pytest.approx(0.8)


def test_chunk_from_the_text_layer_has_no_confidence():
    structure = build_structure(
        [heading("1.1.0 - GENERAL"), Block("paragraph", "Layer text.", 1, method="docling:pdf-text-layer")]
    )
    assert structure.chunks[0].extraction_method == "docling:pdf-text-layer"
    assert structure.chunks[0].extraction_confidence is None
