"""Stage 1: PDF -> ordered blocks, using Docling with local OCR. Resumable in page batches.

Run from the ingestion virtual environment (.venv-ingest). Nothing here talks to the database.
Each batch is written to <out_dir>/blocks-<first>-<last>.json and skipped on the next run.

OCR policy ("full-page" or "auto") decides per page whether the text comes from OCR or from the PDF's own text
layer; contiguous runs of same-method pages inside a batch are converted separately. Every page also gets quality
measurements (see app.ingest.quality) stored in the batch file and summarized in <out_dir>/quality.json.
"""

from __future__ import annotations

import gc
import json
import logging
import re
import statistics
import time
from collections import Counter
from importlib.metadata import version
from itertools import groupby
from pathlib import Path
from typing import Callable

from app.ingest import quality

log = logging.getLogger("ingest.convert")

SKIPPED_LABELS = {"page_header", "page_footer"}
HEADING_LABELS = {"section_header", "title"}
_BATCH_NAME = re.compile(r"blocks-(?P<start>\d+)-(?P<end>\d+)\.json")

SCHEMA_VERSION = 2  # 2: per-page quality measurements, block warnings, per-page extraction method
OCR_POLICIES = ("full-page", "auto")
TEXT_LAYER_MIN_CHARS = 200  # auto policy: pages whose PDF text layer has at least this many characters keep it
PAGE_SCAN_FILE = "page-scan.json"
QUALITY_FILE = "quality.json"

LAYER = "pdf-text-layer"
OCR = "ocr"


def page_count(pdf_path: Path) -> int:
    import pypdfium2

    document = pypdfium2.PdfDocument(str(pdf_path))
    try:
        return len(document)
    finally:
        document.close()


def method_label(kind: str, ocr_backend: str) -> str:
    if kind == LAYER:
        return f"docling-{version('docling')}:{LAYER}"
    return f"docling-{version('docling')}:rapidocr-{ocr_backend}-medium:full_page"


def clear_cache(out_dir: Path) -> int:
    """Delete only this directory's converted batches, page scan and quality report; returns files removed."""
    removed = 0
    for path in [*out_dir.glob("blocks-*.json"), out_dir / PAGE_SCAN_FILE, out_dir / QUALITY_FILE]:
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


# ------------------------------------------------------------------ page scan (pdfium, no Docling)


def _layer_text(page) -> str:
    textpage = page.get_textpage()
    try:
        return " ".join(textpage.get_text_bounded().split())
    finally:
        textpage.close()


def scan_pages(pdf_path: Path, out_dir: Path, first_page: int, last_page: int) -> dict[int, dict]:
    """Per page: Laplacian variance at quality.RENDER_DPI, non-white share and embedded text-layer length.

    Cached in page-scan.json so a resumed run does not re-render.
    """
    import pypdfium2

    cache_path = out_dir / PAGE_SCAN_FILE
    cached: dict[str, dict] = {}
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("dpi") == quality.RENDER_DPI:
            cached = payload["pages"]
    missing = [page for page in range(first_page, last_page + 1) if str(page) not in cached]
    if missing:
        log.info("scanning %s pages (sharpness at %s dpi, text layer)", len(missing), quality.RENDER_DPI)
        document = pypdfium2.PdfDocument(str(pdf_path))
        try:
            for number in missing:
                page = document[number - 1]
                try:
                    gray = page.render(scale=quality.RENDER_DPI / 72, grayscale=True).to_numpy()
                    if gray.ndim == 3:
                        gray = gray[:, :, 0]
                    cached[str(number)] = {
                        "sharpness": quality.laplacian_variance(gray),
                        "ink": quality.ink_fraction(gray),
                        "layer_chars": len(_layer_text(page)),
                    }
                finally:
                    page.close()
        finally:
            document.close()
        _write_json(cache_path, {"dpi": quality.RENDER_DPI, "pages": cached})
    return {page: cached[str(page)] for page in range(first_page, last_page + 1)}


def plan_methods(scan: dict[int, dict], policy: str) -> dict[int, str]:
    """full-page: OCR everywhere. auto: keep the PDF text layer where it has >= TEXT_LAYER_MIN_CHARS characters."""
    if policy == "full-page":
        return {page: OCR for page in scan}
    return {page: LAYER if info["layer_chars"] >= TEXT_LAYER_MIN_CHARS else OCR for page, info in scan.items()}


def _median_sharpness(scan: dict[int, dict]) -> float:
    """Median over pages with visible content, so blank pages do not drag the reference down."""
    values = [info["sharpness"] for info in scan.values() if info["ink"] >= quality.MIN_INK_FRACTION]
    return statistics.median(values) if values else 0.0


# ------------------------------------------------------------------ Docling


def _converter(device: str, ocr_backend: str, do_ocr: bool):
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import OcrMode, PdfPipelineOptions, RapidOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = do_ocr
    options.do_table_structure = True
    options.generate_parsed_pages = do_ocr  # keeps the OCR cells (with recognition confidence) on result.pages
    if do_ocr:
        # The angle classifier mis-rotated some hanging-indent list lines and garbled them; these pages are never rotated.
        options.ocr_options = RapidOcrOptions(
            mode=OcrMode.FULL_PAGE, backend=ocr_backend, use_cls=False, model_size="medium"
        )
    options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice(device))
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})


def _blocks(doc, flags: list[str]) -> list[dict]:
    blocks: list[dict] = []
    for item, _level in doc.iterate_items():
        label = getattr(item.label, "value", str(item.label))
        if label in SKIPPED_LABELS:
            continue
        page = item.prov[0].page_no if getattr(item, "prov", None) else None
        if label == "table":
            try:
                text = item.export_to_markdown(doc=doc)
            except TypeError:
                text = item.export_to_markdown()
            kind = "table"
        elif label in HEADING_LABELS:
            text, kind = getattr(item, "text", ""), "heading"
        elif label == "list_item":
            text, kind = getattr(item, "text", ""), "list_item"
        elif hasattr(item, "text"):
            text, kind = item.text, "paragraph"
        else:
            continue  # pictures and other non-text items
        if text and text.strip():
            blocks.append({"kind": kind, "text": text.strip(), "page": page, "flags": list(flags)})
    return blocks


def _ocr_cells(result) -> dict[int, list]:
    """OCR text cells (with recognition confidence) per page number of a Docling ConversionResult."""
    return {
        page.page_no: [cell for cell in page.cells if getattr(cell, "from_ocr", False)]
        for page in getattr(result, "pages", [])
    }


def _analyse_run(
    result,
    pages: list[int],
    kind: str,
    blocks: list[dict],
    scan: dict[int, dict],
    median: float,
    layer_texts: dict[int, str],
    log_records: list[tuple[int, str]],
    label: str,
) -> tuple[dict[int, dict], list[str]]:
    """Per-page measurements/warnings of one same-method run, plus the run-level warnings that have no page.

    Only warnings attributable to a page go into the page entries (and from there onto chunks); Docling log
    records and page-less errors are returned separately.
    """
    cells = _ocr_cells(result) if kind == OCR else {}
    run_warnings = quality.log_record_warnings(log_records)
    status = getattr(getattr(result, "status", None), "value", None)
    if status not in (None, "success"):
        run_warnings.append(quality.format_warning(quality.DOCLING_WARNING, f"conversion status {status}"))
    page_errors: dict[int, list[str]] = {page: [] for page in pages}
    for error in getattr(result, "errors", []):
        message = quality.truncate(str(getattr(error, "error_message", error)))
        target = getattr(error, "page_no", None)
        if target in page_errors:
            page_errors[target].append(quality.error_item_warning(message, target))
        else:
            run_warnings.append(quality.error_item_warning(message))

    info: dict[int, dict] = {}
    for page in pages:
        page_blocks = [b for b in blocks if b["page"] == page]
        text = "\n".join(b["text"] for b in page_blocks if b["kind"] != "table")
        chars = sum(len(b["text"]) for b in page_blocks)
        warnings: list[str] = [*page_errors[page]]
        entry: dict = {
            "method": label,
            "chars": chars,
            "sharpness": scan[page]["sharpness"],
            "confidence": None,
            "confidence_min": None,
        }
        if kind == OCR:
            confidences = [cell.confidence for cell in cells.get(page, [])]
            stats = quality.confidence_stats(confidences)
            if stats:
                entry["confidence"], entry["confidence_min"] = stats["mean"], stats["min"]
            warnings.append(quality.low_confidence_warning(confidences, page))
            warnings.append(quality.empty_page_warning(chars, page, scan[page]["ink"]))
            warnings.append(
                quality.text_layer_mismatch_warning(
                    layer_texts.get(page, ""), " ".join(cell.text for cell in cells.get(page, [])) or text, page
                )
            )
        if scan[page]["ink"] >= quality.MIN_INK_FRACTION:
            warnings.append(quality.blurry_image_warning(scan[page]["sharpness"], median, page))
        warnings.append(quality.garbled_text_warning(text, page))
        entry["warnings"] = quality.merge_warnings(w for w in warnings if w)
        info[page] = entry
    return info, run_warnings


def _convert_batch(
    start: int,
    end: int,
    methods: dict[int, str],
    converters: dict[str, object],
    make_converter: Callable[[bool], object],
    pdf_path: Path,
    scan: dict[int, dict],
    median: float,
    layer_texts: dict[int, str],
    ocr_backend: str,
) -> tuple[list[dict], dict[int, dict], list[dict]]:
    blocks: list[dict] = []
    page_info: dict[int, dict] = {}
    run_warnings: list[dict] = []
    for kind, group in groupby(range(start, end + 1), key=methods.__getitem__):
        pages = list(group)
        if kind not in converters:
            log.info("loading Docling models (%s)", kind)
            converters[kind] = make_converter(kind == OCR)
        with quality.LogCapture() as capture:
            result = converters[kind].convert(pdf_path, page_range=(pages[0], pages[-1]))
        run_blocks = _blocks(result.document, ["ocr"] if kind == OCR else [])
        info, run_level = _analyse_run(
            result,
            pages,
            kind,
            run_blocks,
            scan,
            median,
            layer_texts,
            capture.records,
            method_label(kind, ocr_backend),
        )
        for block in run_blocks:
            warnings = info.get(block["page"], {}).get("warnings")
            if warnings:
                block["warnings"] = warnings
        blocks.extend(run_blocks)
        page_info.update(info)
        if run_level:
            for warning in run_level:
                log.warning("pages %s-%s: %s", pages[0], pages[-1], warning)
            run_warnings.append({"pages": [pages[0], pages[-1]], "warnings": run_level})
    return blocks, page_info, run_warnings


def convert_pdf(
    pdf_path: Path,
    out_dir: Path,
    *,
    first_page: int = 1,
    last_page: int | None = None,
    batch_pages: int = 20,
    device: str = "cuda",
    ocr_backend: str = "torch",
    ocr_policy: str = "auto",
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Convert pages in resumable batches; returns the quality summary (also written to quality.json)."""
    if ocr_policy not in OCR_POLICIES:
        raise ValueError(f"ocr_policy must be one of {OCR_POLICIES}")
    import pypdfium2

    out_dir.mkdir(parents=True, exist_ok=True)
    total = page_count(pdf_path)
    last = min(last_page or total, total)
    targets = [
        (
            start,
            min(start + batch_pages - 1, last),
            out_dir / f"blocks-{start:04d}-{min(start + batch_pages - 1, last):04d}.json",
        )
        for start in range(first_page, last + 1, batch_pages)
    ]
    for _start, _end, target in targets:
        if target.exists():
            meta = json.loads(target.read_text(encoding="utf-8"))["meta"]
            if meta.get("schema_version") != SCHEMA_VERSION or meta.get("ocr_policy") != ocr_policy:
                raise ValueError(
                    f"{target.name} was converted with schema {meta.get('schema_version')} / policy "
                    f"{meta.get('ocr_policy')!r}, not schema {SCHEMA_VERSION} / {ocr_policy!r}; rerun with --reconvert"
                )

    converters: dict[str, object] = {}
    try:
        if any(not target.exists() for _s, _e, target in targets):
            scan = scan_pages(pdf_path, out_dir, first_page, last)
            methods = plan_methods(scan, ocr_policy)
            median = _median_sharpness(scan)
            log.info(
                "ocr policy %s: %s pages from the text layer, %s OCR; median sharpness %.1f",
                ocr_policy,
                sum(1 for m in methods.values() if m == LAYER),
                sum(1 for m in methods.values() if m == OCR),
                median,
            )
            document = pypdfium2.PdfDocument(str(pdf_path))
        else:
            document = None
        try:
            for start, end, target in targets:
                if target.exists():
                    log.info("pages %s-%s already converted; skipping", start, end)
                else:
                    began = time.perf_counter()
                    layer_texts = {}
                    for number in range(start, end + 1):
                        if scan[number]["layer_chars"]:
                            page = document[number - 1]
                            try:
                                layer_texts[number] = _layer_text(page)
                            finally:
                                page.close()
                    blocks, page_info, run_warnings = _convert_batch(
                        start,
                        end,
                        methods,
                        converters,
                        lambda do_ocr: _converter(device, ocr_backend, do_ocr),
                        pdf_path,
                        scan,
                        median,
                        layer_texts,
                        ocr_backend,
                    )
                    meta = {
                        "schema_version": SCHEMA_VERSION,
                        "ocr_policy": ocr_policy,
                        "ocr_backend": ocr_backend,
                        "pages": [start, end],
                        "median_sharpness": median,
                        "page_info": {str(page): info for page, info in page_info.items()},
                        "run_warnings": run_warnings,
                    }
                    _write_json(target, {"meta": meta, "blocks": blocks})
                    log.info("pages %s-%s -> %s blocks in %.1fs", start, end, len(blocks), time.perf_counter() - began)
                if progress:
                    progress(end, last)
        finally:
            if document is not None:
                document.close()
    finally:
        converters.clear()
        _release_gpu()
    return write_quality_report(out_dir, first_page, last)


def _release_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _batches(out_dir: Path, first_page: int, last_page: int | None):
    """Batch files of this directory intersecting [first_page, last_page], sorted, as (start, end, path)."""
    batches: list[tuple[int, int, Path]] = []
    for path in out_dir.glob("blocks-*.json"):
        match = _BATCH_NAME.fullmatch(path.name)
        if match is None:
            continue
        start, end = int(match["start"]), int(match["end"])
        if end >= first_page and (last_page is None or start <= last_page):
            batches.append((start, end, path))
    batches.sort()
    return batches


def write_quality_report(out_dir: Path, first_page: int, last_page: int) -> dict:
    """Aggregate per-page warnings of the converted range into quality.json and log the summary.

    Warnings from Docling log records or errors without a page are run-level: they are listed under run_warnings
    here and in the batch meta, but they are never attached to a page or chunk.
    """
    page_warnings: dict[int, list[str]] = {}
    run_warnings: list[dict] = []
    methods: dict[str, int] = {}
    median = None
    policy = None
    for _start, _end, path in _batches(out_dir, first_page, last_page):
        meta = json.loads(path.read_text(encoding="utf-8"))["meta"]
        median, policy = meta.get("median_sharpness"), meta.get("ocr_policy")
        run_warnings.extend(meta["run_warnings"])
        for key, info in meta["page_info"].items():
            page = int(key)
            if first_page <= page <= last_page:
                page_warnings[page] = info["warnings"]
                methods[info["method"]] = methods.get(info["method"], 0) + 1
    summary = {
        **quality.summarize(page_warnings, last_page - first_page + 1),
        "run_warning_counts": dict(Counter(quality.warning_code(w) for run in run_warnings for w in run["warnings"])),
        "run_warnings": run_warnings,
        "ocr_policy": policy,
        "pages_by_method": methods,
        "median_sharpness": median,
        "blur_ratio": quality.BLUR_RATIO,
    }
    _write_json(out_dir / QUALITY_FILE, summary)
    log.info("quality summary: policy=%s methods=%s", policy, methods)
    log.info(
        "quality: %s of %s pages have warnings; counts %s",
        summary["pages_with_warnings"],
        summary["pages_total"],
        {code: n for code, n in summary["counts"].items() if n},
    )
    for entry in summary["worst_pages"]:
        log.info("quality worst page %s: %s", entry["page"], " | ".join(entry["warnings"]))
    return summary


def read_blocks(out_dir: Path, first_page: int = 1, last_page: int | None = None):
    """Yield stored blocks in page order for exactly [first_page, last_page].

    Only batch files of this directory that intersect the range are read; overlapping files (from a
    different batch size) or a gap in coverage raise instead of silently duplicating or losing pages.
    Batch files written before warnings existed raise too: re-convert them (--reconvert).
    """
    from app.ingest.structure import Block

    batches = _batches(out_dir, first_page, last_page)
    expected: int | None = None
    for start, end, path in batches:
        if expected is None:
            if start > first_page:
                raise ValueError(f"pages {first_page}-{start - 1} are not converted in {out_dir}")
        elif start < expected:
            raise ValueError(f"{path.name} overlaps an earlier batch file in {out_dir}; remove the stale files")
        elif start > expected:
            raise ValueError(f"pages {expected}-{start - 1} are not converted in {out_dir}")
        expected = end + 1
    if last_page is not None and (expected or first_page) <= last_page:
        raise ValueError(f"pages {expected or first_page}-{last_page} are not converted in {out_dir}")
    for _start, _end, path in batches:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["meta"].get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{path.name} predates extraction warnings (schema {payload['meta'].get('schema_version')}, "
                f"need {SCHEMA_VERSION}); rerun the convert stage with --reconvert"
            )
        page_info = payload["meta"]["page_info"]
        for item in payload["blocks"]:
            page = item.get("page")
            if page is not None and (page < first_page or (last_page is not None and page > last_page)):
                continue
            info = page_info.get(str(page), {})
            yield Block(
                item["kind"],
                item["text"],
                page,
                tuple(item.get("flags", ())),
                tuple(item.get("warnings", ())),
                info.get("method"),
                info.get("confidence"),
            )
