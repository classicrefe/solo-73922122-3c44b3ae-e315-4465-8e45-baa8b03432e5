"""Parsing and validation for the two corrective banding tables and records.

Input formats accepted for a table (one band per line, blank lines and ``#``
comments ignored)::

    0-20 15          # closed interval 0..20 µm, 15 pieces
    21-40 12
    41..60 8         # same thing with ".."
    61 80 5          # low high pieces
    81,100,3         # CSV

For records (one coupon per line)::

    C-001 18 22      # coupon before_um after_um
    C-002,19,23

Both parsers are all-or-nothing: one bad line rejects the whole upload so a
partially applied correction can never be mistaken for a complete one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .selection import Band, Record

MAX_RECORDS = 60
MAX_BANDS = 500
INT_LIMIT = 1_000_000

_RANGE_WITH_DASH = re.compile(
    r"^\s*([+-]?\d+)\s*[-~～–—]\s*([+-]?\d+)\s*[,;\t ]\s*(\d+)\s*$"
)
_RANGE_WITH_DOTS = re.compile(
    r"^\s*([+-]?\d+)\s*\.\.\s*([+-]?\d+)\s*[,;\t ]\s*(\d+)\s*$"
)
_TRIPLE = re.compile(
    r"^\s*([+-]?\d+)\s*[,;\t ]\s*([+-]?\d+)\s*[,;\t ]\s*(\d+)\s*$"
)
_RECORD_LINE = re.compile(
    r"^\s*([^\s,;\t]+)\s*[,;\t ]\s*([+-]?\d+)\s*[,;\t ]\s*([+-]?\d+)\s*$"
)


class ValidationError(ValueError):
    """User-facing validation failure; the message is shown in the page."""

    def __init__(self, message: str, details: Sequence[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = list(details or [])

    def all_messages(self) -> list[str]:
        return [self.message, *self.details]


def _iter_data_lines(text: str):
    for lineno, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        yield lineno, stripped


def parse_bands(text: str) -> list[tuple[int, int, int]]:
    """Parse one table into ``(low, high, pieces)`` triples in sheet order."""
    bands: list[tuple[int, int, int]] = []
    errors: list[str] = []
    for lineno, line in _iter_data_lines(text):
        match = (
            _RANGE_WITH_DOTS.match(line)
            or _RANGE_WITH_DASH.match(line)
            or _TRIPLE.match(line)
        )
        if not match:
            errors.append(f"第 {lineno} 行无法识别：{line!r}")
            continue
        low, high, pieces = (int(part) for part in match.groups())
        if low > high:
            errors.append(f"第 {lineno} 行区间下限 {low} 大于上限 {high}")
            continue
        if abs(low) > INT_LIMIT or abs(high) > INT_LIMIT:
            errors.append(f"第 {lineno} 行读数超出允许范围 ±{INT_LIMIT} µm")
            continue
        if pieces > MAX_RECORDS:
            errors.append(f"第 {lineno} 行件数 {pieces} 超过试片总数上限 {MAX_RECORDS}")
            continue
        bands.append((low, high, pieces))

    if errors:
        raise ValidationError("分档表格式有误，未导入任何内容。", errors)
    if not bands:
        raise ValidationError("分档表为空：请至少填写一个档。")
    if len(bands) > MAX_BANDS:
        raise ValidationError(f"档数 {len(bands)} 超过上限 {MAX_BANDS}。")
    return bands


def validate_bands(bands: Sequence[tuple[int, int, int]], label: str) -> None:
    """Check a single table: non-overlapping closed integer intervals."""
    ordered = sorted(bands, key=lambda b: (b[0], b[1]))
    for previous, current in zip(ordered, ordered[1:]):
        if current[0] <= previous[1]:
            raise ValidationError(
                f"{label}存在重叠档 {previous[0]}–{previous[1]} µm 与 "
                f"{current[0]}–{current[1]} µm。同表各档必须互不重叠。"
            )


@dataclass(frozen=True)
class ParsedTables:
    before: list[tuple[int, int, int]]
    after: list[tuple[int, int, int]]


def parse_and_validate_tables(before_text: str, after_text: str) -> ParsedTables:
    before = parse_bands(before_text)
    after = parse_bands(after_text)
    validate_bands(before, "烘烤前表")
    validate_bands(after, "烘烤后表")

    before_total = sum(b[2] for b in before)
    after_total = sum(b[2] for b in after)
    if before_total != after_total:
        raise ValidationError(
            "两张表总件数不一致，无法同时启用。",
            [
                f"烘烤前表共 {before_total} 件，烘烤后表共 {after_total} 件。",
                "两张表必须对同一批试片给出相同的总件数。",
            ],
        )
    if before_total > MAX_RECORDS:
        raise ValidationError(
            f"两张表总件数 {before_total} 超过试片总数上限 {MAX_RECORDS}。"
        )
    return ParsedTables(before=before, after=after)


def to_bands(parsed: Sequence[tuple[int, int, int]]) -> list[Band]:
    return [
        Band(low=low, high=high, pieces=pieces, position=position)
        for position, (low, high, pieces) in enumerate(parsed)
    ]


def parse_records(text: str) -> list[tuple[str, int, int]]:
    """Parse record lines, preserving the order given (registration order)."""
    rows: list[tuple[str, int, int]] = []
    errors: list[str] = []
    seen: set[str] = set()
    for lineno, line in _iter_data_lines(text):
        match = _RECORD_LINE.match(line)
        if not match:
            errors.append(f"第 {lineno} 行无法识别：{line!r}")
            continue
        coupon, before, after = match.groups()
        before_um, after_um = int(before), int(after)
        if abs(before_um) > INT_LIMIT or abs(after_um) > INT_LIMIT:
            errors.append(f"第 {lineno} 行读数超出允许范围 ±{INT_LIMIT} µm")
            continue
        if coupon in seen:
            errors.append(f"第 {lineno} 行试片号 {coupon} 在本批次内重复")
            continue
        seen.add(coupon)
        rows.append((coupon, before_um, after_um))

    if errors:
        raise ValidationError("试片记录格式有误，未导入任何内容。", errors)
    if not rows:
        raise ValidationError("没有可导入的试片记录。")
    return rows


def to_records(rows: Sequence[tuple[int, str, int, int, str]]) -> list[Record]:
    """Convert ``(id, coupon, before, after, status)`` rows to ``Record``."""
    return [
        Record(
            record_id=row[0],
            coupon_no=row[1],
            before_um=row[2],
            after_um=row[3],
            status=row[4],
        )
        for row in rows
    ]
