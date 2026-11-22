"""Lexicographic maximum-cardinality selection of records for a table pair.

The selection is a capacitated bipartite matching:

* left nodes are the before-bake bands, each with ``pieces`` capacity;
* right nodes are the after-bake bands, each with ``pieces`` capacity;
* each active record is an edge from the before band containing its before
  reading to the after band containing its after reading.

The maximum number of retained records is the max-flow value.  Among all
maximum-cardinality solutions we choose the lexicographically largest
keep-vector in registration order: the earliest record at which two solutions
differ is kept.  This is computed by walking records in registration order and
keeping a record whenever a maximum flow is still attainable with every earlier
decision fixed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence


class _Dinic:
    """Integer-capacity Dinic max-flow (graphs here are tiny)."""

    def __init__(self, n: int) -> None:
        self.graph: list[list[list[int]]] = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, cap: int) -> None:
        if cap <= 0:
            return
        self.graph[u].append([v, len(self.graph[v]), cap])
        self.graph[v].append([u, len(self.graph[u]) - 1, 0])

    def max_flow(self, source: int, sink: int) -> int:
        flow = 0
        n = len(self.graph)
        while True:
            level = [-1] * n
            level[source] = 0
            queue = [source]
            for u in queue:
                for v, _rev, cap in self.graph[u]:
                    if cap > 0 and level[v] < 0:
                        level[v] = level[u] + 1
                        queue.append(v)
            if level[sink] < 0:
                return flow
            it = [0] * n

            def send(u: int, pushed: int) -> int:
                if u == sink:
                    return pushed
                while it[u] < len(self.graph[u]):
                    edge = self.graph[u][it[u]]
                    v, rev, cap = edge
                    if cap > 0 and level[v] == level[u] + 1:
                        got = send(v, min(pushed, cap))
                        if got:
                            edge[2] -= got
                            self.graph[v][rev][2] += got
                            return got
                    it[u] += 1
                return 0

            while True:
                pushed = send(source, 10**9)
                if not pushed:
                    break
                flow += pushed


def matching_size(
    before_caps: Sequence[int],
    after_caps: Sequence[int],
    pair_counts: Counter[tuple[int, int]],
) -> int:
    """Maximum number of records that fit the given band capacities."""
    b = len(before_caps)
    a = len(after_caps)
    source = b + a
    sink = source + 1
    graph = _Dinic(sink + 1)
    for i, cap in enumerate(before_caps):
        graph.add_edge(source, i, cap)
    for j, cap in enumerate(after_caps):
        graph.add_edge(b + j, sink, cap)
    for (i, j), count in pair_counts.items():
        graph.add_edge(i, b + j, count)
    return graph.max_flow(source, sink)


@dataclass(frozen=True)
class Band:
    low: int
    high: int
    pieces: int
    band_id: int | None = None
    position: int = 0

    def contains(self, value: int) -> bool:
        return self.low <= value <= self.high


@dataclass(frozen=True)
class Record:
    record_id: int
    coupon_no: str
    before_um: int
    after_um: int
    status: str = "active"


@dataclass(frozen=True)
class Candidate:
    record: Record
    before_index: int
    after_index: int


@dataclass(frozen=True)
class Retained:
    record: Record
    before_index: int
    after_index: int


@dataclass(frozen=True)
class Displaced:
    record: Record
    # before_out_of_range | after_out_of_range | both_out_of_range | capacity
    reason: str
    before_index: int | None = None
    after_index: int | None = None


@dataclass(frozen=True)
class BandUsage:
    band: Band
    used: int

    @property
    def pieces(self) -> int:
        return self.band.pieces

    @property
    def free(self) -> int:
        return self.band.pieces - self.used


@dataclass
class Preview:
    retained: list[Retained] = field(default_factory=list)
    displaced: list[Displaced] = field(default_factory=list)
    already_under_review: list[Record] = field(default_factory=list)
    before_usage: list[BandUsage] = field(default_factory=list)
    after_usage: list[BandUsage] = field(default_factory=list)
    active_count: int = 0
    max_retainable: int = 0

    @property
    def retained_count(self) -> int:
        return len(self.retained)

    @property
    def displaced_count(self) -> int:
        return len(self.displaced)

    @property
    def all_records_count(self) -> int:
        return self.active_count + len(self.already_under_review)


def _find_band(bands: Sequence[Band], value: int) -> int | None:
    # Validated non-overlapping bands contain a value at most once.
    for index, band in enumerate(bands):
        if band.contains(value):
            return index
    return None


def _out_of_range_reason(before_index: int | None, after_index: int | None) -> str:
    if before_index is None and after_index is None:
        return "both_out_of_range"
    if before_index is None:
        return "before_out_of_range"
    return "after_out_of_range"


def select_records(
    before_bands: Sequence[Band],
    after_bands: Sequence[Band],
    records: Iterable[Record],
) -> Preview:
    """Return the lexicographically best maximum-cardinality retained set.

    Only ``active`` records are candidates.  Records already ``needs_review``
    are reported separately and are never selected.
    """
    ordered = sorted(records, key=lambda r: r.record_id)
    active = [r for r in ordered if r.status == "active"]
    under_review = [r for r in ordered if r.status != "active"]

    candidates: list[Candidate] = []
    displaced: list[Displaced] = []
    for record in active:
        bi = _find_band(before_bands, record.before_um)
        ai = _find_band(after_bands, record.after_um)
        if bi is None or ai is None:
            displaced.append(Displaced(record, _out_of_range_reason(bi, ai), bi, ai))
        else:
            candidates.append(Candidate(record, bi, ai))

    pair_counts = Counter((c.before_index, c.after_index) for c in candidates)
    before_caps = [b.pieces for b in before_bands]
    after_caps = [a.pieces for a in after_bands]
    target = matching_size(before_caps, after_caps, pair_counts)

    selected: list[Candidate] = []
    selected_ids: set[int] = set()
    rejected_ids: set[int] = set()
    used_before = [0] * len(before_bands)
    used_after = [0] * len(after_bands)

    for candidate in candidates:
        record_id = candidate.record.record_id
        bi = candidate.before_index
        ai = candidate.after_index

        if (
            used_before[bi] + 1 > before_caps[bi]
            or used_after[ai] + 1 > after_caps[ai]
        ):
            rejected_ids.add(record_id)
            continue

        remaining_before = [
            capacity - used for capacity, used in zip(before_caps, used_before)
        ]
        remaining_after = [
            capacity - used for capacity, used in zip(after_caps, used_after)
        ]
        remaining_before[bi] -= 1
        remaining_after[ai] -= 1

        available: Counter[tuple[int, int]] = Counter()
        for other in candidates:
            other_id = other.record.record_id
            if (
                other_id == record_id
                or other_id in selected_ids
                or other_id in rejected_ids
            ):
                continue
            available[(other.before_index, other.after_index)] += 1

        extra = matching_size(remaining_before, remaining_after, available)
        if len(selected) + 1 + extra >= target:
            selected.append(candidate)
            selected_ids.add(record_id)
            used_before[bi] += 1
            used_after[ai] += 1
        else:
            rejected_ids.add(record_id)

    if len(selected) != target:  # defensive: invariant broken
        raise RuntimeError(
            f"selection invariant failed: selected {len(selected)} of {target}"
        )

    for candidate in candidates:
        if candidate.record.record_id not in selected_ids:
            displaced.append(
                Displaced(
                    candidate.record,
                    "capacity",
                    candidate.before_index,
                    candidate.after_index,
                )
            )
    displaced.sort(key=lambda d: d.record.record_id)

    before_used = Counter(c.before_index for c in selected)
    after_used = Counter(c.after_index for c in selected)
    return Preview(
        retained=[
            Retained(c.record, c.before_index, c.after_index)
            for c in sorted(selected, key=lambda c: c.record.record_id)
        ],
        displaced=displaced,
        already_under_review=under_review,
        before_usage=[
            BandUsage(band, before_used[i]) for i, band in enumerate(before_bands)
        ],
        after_usage=[
            BandUsage(band, after_used[i]) for i, band in enumerate(after_bands)
        ],
        active_count=len(active),
        max_retainable=target,
    )
