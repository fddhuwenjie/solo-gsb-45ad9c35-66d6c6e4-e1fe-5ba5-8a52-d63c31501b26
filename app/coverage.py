"""文档批注覆盖图（纯函数，不依赖数据库）。

指定文档版本上全部批注的迁移位置，把全文码点区间 ``[0, len(text))``
划分为连续、无重叠的**最大片段**：

- ``resolved`` / 人工确认 ``confirmed`` 的生效范围计入 ``definite``；
- 未确认且非顺延的 ``ambiguous`` 记录：其候选落点（``candidates``）先
  **按批注**把严格重叠的候选合并为连续最大区间，再计入 ``possible``。
  合并后同一批注的 possible 区间两两不相交，扫描线在任一交界点上对该
  ID 只需加入 / 移除一次，重叠候选不会让同一 ID 重复出现或被提前移除；
- 片段状态：含 definite → ``covered``；仅含 possible → ``uncertain``；
  二者皆空 → ``uncovered``；
- 相邻片段只在**状态与 definite / possible 两组 ID 完全相同**时合并。

所有偏移均按 Unicode 码点计（Python 字符串索引即码点索引）。
相同的文本与范围输入必然得到字节级一致的输出：边界位置按升序处理，
ID 一律升序输出。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class CoverageRange:
    """一条批注在目标版本上计入覆盖图的码点区间 ``[start, end)``。

    ``group`` 为 ``"definite"``（resolved / confirmed 生效范围）或
    ``"possible"``（未确认、非顺延 ambiguous 的候选区间）。
    """

    annotation_id: int
    start: int
    end: int


def merge_candidate_ranges(ranges: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """把同一批注的候选范围中严格重叠者合并为连续最大区间（按起点升序）。

    仅严格重叠（``next.start < prev.end``）才合并；首尾相接（半开区间
    ``next.start == prev.end``）的候选保持独立——相接处本就没有空隙，
    扫描线在接点两侧会得到完全相同的 ID 集合，相邻片段在合并阶段自然
    拼成一段。合并结果中同一批注的区间两两不相交。
    """
    ordered = sorted(ranges, key=lambda r: (r[0], r[1]))
    merged: List[Tuple[int, int]] = []
    for start, end in ordered:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _segment_status(definite: frozenset, possible: frozenset) -> str:
    if definite:
        return "covered"
    if possible:
        return "uncertain"
    return "uncovered"


def build_coverage(
    text: str,
    definite: Sequence[CoverageRange],
    possible: Sequence[CoverageRange],
) -> Dict[str, Any]:
    """扫描线把 ``[0, len(text))`` 切成状态与两组 ID 恒定的最大片段。

    ``definite`` 每条批注至多一个区间；``possible`` 中同一批注的区间必须
    已由 :func:`merge_candidate_ranges` 合并为两两不相交（端点相接不算
    相交）。返回 ``source_length`` / ``segment_count`` / ``summary`` /
    ``segments``；每个片段含范围、原文切片、升序的 ``definite_ids`` /
    ``possible_ids`` 与状态，片段首尾相接、无缝铺满全文。
    """
    # 事件表：position -> [(group, annotation_id, delta)]，group 0=definite, 1=possible
    events: Dict[int, List[Tuple[int, int, int]]] = {}

    def add_event(pos: int, group: int, annotation_id: int, delta: int) -> None:
        events.setdefault(pos, []).append((group, annotation_id, delta))

    for r in definite:
        add_event(r.start, 0, r.annotation_id, 1)
        add_event(r.end, 0, r.annotation_id, -1)
    for r in possible:
        add_event(r.start, 1, r.annotation_id, 1)
        add_event(r.end, 1, r.annotation_id, -1)

    active: Tuple[set, set] = (set(), set())
    raw: List[Tuple[int, int, frozenset, frozenset]] = []
    prev = 0
    for pos in sorted(events):
        # [prev, pos) 内两组 ID 集合恒定
        if pos > prev:
            raw.append((prev, pos, frozenset(active[0]), frozenset(active[1])))
        # 同一位置的全部事件一起生效，接点两侧的净变化才准确
        for group, annotation_id, delta in events[pos]:
            if delta == 1:
                active[group].add(annotation_id)
            else:
                active[group].discard(annotation_id)
        prev = pos
    if prev < len(text):
        raw.append((prev, len(text), frozenset(active[0]), frozenset(active[1])))

    # 相邻片段仅在状态与两组 ID 完全相同时合并（raw 已保证首尾相接）
    merged: List[Tuple[int, int, frozenset, frozenset, str]] = []
    for start, end, dset, pset in raw:
        status = _segment_status(dset, pset)
        if (
            merged
            and merged[-1][1] == start
            and merged[-1][2] == dset
            and merged[-1][3] == pset
            and merged[-1][4] == status
        ):
            merged[-1] = (merged[-1][0], end, dset, pset, status)
        else:
            merged.append((start, end, dset, pset, status))

    # 空文本：以一个零长度 uncovered 片段显式覆盖 [0, 0)
    if not merged and len(text) == 0:
        merged.append((0, 0, frozenset(), frozenset(), "uncovered"))

    segments: List[Dict[str, Any]] = []
    summary = {"covered": 0, "uncertain": 0, "uncovered": 0}
    for index, (start, end, dset, pset, status) in enumerate(merged):
        summary[status] += 1
        segments.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "text": text[start:end],
                "definite_ids": sorted(dset),
                "possible_ids": sorted(pset),
                "status": status,
            }
        )

    return {
        "source_length": len(text),
        "segment_count": len(segments),
        "summary": summary,
        "segments": segments,
    }
