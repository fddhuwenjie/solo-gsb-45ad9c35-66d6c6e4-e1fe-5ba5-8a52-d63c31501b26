"""文档批注覆盖图（GET /documents/{id}/coverage）纯函数与端到端测试。"""
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.coverage import (
    CoverageRange,
    build_coverage,
    merge_candidate_ranges,
)


# ---------------------------------------------------------------------------
# 纯函数：按批注合并重叠候选
# ---------------------------------------------------------------------------


def test_merge_candidates_strict_overlap_only():
    # (0,3) 与 (2,5) 严格重叠，再链上 (4,8) → 合并为 (0,8)
    assert merge_candidate_ranges([(4, 8), (0, 3), (2, 5)]) == [(0, 8)]
    # 首尾相接（半开区间端点相等）不合并
    assert merge_candidate_ranges([(0, 2), (2, 4)]) == [(0, 2), (2, 4)]


def test_merge_candidates_contained_and_disjoint():
    # 乱序按起点升序；[2,6) 被 [0,10) 包含，[7,8) 与之重叠 → 合并为 (0,10)
    assert merge_candidate_ranges([(2, 6), (0, 10), (7, 8)]) == [(0, 10)]
    # 互不相交的区间按起点升序原样保留
    assert merge_candidate_ranges([(7, 8), (0, 3), (4, 5)]) == [
        (0, 3),
        (4, 5),
        (7, 8),
    ]


def test_merge_candidates_duplicates():
    assert merge_candidate_ranges([(3, 5), (3, 5), (3, 5)]) == [(3, 5)]


def test_merge_candidates_disjoint_sorted():
    assert merge_candidate_ranges([(20, 30), (0, 4), (10, 12)]) == [
        (0, 4),
        (10, 12),
        (20, 30),
    ]


# ---------------------------------------------------------------------------
# 纯函数：覆盖图
# ---------------------------------------------------------------------------


def _seg_view(segments):
    return [
        (
            s["start"],
            s["end"],
            s["status"],
            tuple(s["definite_ids"]),
            tuple(s["possible_ids"]),
        )
        for s in segments
    ]


def test_build_coverage_statuses_and_seamless_cover():
    text = "abcdefghij"
    definite = [CoverageRange(1, 1, 3)]
    possible = [
        CoverageRange(2, 2, 6),  # 已按批注合并后的区间
        CoverageRange(3, 8, 9),
    ]
    result = build_coverage(text, definite, possible)
    assert _seg_view(result["segments"]) == [
        (0, 1, "uncovered", (), ()),
        (1, 2, "covered", (1,), ()),
        (2, 3, "covered", (1,), (2,)),
        (3, 6, "uncertain", (), (2,)),
        (6, 8, "uncovered", (), ()),
        (8, 9, "uncertain", (), (3,)),
        (9, 10, "uncovered", (), ()),
    ]
    assert result["source_length"] == 10
    assert result["summary"] == {"covered": 2, "uncertain": 2, "uncovered": 3}
    # 无缝覆盖
    assert result["segments"][0]["start"] == 0
    assert result["segments"][-1]["end"] == len(text)
    for prev, nxt in zip(result["segments"], result["segments"][1:]):
        assert prev["end"] == nxt["start"]
        # 相邻片段状态与两组 ID 必不相同（否则应已合并）
        assert (
            prev["status"] != nxt["status"]
            or prev["definite_ids"] != nxt["definite_ids"]
            or prev["possible_ids"] != nxt["possible_ids"]
        )
    assert "".join(s["text"] for s in result["segments"]) == text


def test_build_coverage_adjacent_segments_with_same_state_merge():
    # 同一批注首尾相接的可能区间：接点两侧 ID 集合相同 → 合成最大片段
    merged = merge_candidate_ranges([(0, 2), (2, 4)])
    result = build_coverage(
        "abcdef", [], [CoverageRange(7, *iv) for iv in merged]
    )
    assert _seg_view(result["segments"]) == [
        (0, 4, "uncertain", (), (7,)),
        (4, 6, "uncovered", (), ()),
    ]


def test_build_coverage_definite_takes_precedence():
    result = build_coverage(
        "ab", [CoverageRange(1, 0, 2)], [CoverageRange(2, 0, 2)]
    )
    seg = result["segments"][0]
    assert seg["status"] == "covered"
    assert seg["definite_ids"] == [1]
    assert seg["possible_ids"] == [2]


def test_build_coverage_ids_sorted_and_deterministic():
    r1 = build_coverage(
        "abc",
        [CoverageRange(30, 0, 1), CoverageRange(2, 0, 1), CoverageRange(9, 0, 1)],
        [CoverageRange(40, 1, 2), CoverageRange(1, 1, 2)],
    )
    r2 = build_coverage(
        "abc",
        [CoverageRange(9, 0, 1), CoverageRange(2, 0, 1), CoverageRange(30, 0, 1)],
        [CoverageRange(1, 1, 2), CoverageRange(40, 1, 2)],
    )
    assert json.dumps(r1, ensure_ascii=False, sort_keys=True) == json.dumps(
        r2, ensure_ascii=False, sort_keys=True
    )
    assert r1["segments"][0]["definite_ids"] == [2, 9, 30]
    assert r1["segments"][1]["possible_ids"] == [1, 40]


def test_build_coverage_no_annotations():
    result = build_coverage("原样文本", [], [])
    assert result["segment_count"] == 1
    assert result["segments"][0]["status"] == "uncovered"
    assert (result["segments"][0]["start"], result["segments"][0]["end"]) == (
        0,
        4,
    )


def test_build_coverage_empty_text():
    result = build_coverage("", [], [])
    assert result["source_length"] == 0
    assert result["segments"] == [
        {
            "index": 0,
            "start": 0,
            "end": 0,
            "text": "",
            "definite_ids": [],
            "possible_ids": [],
            "status": "uncovered",
        }
    ]


def test_build_coverage_unicode_code_points():
    text = "a🎉b"  # 🎉 为 BMP 外字符，仍算 1 个码点
    result = build_coverage(text, [CoverageRange(1, 1, 2)], [])
    assert _seg_view(result["segments"]) == [
        (0, 1, "uncovered", (), ()),
        (1, 2, "covered", (1,), ()),
        (2, 3, "uncovered", (), ()),
    ]
    assert result["segments"][1]["text"] == "🎉"


# ---------------------------------------------------------------------------
# 端到端接口
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _make_doc(client, content, title="覆盖图测试文档"):
    r = client.post("/documents", json={"title": title, "content": content})
    assert r.status_code == 201, r.text
    return r.json()["document_id"]


def _make_ann(client, doc, text, selection, index=0, left="", right="", body="批注"):
    start = -1
    for _ in range(index + 1):
        start = text.index(selection, start + 1)
    r = client.post(
        f"/documents/{doc}/annotations",
        json={
            "body": body,
            "selection": selection,
            "left_context": left,
            "right_context": right,
            "start": start,
            "end": start + len(selection),
        },
    )
    assert r.status_code == 201, r.text
    return r.json()["annotation_id"]


def _new_version(client, doc, content):
    r = client.post(f"/documents/{doc}/versions", json={"content": content})
    assert r.status_code == 201, r.text
    return r.json()


def _coverage(client, doc, version):
    r = client.get(f"/documents/{doc}/coverage", params={"version": version})
    assert r.status_code == 200, r.text
    return r.json()


def _assert_seamless(data, text):
    segs = data["segments"]
    assert segs[0]["start"] == 0
    assert segs[-1]["end"] == len(text)
    for prev, nxt in zip(segs, segs[1:]):
        assert prev["end"] == nxt["start"]
        assert not (
            prev["status"] == nxt["status"]
            and prev["definite_ids"] == nxt["definite_ids"]
            and prev["possible_ids"] == nxt["possible_ids"]
        )
    assert "".join(s["text"] for s in segs) == text


def test_api_resolved_coverage(client):
    text = "开头张三丰结尾"
    doc = _make_doc(client, text)
    a1 = _make_ann(client, doc, text, "张三丰")  # [2,5)
    data = _coverage(client, doc, 1)
    assert data["source_length"] == len(text)
    assert data["exclusions"] == []
    assert _seg_view(data["segments"]) == [
        (0, 2, "uncovered", (), ()),
        (2, 5, "covered", (a1,), ()),
        (5, len(text), "uncovered", (), ()),
    ]
    assert data["segments"][1]["text"] == "张三丰"
    _assert_seamless(data, text)
    assert data["summary"]["definite_annotation_count"] == 1


def test_api_definite_overlaps_and_abuts_merge_maximally(client):
    text = "甲乙丙丁戊"
    doc = _make_doc(client, text)
    a1 = _make_ann(client, doc, text, "甲乙丙")  # [0,3)
    a2 = _make_ann(client, doc, text, "丙丁")    # [2,4) 与 a1 重叠
    a3 = _make_ann(client, doc, text, "戊")      # [4,5) 首尾相接
    data = _coverage(client, doc, 1)
    assert _seg_view(data["segments"]) == [
        (0, 2, "covered", (a1,), ()),
        (2, 3, "covered", (a1, a2), ()),
        (3, 4, "covered", (a2,), ()),
        (4, 5, "covered", (a3,), ()),
    ]
    # 状态同为 covered 但两组 ID 集合在每段不同 → 不合并；ID 始终升序
    _assert_seamless(data, text)


def test_api_ambiguous_unconfirmed_candidates_are_possible(client):
    # “目标”两处并列最高分（无上下文区分）→ ambiguous，候选 [0,2)/[4,6)
    text = "目标AA目标"
    doc = _make_doc(client, text)
    a1 = _make_ann(client, doc, text, "目标")
    _new_version(client, doc, text)  # 同文本再提交一版触发重新匹配 → ambiguous
    data = _coverage(client, doc, 2)
    assert data["exclusions"] == []
    assert _seg_view(data["segments"]) == [
        (0, 2, "uncertain", (), (a1,)),
        (2, 4, "uncovered", (), ()),
        (4, 6, "uncertain", (), (a1,)),
    ]
    assert data["summary"]["possible_annotation_count"] == 1
    _assert_seamless(data, text)


def test_api_ambiguous_overlapping_candidates_same_id_not_duplicated(client):
    # 单字符选区在连续重复串中大量重叠候选：合并后同一 ID 不重复出现
    text = "AAAA"
    doc = _make_doc(client, text)
    a1 = _make_ann(client, doc, text, "A")  # 创建版 resolved [0,1)
    _new_version(client, doc, text)         # 4 个落点同分 → ambiguous
    data = _coverage(client, doc, 2)
    assert data["exclusions"] == []
    # 候选全部重叠链 → 合并为 [0,4)，整段 uncertain，每处 ID 只列一次
    assert _seg_view(data["segments"]) == [
        (0, 4, "uncertain", (), (a1,)),
    ]
    assert data["segments"][0]["possible_ids"] == [a1]
    _assert_seamless(data, text)


def test_api_confirmed_ambiguous_becomes_definite(client):
    text = "目标AA目标"
    doc = _make_doc(client, text)
    a1 = _make_ann(client, doc, text, "目标")
    _new_version(client, doc, text)  # ambiguous
    chosen = client.get(f"/documents/{doc}/versions/2").json()["locations"][0][
        "candidates"
    ][1]  # 确认后一处 [6,8)
    cr = client.post(
        f"/documents/{doc}/annotations/{a1}/confirmations",
        json={"version_number": 2, "start": chosen["start"], "end": chosen["end"]},
    )
    assert cr.status_code == 201, cr.text

    data = _coverage(client, doc, 2)
    assert data["exclusions"] == []
    assert _seg_view(data["segments"]) == [
        (0, 4, "uncovered", (), ()),
        (4, 6, "covered", (a1,), ()),
    ]


def test_api_carried_forward_placeholder_excluded(client):
    text = "目标AA目标"
    doc = _make_doc(client, text)
    a1 = _make_ann(client, doc, text, "目标")
    _new_version(client, doc, text)  # v2 ambiguous，不确认
    _new_version(client, doc, text)  # v3 顺延占位

    data = _coverage(client, doc, 3)
    # 全文 uncovered；顺延占位进入 exclusions
    assert all(s["status"] == "uncovered" for s in data["segments"])
    assert len(data["exclusions"]) == 1
    exc = data["exclusions"][0]
    assert exc["annotation_id"] == a1
    assert exc["reason"] == "ambiguous_carried_forward"
    assert exc["status"] == "ambiguous"
    assert exc["carried_from_location_id"] is not None
    assert "顺延" in exc["message"]
    assert data["summary"]["excluded_annotation_count"] == 1
    _assert_seamless(data, text)


def test_api_unresolved_excluded(client):
    text = "甲乙"
    doc = _make_doc(client, text)
    a1 = _make_ann(client, doc, text, "甲")
    _new_version(client, doc, "乙乙")  # 甲消失 → unresolved
    data = _coverage(client, doc, 2)
    assert all(s["status"] == "uncovered" for s in data["segments"])
    assert len(data["exclusions"]) == 1
    exc = data["exclusions"][0]
    assert exc["annotation_id"] == a1
    assert exc["reason"] == "unresolved"
    assert exc["status"] == "unresolved"
    assert "未解析" in exc["message"]
    assert "carried_from_location_id" not in exc
    _assert_seamless(data, "乙乙")


def test_api_annotation_created_later_is_location_missing(client):
    doc = _make_doc(client, "第一版文本")
    _new_version(client, doc, "第二版文本")
    a1 = _make_ann(client, doc, "第二版文本", "第二版")  # 创建在最新（第 2）版
    data = _coverage(client, doc, 1)
    assert all(s["status"] == "uncovered" for s in data["segments"])
    exc = data["exclusions"][0]
    assert exc["annotation_id"] == a1
    assert exc["reason"] == "location_missing"
    assert exc["status"] is None
    assert "更晚版本" in exc["message"]
    _assert_seamless(data, "第一版文本")


def test_api_mixed_statuses_in_one_map(client):
    # v2：a2 ambiguous（possible）、a1 与 a3 unresolved（excluded）、
    # a4 创建于更晚版本（location_missing）
    v1 = "甲乙甲丙"
    doc = _make_doc(client, v1)
    a1 = _make_ann(client, doc, v1, "乙")          # [1,2)
    a2 = _make_ann(client, doc, v1, "甲", index=0)  # [0,1)
    a3 = _make_ann(client, doc, v1, "丙")           # [3,4)
    v2 = "甲甲甲"  # “甲”三处并列 → a2 ambiguous；“乙”“丙”消失 → unresolved
    _new_version(client, doc, v2)
    _new_version(client, doc, v2 + "X")
    a4 = _make_ann(client, doc, v2 + "X", "X")      # 创建在第 3 版

    data = _coverage(client, doc, 2)
    by_id = {e["annotation_id"]: e for e in data["exclusions"]}
    assert by_id[a1]["reason"] == "unresolved"
    assert by_id[a3]["reason"] == "unresolved"
    assert by_id[a4]["reason"] == "location_missing"
    # a2 的三个候选 [0,1)/[1,2)/[2,3) 首尾相接 → 合并为整段 uncertain
    assert _seg_view(data["segments"]) == [
        (0, 3, "uncertain", (), (a2,)),
    ]
    summary = data["summary"]
    assert summary["definite_annotation_count"] == 0
    assert summary["possible_annotation_count"] == 1
    assert summary["excluded_annotation_count"] == 3
    assert summary["annotation_count"] == 4
    _assert_seamless(data, v2)


def test_api_pins_version(client):
    doc = _make_doc(client, "目标在这")
    a1 = _make_ann(client, doc, "目标在这", "目标")  # [0,2)
    _new_version(client, doc, "开头目标在这")        # resolved [2,4)
    d1 = _coverage(client, doc, 1)
    d2 = _coverage(client, doc, 2)
    covered_1 = [s for s in d1["segments"] if s["status"] == "covered"]
    covered_2 = [s for s in d2["segments"] if s["status"] == "covered"]
    assert [(s["start"], s["end"]) for s in covered_1] == [(0, 2)]
    assert [(s["start"], s["end"]) for s in covered_2] == [(2, 4)]


def test_api_read_only_and_byte_stable(client):
    text = "甲乙丙丁"
    doc = _make_doc(client, text)
    a1 = _make_ann(client, doc, text, "乙")
    a2 = _make_ann(client, doc, text, "丁")
    r1 = client.get(f"/documents/{doc}/coverage", params={"version": 1})
    r2 = client.get(f"/documents/{doc}/coverage", params={"version": 1})
    assert r1.status_code == r2.status_code == 200
    b1 = json.dumps(r1.json(), ensure_ascii=False, sort_keys=True,
                    separators=(",", ":")).encode("utf-8")
    b2 = json.dumps(r2.json(), ensure_ascii=False, sort_keys=True,
                    separators=(",", ":")).encode("utf-8")
    assert b1 == b2
    # 只读：覆盖图不新增迁移记录 / 批注
    locs = client.get(f"/documents/{doc}/versions/1").json()["locations"]
    assert [l["annotation_id"] for l in locs] == [a1, a2]
    assert client.get(f"/documents/{doc}").json()["annotation_count"] == 2


def test_api_coverage_document_or_version_missing(client):
    doc = _make_doc(client, "abc")
    r = client.get(f"/documents/{doc}/coverage", params={"version": 99})
    assert r.status_code == 404
    r = client.get("/documents/999999/coverage", params={"version": 1})
    assert r.status_code == 404


def test_api_coverage_rejects_invalid_version(client):
    doc = _make_doc(client, "abc")
    r = client.get(f"/documents/{doc}/coverage", params={"version": 0})
    assert r.status_code == 422
