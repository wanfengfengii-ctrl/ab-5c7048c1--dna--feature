"""HTTP-level integration tests for POST /api/phase and /health."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def make_read(rid, start, end, obs, costs=None, allow=None, molecule_id=None):
    width = end - start
    read = {
        "id": rid,
        "start": start,
        "end": end,
        "observations": list(obs),
        "mismatch_costs": list(costs) if costs is not None else [1] * width,
        "max_mismatches": width if allow is None else allow,
    }
    if molecule_id is not None:
        read["molecule_id"] = molecule_id
    return read


def clean_payload():
    hap = [0, 1, 1, 0, 1, 0, 0, 1]
    comp = [1 - b for b in hap]
    spans0 = [(0, 3), (2, 5), (4, 7), (1, 4), (5, 8)]
    spans1 = [(0, 2), (3, 6), (6, 8), (2, 4), (4, 8)]
    reads = []
    for i, (s, e) in enumerate(spans0):
        reads.append(make_read(f"a{i}", s, e, hap[s:e], allow=0))
    for i, (s, e) in enumerate(spans1):
        reads.append(make_read(f"b{i}", s, e, comp[s:e], allow=0))
    return {"n_sites": 8, "reads": reads}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_phase_success_envelope():
    r = client.post("/api/phase", json=clean_payload())
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    data = body["data"]
    assert data["unique"] is True
    assert data["solution"]["total_mismatch_cost"] == 0
    assert len(data["solution"]["per_read"]) == 10
    # canonical first site
    assert data["solution"]["haplotype"][0] == 0
    # assignments and evidence present per read
    for row in data["solution"]["per_read"]:
        assert set(row) == {"id", "group", "mismatch_count", "mismatch_cost", "mismatch_positions"}
        assert row["group"] in (0, 1)


def test_phase_discontinuous_returns_business_code():
    payload = clean_payload()
    # shrink all group-1 reads so site 7 is uncovered (still 10 reads)
    for r in payload["reads"]:
        if r["id"] == "a4":
            r["end"] = 7
            r["observations"].pop()
            r["mismatch_costs"].pop()
        if r["id"] == "b2":
            r["start"] = 6
            r["end"] = 7
            r["observations"] = [0]
            r["mismatch_costs"] = [1]
            r["max_mismatches"] = 0
        if r["id"] == "b4":
            r["start"] = 4
            r["end"] = 7
            r["observations"] = [1, 0, 0]
            r["mismatch_costs"] = [1, 1, 1]
            r["max_mismatches"] = 0
    r = client.post("/api/phase", json=payload)
    assert r.status_code == 409
    body = r.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "DISCONTINUOUS_INPUT"


def test_phase_no_solution_returns_business_code():
    reads = []
    pattern = [0, 0, 1, 1, 0, 1, 0, 1]
    other = [1, 1, 0, 0, 1, 0, 1, 0]
    third = [0, 1, 0, 1, 0, 1, 0, 1]
    for j, p in enumerate([pattern, other, third]):
        reads.append(make_read(f"x{j}", 0, 8, p, allow=0))
    for j in range(7):
        p = pattern if j % 2 == 0 else other
        reads.append(make_read(f"p{j}", 0, 8, p, allow=0))
    r = client.post("/api/phase", json={"n_sites": 8, "reads": reads})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "NO_SOLUTION"


def test_phase_invalid_input_422():
    payload = clean_payload()
    payload["n_sites"] = 5
    r = client.post("/api/phase", json=payload)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_INPUT"


def test_bad_json_400():
    r = client.post("/api/phase", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "BAD_JSON"


def test_ambiguous_payload_reports_two_solutions():
    reads = []
    a_specs = [
        ("a0", 0, 2, [0, 0]),
        ("a1", 1, 4, [0, 1, 1]),
        ("a2", 0, 3, [0, 0, 1]),
        ("a3", 2, 4, [0, 0]),
        ("a4", 0, 2, [1, 1]),
    ]
    b_specs = [
        ("b0", 4, 6, [0, 0]),
        ("b1", 5, 8, [0, 1, 1]),
        ("b2", 4, 7, [0, 0, 1]),
        ("b3", 6, 8, [0, 0]),
        ("b4", 4, 6, [1, 1]),
    ]
    for rid, s, e, obs in a_specs + b_specs:
        reads.append(make_read(rid, s, e, obs, allow=0))
    r = client.post("/api/phase", json={"n_sites": 8, "reads": reads})
    body = r.json()
    assert r.status_code == 200
    assert body["data"]["unique"] is False
    assert len(body["data"]["solutions"]) == 2
    assert body["data"]["note"]


def _family_payload():
    """Clean 10-read instance with five 2-member molecule families."""
    hap = [0, 1, 1, 0, 1, 0, 0, 1]
    comp = [1 - b for b in hap]
    spans0 = [(0, 3), (2, 5), (4, 7), (1, 4), (5, 8)]
    spans1 = [(0, 2), (3, 6), (6, 8), (2, 4), (4, 8)]
    reads = []
    for i, (s, e) in enumerate(spans0):
        allow = len(hap[s:e]) if i == 4 else 0
        reads.append(make_read(f"a{i}", s, e, hap[s:e], allow=allow))
    for i, (s, e) in enumerate(spans1):
        allow = len(comp[s:e]) if i == 4 else 0
        reads.append(make_read(f"b{i}", s, e, comp[s:e], allow=allow))
    pairs = [("a0", "a1"), ("a2", "a3"), ("b0", "b1"), ("b2", "b3"), ("a4", "b4")]
    mol_by_id = {rid: f"mol{k}" for k, pair in enumerate(pairs) for rid in pair}
    for r in reads:
        r["molecule_id"] = mol_by_id[r["id"]]
    return {"n_sites": 8, "reads": reads}


def test_phase_family_atomic_and_evidence():
    r = client.post("/api/phase", json=_family_payload())
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    sol = data["solution"]
    groups = {row["id"]: row["group"] for row in sol["per_read"]}
    # every family is assigned as a whole
    for block in sol["families"]:
        g = block["group"]
        assert all(groups[m] == g for m in block["members"])
        assert 2 <= len(block["members"]) <= 4
    # group bounds still computed on read counts
    assert len(sol["groups"]["haplotype"]) >= 2
    assert len(sol["groups"]["complement"]) >= 2
    # family aggregates reconcile with member evidence
    per_read = {row["id"]: row for row in sol["per_read"]}
    for block in sol["families"]:
        assert block["mismatch_cost"] == sum(
            per_read[m]["mismatch_cost"] for m in block["members"]
        )
        assert block["mismatch_count"] == sum(
            per_read[m]["mismatch_count"] for m in block["members"]
        )
    # per-read rows keep the legacy key set
    for row in sol["per_read"]:
        assert set(row) == {
            "id", "group", "mismatch_count", "mismatch_cost", "mismatch_positions"
        }


def test_phase_family_changes_optimum_vs_unlabelled():
    labelled = _family_payload()
    unlabelled = {
        "n_sites": 8,
        "reads": [
            {k: v for k, v in r.items() if k != "molecule_id"}
            for r in labelled["reads"]
        ],
    }
    free = client.post("/api/phase", json=unlabelled).json()["data"]
    fam = client.post("/api/phase", json=labelled).json()["data"]
    assert free["solution"]["total_mismatch_cost"] == 0
    assert fam["solution"]["total_mismatch_cost"] > 0
    # both reported solutions keep the cross-group pair a4/b4 together
    for sol in fam["solutions"]:
        g = {row["id"]: row["group"] for row in sol["per_read"]}
        assert g["a4"] == g["b4"]
    # unlabelled response still carries the empty family list without
    # changing any legacy field
    assert free["solution"]["families"] == []


def test_phase_invalid_family_size_422():
    payload = _family_payload()
    payload["reads"][0]["molecule_id"] = "lonely"  # singleton label
    r = client.post("/api/phase", json=payload)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_INPUT"


def test_phase_family_no_solution_409():
    hap = [0, 0, 1, 1, 0, 1, 0, 1]
    comp = [1 - b for b in hap]
    reads = [
        make_read("f0a", 0, 8, hap, allow=0, molecule_id="f0"),
        make_read("f0b", 0, 8, comp, allow=0, molecule_id="f0"),
    ]
    for k in range(4):
        reads.append(make_read(f"p{k}", 0, 4, hap[0:4], allow=0))
    for k in range(4):
        reads.append(make_read(f"q{k}", 0, 4, comp[0:4], allow=0))
    r = client.post("/api/phase", json={"n_sites": 8, "reads": reads})
    assert r.status_code == 409
    body = r.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "FAMILY_NO_SOLUTION"
