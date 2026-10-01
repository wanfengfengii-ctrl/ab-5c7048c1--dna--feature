"""Unit tests for the phasing solver, including exhaustive cross-checks."""

from __future__ import annotations

import random

import pytest

from app.solver import PhaseError, enumerate_solutions, parse_input, phase


# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------


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


def clean_instance():
    """Two clean complementary haplotypes, 8 sites, 10 reads, no damage."""
    hap = [0, 1, 1, 0, 1, 0, 0, 1]
    comp = [1 - b for b in hap]
    # group-0 reads
    spans0 = [(0, 3), (2, 5), (4, 7), (1, 4), (5, 8)]
    # group-1 reads
    spans1 = [(0, 2), (3, 6), (6, 8), (2, 4), (4, 8)]
    reads = []
    for i, (s, e) in enumerate(spans0):
        reads.append(make_read(f"a{i}", s, e, hap[s:e], allow=0))
    for i, (s, e) in enumerate(spans1):
        reads.append(make_read(f"b{i}", s, e, comp[s:e], allow=0))
    return 8, reads, hap


def random_instance(n_sites, m, seed):
    """Generate a random covered read set with damaged observations."""
    rng = random.Random(seed)
    hap = [rng.randrange(2) for _ in range(n_sites)]
    comp = [1 - b for b in hap]

    reads = []
    for _attempt in range(200):
        reads = []
        for j in range(m):
            length = rng.randint(2, min(5, n_sites))
            start = rng.randint(0, n_sites - length)
            src = hap if rng.randrange(2) == 0 else comp
            obs = list(src[start : start + length])
            flips = 0
            for k in range(length):
                if rng.random() < 0.15:
                    obs[k] ^= 1
                    flips += 1
            costs = [rng.randint(1, 9) for _ in range(length)]
            # allowance is sometimes tight, sometimes loose; feasibility is
            # left for the solver to decide
            allow = rng.choice([0, flips, flips, min(flips + 1, length), length])
            reads.append(make_read(f"r{j}", start, start + length, obs, costs, allow))
        covered = [False] * n_sites
        for r in reads:
            for s in range(r["start"], r["end"]):
                covered[s] = True
        if all(covered):
            return n_sites, reads
    raise AssertionError("could not generate a covering instance")


def brute_force(n_sites, reads):
    """Enumerate every (canonical haplotype, assignment) pair.

    Returns the sorted list of optimum distinct solutions as
    ``(total, maxmm, hap_tuple, assignment_tuple)``.
    """
    parsed = parse_input({"n_sites": n_sites, "reads": reads})
    n_sites, pr, families = parsed
    families_by_read = {i: mol for mol, members in families for i in members}
    n = len(pr)
    best = None
    winners = set()
    for hap_bits in range(1 << (n_sites - 1)):
        hap = (0,) + tuple((hap_bits >> (n_sites - 2 - s)) & 1 for s in range(n_sites - 1))
        # Enumerate assignments per family unit: every member of a family
        # shares one group bit.  Unlabelled reads keep their own bit.
        seen_assignments: set[tuple[int, ...]] = set()
        family_mols = [mol for mol, _m in families]
        n_fam = len(families)
        n_solo = n - sum(len(m) for _mol, m in families)
        for ub in range(1 << (n_fam + n_solo)):
            fam_bits = ub >> n_solo
            assign = []
            solo_k = 0
            for i in range(n):
                if i in families_by_read:
                    fi = family_mols.index(families_by_read[i])
                    assign.append((fam_bits >> (n_fam - 1 - fi)) & 1)
                else:
                    assign.append((ub >> (n_solo - 1 - solo_k)) & 1)
                    solo_k += 1
            assign = tuple(assign)
            if assign in seen_assignments:
                continue
            seen_assignments.add(assign)
            if assign.count(0) < 2 or assign.count(1) < 2:
                continue
            total = 0
            maxmm = 0
            ok = True
            for r, g in zip(pr, assign):
                mm = 0
                for k, site in enumerate(range(r.start, r.end)):
                    mismatch = (r.obs[k] != hap[site]) if g == 0 else (r.obs[k] == hap[site])
                    if mismatch:
                        mm += 1
                        total += r.costs[k]
                if mm > r.max_mismatches:
                    ok = False
                    break
                maxmm = max(maxmm, mm)
            if not ok:
                continue
            key = (total, maxmm)
            if best is None or key < best:
                best = key
                winners = {(total, maxmm, hap, assign)}
            elif key == best:
                winners.add((total, maxmm, hap, assign))
    # Canonical solution order: haplotype first, then assignment.
    return sorted(winners, key=lambda r: (r[2], r[3]))


# --------------------------------------------------------------------------
# basic behaviour
# --------------------------------------------------------------------------


def test_clean_instance_unique_zero_cost():
    n_sites, reads, hap = clean_instance()
    out = phase({"n_sites": n_sites, "reads": reads})
    assert out["unique"] is True
    sol = out["solution"]
    assert sol["total_mismatch_cost"] == 0
    assert sol["max_per_read_mismatches"] == 0
    assert sol["haplotype"][0] == 0  # canonical
    assert sol["haplotype"] == hap or sol["haplotype"] == [1 - b for b in hap]
    # canonical normalization: the returned haplotype itself must be the
    # oriented one; its first bit is fixed to 0
    assert sol["haplotype"][0] == 0
    assert len(sol["groups"]["haplotype"]) >= 2
    assert len(sol["groups"]["complement"]) >= 2
    assert len(out["solutions"]) == 1


def test_mismatch_evidence_is_consistent():
    n_sites, reads, _ = clean_instance()
    # damage two observations on reads a0 and b2, each with an allowance of 1
    reads[0]["observations"][0] ^= 1
    reads[0]["mismatch_costs"][0] = 7
    reads[0]["max_mismatches"] = 1
    reads[7]["observations"][1] ^= 1
    reads[7]["mismatch_costs"] = [5, 3]
    reads[7]["max_mismatches"] = 1

    out = phase({"n_sites": n_sites, "reads": reads})
    sol = out["solution"]
    assert sol["total_mismatch_cost"] == 10
    assert sol["max_per_read_mismatches"] == 1

    by_id = {row["id"]: row for row in sol["per_read"]}
    assert by_id["a0"]["mismatch_count"] == 1
    assert by_id["a0"]["mismatch_cost"] == 7
    assert by_id["a0"]["mismatch_positions"] == [0]
    assert by_id["b2"]["mismatch_count"] == 1
    assert by_id["b2"]["mismatch_cost"] == 3
    assert by_id["b2"]["mismatch_positions"] == [7]

    # evidence totals reconcile
    assert sum(r["mismatch_cost"] for r in sol["per_read"]) == sol["total_mismatch_cost"]
    assert max(r["mismatch_count"] for r in sol["per_read"]) == sol["max_per_read_mismatches"]
    # assignments reconcile with the group member lists
    for row in sol["per_read"]:
        side = "haplotype" if row["group"] == 0 else "complement"
        assert row["id"] in sol["groups"][side]
    # haplotype and complement are bitwise complements
    assert sol["complement"] == [1 - b for b in sol["haplotype"]]


def test_higher_cost_does_not_buy_lower_max():
    """Secondary objective is optimized only among minimum-cost solutions."""
    n_sites, reads, _ = clean_instance()
    # cheap single damage vs an alternative assignment that would be dearer
    reads[0]["observations"][0] ^= 1
    reads[0]["mismatch_costs"] = [1, 100, 100]
    reads[0]["max_mismatches"] = 2
    out = phase({"n_sites": n_sites, "reads": reads})
    assert out["solution"]["total_mismatch_cost"] == 1


# --------------------------------------------------------------------------
# ambiguity
# --------------------------------------------------------------------------


def test_ambiguous_unlinked_components():
    """Two read blocks with no spanning read => relative phase is ambiguous.

    Block A covers sites 0..3 (haplotype 0011) and block B sites 4..7
    (haplotype 0011).  Each block contains reads from both chromosomes, so
    both groups are populated either way; flipping block B produces a second
    distinct zero-cost canonical haplotype, while flipping block A would put
    a 1 at site 0 and is already quotiented by canonicalization.
    """
    n_sites = 8
    reads = []
    # Block A, sites 0..3, hap 0011 / comp 1100
    a_specs = [
        ("a0", 0, 2, [0, 0]),       # group 0
        ("a1", 1, 4, [0, 1, 1]),    # group 0
        ("a2", 0, 3, [0, 0, 1]),    # group 0
        ("a3", 2, 4, [0, 0]),       # group 1 (comp tail)
        ("a4", 0, 2, [1, 1]),       # group 1 (comp head)
    ]
    # Block B, sites 4..7, hap 0011 / comp 1100
    b_specs = [
        ("b0", 4, 6, [0, 0]),       # group 0
        ("b1", 5, 8, [0, 1, 1]),    # group 0
        ("b2", 4, 7, [0, 0, 1]),    # group 0
        ("b3", 6, 8, [0, 0]),       # group 1
        ("b4", 4, 6, [1, 1]),       # group 1
    ]
    for rid, s, e, obs in a_specs + b_specs:
        reads.append(make_read(rid, s, e, obs, allow=0))

    out = phase({"n_sites": n_sites, "reads": reads})
    assert out["unique"] is False
    assert len(out["solutions"]) == 2
    s1, s2 = out["solutions"]
    assert s1["total_mismatch_cost"] == s2["total_mismatch_cost"] == 0
    assert s1["max_per_read_mismatches"] == s2["max_per_read_mismatches"] == 0
    assert s1["haplotype"] != s2["haplotype"]
    # first two canonical (lexicographic) solutions
    assert s1["haplotype"] < s2["haplotype"]
    assert len(s1["groups"]["haplotype"]) >= 2
    assert len(s1["groups"]["complement"]) >= 2
    assert len(s2["groups"]["haplotype"]) >= 2
    assert len(s2["groups"]["complement"]) >= 2


# --------------------------------------------------------------------------
# ambiguity from two optimal assignments under one haplotype
# --------------------------------------------------------------------------


def test_ambiguous_two_assignments_under_one_haplotype():
    """A perfectly neutral read can join either group at equal cost.

    The neutral read spans sites 0..1 with observations [0,0]; against the
    true haplotype [0,1] it mismatches once (cost 3), and against the
    complement [1,0] it also mismatches once (cost 3).  Every other read is
    damage-free, so both placements are globally optimal with the same
    haplotype -- two distinct solutions that are NOT swap-equivalent (each
    differs by one read only).
    """
    n_sites, reads, hap = clean_instance()
    # replace read b0 (a clean complement read over [0,2)) with a neutral one
    reads[5] = make_read("b0", 0, 2, [0, 0], costs=[3, 3], allow=1)
    for r in reads:
        if r["id"] != "b0":
            r["max_mismatches"] = 0

    out = phase({"n_sites": n_sites, "reads": reads})
    assert out["unique"] is False
    s1, s2 = out["solutions"]
    assert s1["haplotype"] == s2["haplotype"] == hap
    assert s1["assignments"] != s2["assignments"]
    # they differ exactly in the neutral read's group
    diffs = [i for i, (a, b) in enumerate(zip(s1["assignments"], s2["assignments"])) if a != b]
    assert diffs == [5]
    assert s1["total_mismatch_cost"] == s2["total_mismatch_cost"] == 3
    assert s1["max_per_read_mismatches"] == s2["max_per_read_mismatches"] == 1
    # first solution assigns the earliest-differing read to group 0
    assert s1["assignments"][5] == 0
    for s in (s1, s2):
        assert len(s["groups"]["haplotype"]) >= 2
        assert len(s["groups"]["complement"]) >= 2


# --------------------------------------------------------------------------
# error paths
# --------------------------------------------------------------------------


def test_discontinuous_coverage_rejected():
    n_sites = 8
    reads = []
    # ten reads tiling sites 0..5 and 7, leaving site 6 uncovered
    spans = [(0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (0, 3), (1, 4), (2, 5), (7, 8), (0, 4)]
    for j, (s, e) in enumerate(spans):
        obs = [0] * (e - s)
        reads.append(make_read(f"d{j}", s, e, obs, allow=e - s))
    with pytest.raises(PhaseError) as exc:
        parse_input({"n_sites": n_sites, "reads": reads})
    assert exc.value.code == "DISCONTINUOUS_INPUT"
    assert "6" in exc.value.message


def test_no_solution_when_allowances_too_tight():
    # Three reads over the same span with mutually incompatible observations
    # and zero mismatch allowance: no single binary haplotype can match all.
    n_sites = 8
    reads = []
    pattern = [0, 0, 1, 1, 0, 1, 0, 1]
    other = [1, 1, 0, 0, 1, 0, 1, 0]
    third = [0, 1, 0, 1, 0, 1, 0, 1]
    for j, p in enumerate([pattern, other, third]):
        reads.append(make_read(f"x{j}", 0, 8, p, allow=0))
    # pad with duplicate-origin reads (still incompatible everywhere)
    for j in range(7):
        p = pattern if j % 2 == 0 else other
        reads.append(make_read(f"p{j}", 0, 8, p, allow=0))
    with pytest.raises(PhaseError) as exc:
        phase({"n_sites": n_sites, "reads": reads})
    assert exc.value.code == "NO_SOLUTION"


def test_invalid_counts_and_shapes():
    n_sites, reads, _ = clean_instance()
    with pytest.raises(PhaseError) as ei:
        phase({"n_sites": 7, "reads": reads})
    assert ei.value.code == "INVALID_INPUT"

    with pytest.raises(PhaseError) as ei:
        phase({"n_sites": n_sites, "reads": reads[:9]})
    assert ei.value.code == "INVALID_INPUT"

    bad = [dict(r) for r in reads]
    bad[0]["observations"] = bad[0]["observations"][:-1]
    with pytest.raises(PhaseError) as ei:
        phase({"n_sites": n_sites, "reads": bad})
    assert ei.value.code == "INVALID_INPUT"

    bad = [dict(r) for r in reads]
    bad[0]["mismatch_costs"] = [1, 0, 1][: len(bad[0]["observations"])]
    with pytest.raises(PhaseError) as ei:
        phase({"n_sites": n_sites, "reads": bad})
    assert ei.value.code == "INVALID_INPUT"


# --------------------------------------------------------------------------
# exhaustive cross-check
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_matches_brute_force_random(seed):
    n_sites, reads = random_instance(8, 10, seed)
    expected = brute_force(n_sites, reads)
    n_sites_p, parsed, families = parse_input({"n_sites": n_sites, "reads": reads})
    got, _flag = enumerate_solutions(n_sites_p, parsed, families)

    if not expected:
        assert got == []
        return

    assert len(got) <= 2
    for k, sol in enumerate(got):
        exp = expected[k]
        assert sol.haplotype == exp[2]
        assert sol.assignments == exp[3]
        assert sol.total_cost == exp[0]
        assert sol.max_mismatches == exp[1]

    # ambiguity flag must match whether more than one distinct optimum exists
    out = phase({"n_sites": n_sites, "reads": reads})
    assert out["unique"] is (len(expected) == 1)
    if len(expected) >= 2:
        assert len(out["solutions"]) == 2
    else:
        assert len(out["solutions"]) == 1


# --------------------------------------------------------------------------
# molecule families
# --------------------------------------------------------------------------


def family_instance(seed=0):
    """10 reads over 8 sites in five 2-member families.

    Four pairs stay on their native chromosome (zero allowance).  The last
    pair a4 (native haplotype) + b4 (native complement) is a cross-chromosome
    family: with loose allowances it is feasible on both sides but not free
    on either, forcing a positive-cost optimum that no "solve then repair"
    pass could reach.
    """
    n_sites, reads, hap = clean_instance()
    for r in reads:
        if r["id"] in ("a4", "b4"):
            width = r["end"] - r["start"]
            r["max_mismatches"] = width
    pairs = [("a0", "a1"), ("a2", "a3"), ("b0", "b1"), ("b2", "b3"), ("a4", "b4")]
    mol_by_id = {rid: f"mol{k}" for k, pair in enumerate(pairs) for rid in pair}
    for r in reads:
        if r["id"] in mol_by_id:
            r["molecule_id"] = mol_by_id[r["id"]]
    return n_sites, reads, hap, pairs


def test_family_members_always_share_a_group():
    n_sites, reads, hap, pairs = family_instance()
    out = phase({"n_sites": n_sites, "reads": reads})
    assert out["unique"] is True
    sol = out["solution"]
    groups_by_id = {row["id"]: row["group"] for row in sol["per_read"]}
    for a, b in pairs:
        assert groups_by_id[a] == groups_by_id[b], f"family {a},{b} was split"
    assert len(sol["groups"]["haplotype"]) >= 2
    assert len(sol["groups"]["complement"]) >= 2
    # four same-chromosome pairs cost zero; the cross-group pair must pay:
    # both on haplotype costs b4's 4 mismatches, both on complement costs
    # a4's 3 mismatches -> optimum 3
    assert sol["total_mismatch_cost"] == 3
    assert sol["max_per_read_mismatches"] == 3

    # family blocks carry group, members and aggregate evidence
    assert len(sol["families"]) == 5
    fam_by_mol = {f["molecule_id"]: f for f in sol["families"]}
    for k, (a, b) in enumerate(pairs):
        block = fam_by_mol[f"mol{k}"]
        assert block["members"] == [a, b]
        assert block["group"] == groups_by_id[a] == groups_by_id[b]
        assert block["mismatch_count"] == sum(
            row["mismatch_count"] for row in sol["per_read"] if row["id"] in (a, b)
        )
        assert block["mismatch_cost"] == sum(
            row["mismatch_cost"] for row in sol["per_read"] if row["id"] in (a, b)
        )
        assert block["mismatch_positions"] == sorted(set(block["mismatch_positions"]))


def test_family_changes_the_optimum_versus_independent_reads():
    """The cross-group family a4(group0 truth) + b4(group1 truth) is clean,
    so placing both on one chromosome costs mismatches whichever side wins.
    Independently the same reads cost zero; the family must force positive
    cost -- solving first without the constraint could never "repair" into
    this answer, since both old optima split the family.
    """
    n_sites, reads, _hap, _pairs = family_instance()
    free = [{k: v for k, v in r.items() if k != "molecule_id"} for r in reads]
    out_free = phase({"n_sites": n_sites, "reads": free})
    out_fam = phase({"n_sites": n_sites, "reads": reads})
    assert out_free["solution"]["total_mismatch_cost"] == 0
    free_opt = [tuple(s["assignments"]) for s in out_free["solutions"]]
    a4_idx = next(i for i, r in enumerate(free) if r["id"] == "a4")
    b4_idx = next(i for i, r in enumerate(free) if r["id"] == "b4")
    for assign in free_opt:
        assert assign[a4_idx] != assign[b4_idx]  # every old optimum splits them
    assert out_fam["solution"]["total_mismatch_cost"] > 0
    assert all(
        s["assignments"][a4_idx] == s["assignments"][b4_idx]
        for s in out_fam["solutions"]
    )


def random_family_instance(n_sites, m, seed):
    """Random covered instance with reads partitioned into 2/3-member families."""
    rng = random.Random(seed)
    hap = [rng.randrange(2) for _ in range(n_sites)]
    comp = [1 - b for b in hap]

    for _attempt in range(400):
        reads = []
        # partition m reads into units of size 1..3, guaranteeing at least
        # one family and no tail that would leave group bounds untested
        sizes: list[int] = []
        j = 0
        while j < m:
            remaining = m - j
            if remaining == 1:
                sizes[-1] += 1  # absorb a lone tail into the previous unit
                j += 1
                continue
            take = rng.choice([1, 2, 2, 3]) if remaining >= 3 else rng.choice([1, 2])
            sizes.append(take)
            j += take
        if not any(s >= 2 for s in sizes):
            continue  # retry: this draw produced no family
        mol_counter = 0
        for take in sizes:
            mol = f"fam{mol_counter}" if take >= 2 else None
            if take >= 2:
                mol_counter += 1
            for _ in range(take):
                length = rng.randint(2, min(5, n_sites))
                start = rng.randint(0, n_sites - length)
                src = hap if rng.randrange(2) == 0 else comp
                obs = list(src[start : start + length])
                flips = 0
                for k in range(length):
                    if rng.random() < 0.15:
                        obs[k] ^= 1
                        flips += 1
                costs = [rng.randint(1, 9) for _ in range(length)]
                allow = rng.choice([0, flips, flips, min(flips + 1, length), length])
                reads.append(
                    make_read(
                        f"r{len(reads)}",
                        start,
                        start + length,
                        obs,
                        costs,
                        allow,
                        molecule_id=mol,
                    )
                )
        covered = [False] * n_sites
        for r in reads:
            for s in range(r["start"], r["end"]):
                covered[s] = True
        if all(covered) and 10 <= len(reads) <= 36:
            return n_sites, reads
    raise AssertionError("could not generate a covering family instance")


@pytest.mark.parametrize("seed", range(30))
def test_matches_brute_force_random_families(seed):
    n_sites, reads = random_family_instance(8, 10, seed)
    expected = brute_force(n_sites, reads)
    n_sites_p, parsed, families = parse_input({"n_sites": n_sites, "reads": reads})
    assert families  # the generator really produced families
    got, flag = enumerate_solutions(n_sites_p, parsed, families)

    if not expected:
        assert got == []
        assert flag is False or True  # brute force only says "no feasible solution"
        return

    assert flag is True
    assert len(got) <= 2
    for k, sol in enumerate(got):
        exp = expected[k]
        assert sol.haplotype == exp[2]
        assert sol.assignments == exp[3]
        assert sol.total_cost == exp[0]
        assert sol.max_mismatches == exp[1]
        # families are never split in any reported solution
        for _mol, members in families:
            assert len({sol.assignments[i] for i in members}) == 1

    out = phase({"n_sites": n_sites, "reads": reads})
    assert out["unique"] is (len(expected) == 1)


def test_invalid_family_sizes():
    n_sites, reads, _hap, _pairs = family_instance()
    # singleton molecule label
    bad = [dict(r) for r in reads]
    bad[0]["molecule_id"] = "lonely"
    with pytest.raises(PhaseError) as ei:
        parse_input({"n_sites": n_sites, "reads": bad})
    assert ei.value.code == "INVALID_INPUT"
    assert "lonely" in ei.value.message

    # five-member family
    bad = [dict(r) for r in reads]
    for i in range(5):
        bad[i]["molecule_id"] = "big"
    with pytest.raises(PhaseError) as ei:
        parse_input({"n_sites": n_sites, "reads": bad})
    assert ei.value.code == "INVALID_INPUT"
    assert "big" in ei.value.message

    # non-string label
    bad = [dict(r) for r in reads]
    bad[0]["molecule_id"] = 123
    with pytest.raises(PhaseError) as ei:
        parse_input({"n_sites": n_sites, "reads": bad})
    assert ei.value.code == "INVALID_INPUT"


def test_family_no_solution_distinguishable():
    """A family whose members can only fit on opposite chromosomes under
    every canonical haplotype (zero allowance, complementary observations)
    must fail with FAMILY_NO_SOLUTION.  Without the labels the same reads
    phase perfectly with the two members on opposite groups -- proving the
    dead end is specifically the family constraint, and no half-finished
    grouping is leaked.
    """
    n_sites = 8
    pattern = [0, 0, 1, 1, 0, 1, 0, 1]
    other = [1, 1, 0, 0, 1, 0, 1, 0]
    reads = [
        make_read("f0a", 0, 8, pattern, allow=0, molecule_id="f0"),
        make_read("f0b", 0, 8, other, allow=0, molecule_id="f0"),
    ]
    # eight zero-allowance reads over [0,4): four per chromosome
    pad_specs = [
        ("p0", pattern[0:4]), ("p1", pattern[0:4]),
        ("p2", pattern[0:4]), ("p6", pattern[0:4]),
        ("p3", other[0:4]), ("p4", other[0:4]),
        ("p5", other[0:4]), ("p7", other[0:4]),
    ]
    for rid, obs in pad_specs:
        reads.append(make_read(rid, 0, 4, obs, allow=0))
    assert len(reads) == 10

    free = [{k: v for k, v in r.items() if k != "molecule_id"} for r in reads]
    out_free = phase({"n_sites": n_sites, "reads": free})
    assert out_free["solution"]["total_mismatch_cost"] == 0
    by_id = {row["id"]: row for row in out_free["solution"]["per_read"]}
    assert by_id["f0a"]["group"] != by_id["f0b"]["group"]

    with pytest.raises(PhaseError) as ei:
        phase({"n_sites": n_sites, "reads": reads})
    assert ei.value.code == "FAMILY_NO_SOLUTION"
    assert "f0" not in ei.value.message  # no partial assignment leaked


def test_four_member_family_with_overlapping_intervals():
    """A size-4 family (the upper legal bound) whose members overlap must be
    placed together; both groups still meet the two-read bounds."""
    n_sites = 8
    hap = [0, 1, 0, 1, 1, 0, 1, 0]
    comp = [1 - b for b in hap]
    # four haplotype-side fragments of one molecule, varied/overlapping spans
    fam_specs = [
        ("f0", 0, 3), ("f1", 0, 2), ("f2", 2, 5), ("f3", 4, 8),
    ]
    reads = [make_read(rid, s, e, hap[s:e], allow=0, molecule_id="big")
             for rid, s, e in fam_specs]
    # three clean complement reads and three clean haplotype reads
    comp_specs = [("b0", 0, 3), ("b1", 2, 6), ("b2", 5, 8)]
    hap_specs = [("a0", 1, 4), ("a1", 3, 7), ("a2", 6, 8)]
    for rid, s, e in comp_specs:
        reads.append(make_read(rid, s, e, comp[s:e], allow=0))
    for rid, s, e in hap_specs:
        reads.append(make_read(rid, s, e, hap[s:e], allow=0))
    assert len(reads) == 10
    out = phase({"n_sites": n_sites, "reads": reads})
    sol = out["solution"]
    assert sol["total_mismatch_cost"] == 0
    block, = sol["families"]
    assert block["molecule_id"] == "big"
    assert block["members"] == ["f0", "f1", "f2", "f3"]
    groups = {row["id"]: row["group"] for row in sol["per_read"]}
    assert len({groups[m] for m in block["members"]}) == 1
    assert len(sol["groups"]["haplotype"]) >= 2
    assert len(sol["groups"]["complement"]) >= 2


def test_unlabelled_request_response_unchanged_shape():
    """No molecule_id anywhere -> response keeps the legacy shape per read
    and exposes an empty families list; solution equals the independent one.
    """
    n_sites, reads, _ = clean_instance()
    out = phase({"n_sites": n_sites, "reads": reads})
    sol = out["solution"]
    assert sol["families"] == []
    for row in sol["per_read"]:
        assert "molecule_id" not in row
        assert set(row) == {"id", "group", "mismatch_count", "mismatch_cost", "mismatch_positions"}
