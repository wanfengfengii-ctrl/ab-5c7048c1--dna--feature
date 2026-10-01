"""Complementary-haplotype phasing core.

Jointly recovers a pair of complementary binary haplotypes from degraded
molecular reads.  For every canonical haplotype candidate (first site fixed
to ``0``, which removes the group-swap symmetry) each read is assigned to
exactly one of the two groups (haplotype / complement) so that:

1. the read's number of mismatching positions does not exceed its allowance,
2. each group holds at least two reads (counted by read, not family),
3. total mismatch cost is minimized, then
4. the largest per-read mismatch count is minimized, then
5. the assignment (and, on full ties, the haplotype) is lexicographically
   smallest -- giving a stable decision between uniqueness and ambiguity.

Reads may optionally carry ``molecule_id``.  Reads sharing a molecule id
form a *family* (2..4 members): fragments of one original molecule must be
assigned to a single group as a whole, so evidence of one family can never
be split across the two homologous chromosomes.  Unlabelled reads stay
independent.  The group-size bounds ("two reads per group") and every
mismatch limit remain per read.

Algorithm (n_sites <= 18, reads <= 36):

* all 2**(n_sites-1) canonical candidates are tabulated with vectorized
  numpy (per-read mismatch counts and costs against each side);
* minimum achievable cost per candidate:
    - without families, the original O(reads) greedy (the two-per-group
      bounds never cost more than flipping at most two strict reads per
      side; equal-cost neutral reads fill deficits for free);
    - with families, an exact vectorized DP over the group-0 *read count*,
      in which a family is one atomic, weighted (2..4 reads) edge;
* among candidates attaining the global minimum cost, a vectorized DP whose
  only state is the group-0 read count is run with a rising cap K on the
  per-read mismatch count.  The first K at which the minimum cost is
  reachable with 2..n-2 reads in group 0 is optimal; the first two distinct
  candidates feasible there are the reported tie set;
* a final exact (Python) DP for those at most two candidates yields the
  lexicographically smallest assignment and its mismatch evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ----- input validation -----------------------------------------------------


class PhaseError(ValueError):
    """Business-level validation / infeasibility error.

    ``code`` is a stable machine-readable reason returned by the API.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Read:
    id: str
    start: int  # inclusive, 0-based
    end: int  # exclusive
    obs: tuple[int, ...]
    costs: tuple[int, ...]
    max_mismatches: int
    molecule_id: str | None = None


@dataclass(frozen=True)
class Solution:
    haplotype: tuple[int, ...]
    assignments: tuple[int, ...]  # 0/1 per read, aligned to the input order
    total_cost: int
    max_mismatches: int
    mismatch_counts: tuple[int, ...]
    mismatch_costs: tuple[int, ...]
    mismatch_positions: tuple[tuple[int, ...], ...]  # global site indices


def _as_int_list(value, what: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise PhaseError("INVALID_INPUT", f"{what} must be a non-empty list")
    out: list[int] = []
    for i, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise PhaseError("INVALID_INPUT", f"{what}[{i}] must be an integer")
        out.append(item)
    return out


def parse_input(payload: object) -> tuple[int, list[Read], list[tuple[str, tuple[int, ...]]]]:
    if not isinstance(payload, dict):
        raise PhaseError("INVALID_INPUT", "request body must be a JSON object")

    n_sites = payload.get("n_sites")
    if isinstance(n_sites, bool) or not isinstance(n_sites, int):
        raise PhaseError("INVALID_INPUT", "n_sites must be an integer")
    if not 8 <= n_sites <= 18:
        raise PhaseError("INVALID_INPUT", "n_sites must be between 8 and 18")

    raw_reads = payload.get("reads")
    if not isinstance(raw_reads, list):
        raise PhaseError("INVALID_INPUT", "reads must be a list")
    if not 10 <= len(raw_reads) <= 36:
        raise PhaseError("INVALID_INPUT", "reads must contain between 10 and 36 items")

    reads: list[Read] = []
    seen_ids: set[str] = set()
    molecule_members: dict[str, list[int]] = {}
    molecule_order: list[str] = []
    for idx, item in enumerate(raw_reads):
        if not isinstance(item, dict):
            raise PhaseError("INVALID_INPUT", f"reads[{idx}] must be an object")
        rid = item.get("id", f"r{idx}")
        if not isinstance(rid, str) or not rid:
            raise PhaseError("INVALID_INPUT", f"reads[{idx}].id must be a non-empty string")
        if rid in seen_ids:
            raise PhaseError("INVALID_INPUT", f"duplicate read id: {rid}")
        seen_ids.add(rid)

        molecule = item.get("molecule_id")
        if molecule is not None:
            if not isinstance(molecule, str) or not molecule:
                raise PhaseError(
                    "INVALID_INPUT",
                    f"reads[{idx}].molecule_id must be a non-empty string when present",
                )
            if molecule not in molecule_members:
                molecule_members[molecule] = []
                molecule_order.append(molecule)
            molecule_members[molecule].append(idx)

        start = item.get("start")
        end = item.get("end")
        if isinstance(start, bool) or not isinstance(start, int):
            raise PhaseError("INVALID_INPUT", f"reads[{idx}].start must be an integer")
        if isinstance(end, bool) or not isinstance(end, int):
            raise PhaseError("INVALID_INPUT", f"reads[{idx}].end must be an integer")
        if not 0 <= start < end <= n_sites:
            raise PhaseError(
                "DISCONTINUOUS_INPUT",
                f"reads[{idx}] span [{start},{end}) is empty or outside [0,{n_sites})",
            )

        obs = _as_int_list(item.get("observations"), f"reads[{idx}].observations")
        costs = _as_int_list(item.get("mismatch_costs"), f"reads[{idx}].mismatch_costs")
        width = end - start
        if len(obs) != width or len(costs) != width:
            raise PhaseError(
                "INVALID_INPUT",
                f"reads[{idx}] observations/mismatch_costs length must equal end-start",
            )
        if any(b not in (0, 1) for b in obs):
            raise PhaseError("INVALID_INPUT", f"reads[{idx}] observations must be binary 0/1")
        if any(c <= 0 for c in costs):
            raise PhaseError("INVALID_INPUT", f"reads[{idx}] mismatch costs must be positive integers")

        allow = item.get("max_mismatches")
        if isinstance(allow, bool) or not isinstance(allow, int) or allow < 0:
            raise PhaseError(
                "INVALID_INPUT", f"reads[{idx}].max_mismatches must be a non-negative integer"
            )
        if allow > width:
            raise PhaseError("INVALID_INPUT", f"reads[{idx}].max_mismatches exceeds its span")

        reads.append(
            Read(
                id=rid,
                start=start,
                end=end,
                obs=tuple(obs),
                costs=tuple(costs),
                max_mismatches=allow,
                molecule_id=molecule,
            )
        )

    # A molecule family is several (2..4) fragments of one original molecule.
    # One labelled read carries no family constraint and five or more cannot
    # be the fragments distinguished here; both are explicit input errors.
    for mol in molecule_order:
        size = len(molecule_members[mol])
        if not 2 <= size <= 4:
            word = "read" if size == 1 else "reads"
            raise PhaseError(
                "INVALID_INPUT",
                f"molecule_id {mol!r} groups {size} {word}; a molecule family "
                "must contain between 2 and 4 reads",
            )
    families = [(mol, tuple(molecule_members[mol])) for mol in molecule_order]

    # The optimizer tabulates costs in signed 64-bit integers; reject inputs
    # whose theoretical maximum total cost could interfere with the DP's
    # infinity sentinel (inf ~ 2**61, so path sums must stay far below it).
    grand_total = sum(sum(r.costs) for r in reads)
    if grand_total > (1 << 58):
        raise PhaseError(
            "INVALID_INPUT",
            "sum of mismatch costs is too large to score exactly; costs must be "
            "small positive integers (aggregate cost must fit in 59 bits)",
        )

    # Every read is a contiguous interval by construction.  Across reads the
    # union must tile the full locus: an uncovered site cannot be phased and
    # is a business-level "discontinuous input" rejection.
    covered = [False] * n_sites
    for r in reads:
        for s in range(r.start, r.end):
            covered[s] = True
    if not all(covered):
        gap = next(i for i, c in enumerate(covered) if not c)
        raise PhaseError(
            "DISCONTINUOUS_INPUT",
            f"site {gap} is not covered by any read; reads do not form a continuous tiling",
        )

    return n_sites, reads, families


# ----- candidate tables -----------------------------------------------------


def candidate_tables(n_sites: int, reads: list[Read]):
    """Per-candidate mismatch / cost / feasibility tables.

    Arrays shaped ``(C, m)`` with ``C = 2**(n_sites-1)``; candidates are in
    lexicographic order of the free sites (candidate 0 = all-zero
    haplotype; site 0 is fixed to 0):

    * ``mm0``   uint8 mismatches vs the canonical haplotype (group 0)
    * ``mm1``   uint8 mismatches vs the complement (group 1)
    * ``cost0`` int64 mismatch cost vs the canonical haplotype
    * ``cost1`` int64 mismatch cost vs the complement
    * ``feas0`` bool  group-0 assignment respects the mismatch allowance
    * ``feas1`` bool  group-1 assignment respects the mismatch allowance
    """
    m = len(reads)
    n_bits = n_sites - 1
    c_count = 1 << n_bits

    mm0 = np.zeros((c_count, m), dtype=np.uint8)
    cost0 = np.zeros((c_count, m), dtype=np.int64)

    idx = np.arange(c_count, dtype=np.int64)
    site_cols: list[np.ndarray] = [np.zeros(c_count, dtype=np.uint8)]
    for s in range(1, n_sites):
        site_cols.append(((idx >> (n_bits - s)) & 1).astype(np.uint8))

    for j, r in enumerate(reads):
        span = r.end - r.start
        obs = np.asarray(r.obs, dtype=np.uint8)
        costs = np.asarray(r.costs, dtype=np.int64)
        mis = np.empty((span, c_count), dtype=np.uint8)
        for k, site in enumerate(range(r.start, r.end)):
            np.bitwise_xor(obs[k], site_cols[site], out=mis[k])
        mm0[:, j] = mis.sum(axis=0)
        cost0[:, j] = costs @ mis

    allow = np.asarray([r.max_mismatches for r in reads], dtype=np.uint8)
    spans = np.asarray([r.end - r.start for r in reads], dtype=np.uint8)
    span_cost = np.asarray([sum(r.costs) for r in reads], dtype=np.int64)
    mm1 = spans[None, :].astype(np.uint8) - mm0
    cost1 = span_cost[None, :] - cost0
    feas0 = mm0 <= allow[None, :]
    feas1 = mm1 <= allow[None, :]
    return mm0, mm1, cost0, cost1, feas0, feas1


def build_units(
    families: list[tuple[str, tuple[int, ...]]],
    n_reads: int,
    mm0: np.ndarray,
    mm1: np.ndarray,
    cost0: np.ndarray,
    cost1: np.ndarray,
    feas0: np.ndarray,
    feas1: np.ndarray,
):
    """Aggregate per-read tables into atomic assignment *units*.

    A unit is one whole molecule family (2..4 members) or one independent
    read.  A unit is feasible on a side only when *every* member is feasible
    there; its mismatch count on a side is the members' maximum (used for
    the rising cap) and its cost is the member total.  Family units come
    first (in first-appearance order), then the singleton reads in input
    order.  Returns the member lists and arrays shaped ``(C, U)`` plus the
    per-unit read weights.
    """
    c_count = cost0.shape[0]
    family_index_lists = [list(members) for _mol, members in families]
    grouped = {i for members in family_index_lists for i in members}
    single_indices = [[i] for i in range(n_reads) if i not in grouped]
    member_lists = family_index_lists + single_indices

    u = len(member_lists)
    uc0 = np.empty((c_count, u), dtype=np.int64)
    uc1 = np.empty((c_count, u), dtype=np.int64)
    um0 = np.empty((c_count, u), dtype=np.uint8)
    um1 = np.empty((c_count, u), dtype=np.uint8)
    uf0 = np.empty((c_count, u), dtype=bool)
    uf1 = np.empty((c_count, u), dtype=bool)
    weights = np.empty(u, dtype=np.int64)

    for ui, members in enumerate(member_lists):
        cols = np.asarray(members, dtype=np.int64)
        uc0[:, ui] = cost0[:, cols].sum(axis=1)
        uc1[:, ui] = cost1[:, cols].sum(axis=1)
        um0[:, ui] = mm0[:, cols].max(axis=1)
        um1[:, ui] = mm1[:, cols].max(axis=1)
        uf0[:, ui] = feas0[:, cols].all(axis=1)
        uf1[:, ui] = feas1[:, cols].all(axis=1)
        weights[ui] = len(members)

    return member_lists, uc0, uc1, um0, um1, uf0, uf1, weights


# ----- minimum-cost analysis: greedy when every unit is a single read -------


def min_cost_for_rows(
    m0: np.ndarray,
    m1: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    f0: np.ndarray,
    f1: np.ndarray,
    offset: int,
) -> dict[int, int]:
    """Exact minimum achievable cost for each row of a candidate chunk.

    With every read sent to its cheapest feasible side, each group already
    holds some count of strict-preference and forced reads.  Satisfying the
    "at least two reads per group" bounds can require flipping at most two
    strict reads per side to their dearer side (equal-cost *neutral* reads
    fill deficits free of charge), so only the two cheapest flip penalties
    per side ever matter.
    """
    out: dict[int, int] = {}
    n = m0.shape[1]
    for row in range(m0.shape[0]):
        forced_count0 = 0
        forced_count1 = 0
        pen1to0: list[int] = []  # strict group-1 pref: penalty to flip to 0
        pen0to1: list[int] = []  # strict group-0 pref: penalty to flip to 1
        neutral = 0
        base = 0
        dead = False
        for i in range(n):
            a = bool(f0[row, i])
            b = bool(f1[row, i])
            if not a and not b:
                dead = True
                break
            cc0 = int(c0[row, i])
            cc1 = int(c1[row, i])
            if a and not b:
                forced_count0 += 1
                base += cc0
            elif b and not a:
                forced_count1 += 1
                base += cc1
            elif cc0 < cc1:
                forced_count0 += 1
                pen0to1.append(cc1 - cc0)
                base += cc0
            elif cc1 < cc0:
                forced_count1 += 1
                pen1to0.append(cc0 - cc1)
                base += cc1
            else:
                neutral += 1
                base += cc0
        if dead:
            continue

        a0 = forced_count0
        a1 = forced_count1
        need_x = max(0, 2 - a0)  # extra group-0 reads potentially required
        need_y = max(0, 2 - a1)  # extra group-1 reads potentially required
        if need_x > len(pen1to0) + neutral or need_y > len(pen0to1) + neutral:
            continue

        pen1to0.sort()
        pen0to1.sort()
        sx = [0]
        for p in pen1to0[:need_x]:
            sx.append(sx[-1] + p)
        sy = [0]
        for p in pen0to1[:need_y]:
            sy.append(sy[-1] + p)

        best_extra: int | None = None
        max_flip0 = min(need_x, len(pen1to0))  # strict group-1 reads -> 0
        max_flip1 = min(need_y, len(pen0to1))  # strict group-0 reads -> 1
        for x in range(max_flip0 + 1):
            for y in range(max_flip1 + 1):
                # group-0 count = a0 + x - y + t neutrals placed in group 0
                lo_t = max(0, 2 - a0 - x + y)
                hi_t = min(neutral, n - 2 - a0 - x + y)
                if lo_t <= hi_t:
                    extra = sx[x] + sy[y]
                    if best_extra is None or extra < best_extra:
                        best_extra = extra
        if best_extra is not None:
            out[offset + row] = base + best_extra
    return out


# ----- minimum-cost analysis: exact weighted unit DP with families ---------


def _unit_weighted_dp(
    c0: np.ndarray,
    c1: np.ndarray,
    a0: np.ndarray,
    a1: np.ndarray,
    weights: np.ndarray,
    n_reads: int,
    cap_cost: int | None = None,
) -> np.ndarray:
    """Minimum cost per group-0 read count, vectorized over candidate rows.

    Each unit contributes its read weight (1..4) to the chosen group's
    count.  ``a0``/``a1`` gate usable edges (allowance plus, optionally, the
    rising mismatch cap).  Returns a ``(rows, n_reads+1)`` array; entries
    that are unreachable or exceed ``cap_cost`` hold the infinity sentinel.
    """
    t = c0.shape[0]
    inf = np.int64(np.iinfo(np.int64).max // 4)
    work = np.full((t, n_reads + 1), inf, dtype=np.int64)
    work[:, 0] = 0
    for i, w in enumerate(int(x) for x in weights):
        cand0 = np.full((t, n_reads + 1), inf, dtype=np.int64)
        cand1 = np.full((t, n_reads + 1), inf, dtype=np.int64)
        rows0 = np.where(a0[:, i])[0]
        rows1 = np.where(a1[:, i])[0]
        if rows0.size:
            # group-0 edge raises the group-0 read count by the unit weight
            cand0[rows0[:, None], np.arange(w, n_reads + 1)[None, :]] = (
                work[rows0, : n_reads + 1 - w] + c0[rows0, i : i + 1]
            )
        if rows1.size:
            cand1[rows1, :] = work[rows1, :] + c1[rows1, i : i + 1]
        work = np.minimum(cand0, cand1)
        # One step can at most add the aggregate cost (<= 2**58) to inf,
        # which stays inside int64; clip each step so sums never accumulate
        # far enough to wrap negative.
        work[work > inf] = inf
        if cap_cost is not None:
            # prune paths already above the target; positive costs mean they
            # can never return to it
            work[work > cap_cost] = inf
    return work


def min_cost_units_for_rows(
    c0: np.ndarray,
    c1: np.ndarray,
    f0: np.ndarray,
    f1: np.ndarray,
    weights: np.ndarray,
    n_reads: int,
    offset: int,
    chunk: int = 512,
) -> dict[int, int]:
    """Exact minimum achievable cost per candidate row with atomic families."""
    inf = np.int64(np.iinfo(np.int64).max // 4)
    out: dict[int, int] = {}
    total_rows = f0.shape[0]
    for lo in range(0, total_rows, chunk):
        hi = min(total_rows, lo + chunk)
        work = _unit_weighted_dp(
            c0[lo:hi], c1[lo:hi], f0[lo:hi], f1[lo:hi], weights, n_reads
        )
        best = work[:, 2 : n_reads - 1].min(axis=1)
        for k in np.where(best < inf)[0]:
            out[offset + lo + int(k)] = int(best[k])
    return out


# ----- secondary objective: rising per-read mismatch cap -------------------


def feasible_under_cap(
    mm0: np.ndarray,
    mm1: np.ndarray,
    cost0: np.ndarray,
    cost1: np.ndarray,
    feas0: np.ndarray,
    feas1: np.ndarray,
    candidate_ids: np.ndarray,
    target_cost: int,
    cap: int,
    chunk: int = 4096,
) -> list[int]:
    """Read-level: candidates reaching ``target_cost`` with max mismatch <= cap.

    Per candidate the DP keeps, for each possible number of reads assigned
    to group 0, the minimum total mismatch cost achievable; an edge is only
    usable when its side is feasible (allowance) and its mismatch count does
    not exceed ``cap``.  A candidate succeeds iff some group-0 count in
    ``[2, n-2]`` attains ``target_cost``.
    """
    n = mm0.shape[1]
    winners: list[int] = []

    for start in range(0, len(candidate_ids), chunk):
        ids = candidate_ids[start : start + chunk]
        a0 = feas0[ids] & (mm0[ids] <= cap)
        a1 = feas1[ids] & (mm1[ids] <= cap)
        work = _unit_weighted_dp(
            cost0[ids],
            cost1[ids],
            a0,
            a1,
            np.ones(n, dtype=np.int64),
            n,
            cap_cost=target_cost,
        )
        ok = (work[:, 2 : n - 1] == target_cost).any(axis=1)
        winners.extend(int(candidate_ids[start + k]) for k in np.where(ok)[0])
    return winners


def feasible_units_under_cap(
    uc0: np.ndarray,
    uc1: np.ndarray,
    um0: np.ndarray,
    um1: np.ndarray,
    uf0: np.ndarray,
    uf1: np.ndarray,
    weights: np.ndarray,
    n_reads: int,
    candidate_ids: np.ndarray,
    target_cost: int,
    cap: int,
    chunk: int = 4096,
) -> list[int]:
    """Unit-level variant: families move atomically; counts are read weights."""
    winners: list[int] = []

    for start in range(0, len(candidate_ids), chunk):
        ids = candidate_ids[start : start + chunk]
        a0 = uf0[ids] & (um0[ids] <= cap)
        a1 = uf1[ids] & (um1[ids] <= cap)
        work = _unit_weighted_dp(
            uc0[ids],
            uc1[ids],
            a0,
            a1,
            weights,
            n_reads,
            cap_cost=target_cost,
        )
        ok = (work[:, 2 : n_reads - 1] == target_cost).any(axis=1)
        winners.extend(int(candidate_ids[start + k]) for k in np.where(ok)[0])
    return winners


# ----- exact assignment DP for reported candidates -------------------------


def solve_assignments(
    m0: list[int],
    m1: list[int],
    c0: list[int],
    c1: list[int],
    f0: list[bool],
    f1: list[bool],
    weights: list[int],
    masks: list[int],
    n_reads: int,
    target_cost: int,
    cap: int,
    limit: int = 2,
) -> list[tuple[int, int]]:
    """Up to ``limit`` lexicographically smallest assignments at optimum.

    Every returned pair ``(actual max mismatches, assignment bits)`` reaches
    ``target_cost`` with every read mismatching at most ``cap`` and both
    groups holding at least two reads.  Families move atomically: a unit-1
    edge sets all of the family's member bits.  When ``cap`` is the optimal
    secondary-objective value, each returned assignment has max mismatches
    exactly ``cap`` (an assignment with a smaller maximum would have been
    feasible at the previous cap).  Reads occupy bits from the most
    significant end, so integer order is the lexicographic order of group
    labels.
    """
    n = len(m0)
    # state: (group0 read count, max-mm group0, max-mm group1) -> (cost, bits)
    dp: dict[tuple[int, int, int], tuple[int, int]] = {(0, 0, 0): (0, 0)}

    for i in range(n):
        w = weights[i]
        mask = masks[i]
        can0 = f0[i] and m0[i] <= cap
        can1 = f1[i] and m1[i] <= cap
        nxt: dict[tuple[int, int, int], tuple[int, int]] = {}
        for (cnt, mx0, mx1), (tot, assign) in dp.items():
            if can0:
                t = tot + c0[i]
                if t <= target_cost:
                    key = (cnt + w, max(mx0, m0[i]), mx1)
                    val = (t, assign)
                    old = nxt.get(key)
                    if old is None or val < old:
                        nxt[key] = val
            if can1:
                t = tot + c1[i]
                if t <= target_cost:
                    key = (cnt, mx0, max(mx1, m1[i]))
                    val = (t, assign | mask)
                    old = nxt.get(key)
                    if old is None or val < old:
                        nxt[key] = val
        dp = nxt

    finals: list[tuple[int, int]] = []  # (bits, max mm)
    for (cnt, mx0, mx1), (tot, assign) in dp.items():
        if tot == target_cost and 2 <= cnt <= n_reads - 2:
            finals.append((assign, max(mx0, mx1)))
    finals.sort()
    return [(mx, bits) for bits, mx in finals[:limit]]


# ----- top level ------------------------------------------------------------


def _hap_from_bits(hap_bits: int, n_sites: int) -> tuple[int, ...]:
    return (0,) + tuple((hap_bits >> (n_sites - 2 - s)) & 1 for s in range(n_sites - 1))


def _bits_to_tuple(bits: int, n: int) -> tuple[int, ...]:
    return tuple((bits >> (n - 1 - i)) & 1 for i in range(n))


def enumerate_solutions(
    n_sites: int,
    reads: list[Read],
    families: list[tuple[str, tuple[int, ...]]],
) -> tuple[list[Solution], bool]:
    """Optimal solutions plus a family-level feasibility flag.

    The flag is False iff at least one family exists yet no canonical
    haplotype lets every family be placed wholly on one side within its
    members' mismatch allowances -- a distinguishable failure from the
    ordinary count / allowance dead end, and it is decided before any
    assignment is produced.
    """
    n = len(reads)
    mm0, mm1, cost0, cost1, feas0, feas1 = candidate_tables(n_sites, reads)
    c_count = 1 << (n_sites - 1)
    have_families = bool(families)

    if have_families:
        (
            member_lists,
            uc0,
            uc1,
            um0,
            um1,
            uf0,
            uf1,
            weights,
        ) = build_units(families, n, mm0, mm1, cost0, cost1, feas0, feas1)

        fam_cols = np.arange(len(families))
        family_feasible_anywhere = bool(
            ((uf0[:, fam_cols] | uf1[:, fam_cols])).all(axis=1).any()
        )

        min_costs: dict[int, int] = {}
        global_best: int | None = None
        for base in range(0, c_count, 4096):
            stop = min(c_count, base + 4096)
            part = min_cost_units_for_rows(
                uc0[base:stop],
                uc1[base:stop],
                uf0[base:stop],
                uf1[base:stop],
                weights,
                n,
                base,
            )
            if part:
                min_costs.update(part)
                chunk_best = min(part.values())
                if global_best is None or chunk_best < global_best:
                    global_best = chunk_best
    else:
        member_lists = [[i] for i in range(n)]
        weights = np.ones(n, dtype=np.int64)
        uc0 = uc1 = um0 = um1 = uf0 = uf1 = None  # type: ignore[assignment]
        family_feasible_anywhere = True

        min_costs = {}
        global_best = None
        for base in range(0, c_count, 4096):
            stop = min(c_count, base + 4096)
            part = min_cost_for_rows(
                mm0[base:stop],
                mm1[base:stop],
                cost0[base:stop],
                cost1[base:stop],
                feas0[base:stop],
                feas1[base:stop],
                base,
            )
            if part:
                min_costs.update(part)
                chunk_best = min(part.values())
                if global_best is None or chunk_best < global_best:
                    global_best = chunk_best

    if global_best is None:
        return [], family_feasible_anywhere

    tied = np.asarray(
        sorted(hb for hb, mc in min_costs.items() if mc == global_best), dtype=np.int64
    )

    # Pass 2: rising cap on the per-read mismatch count.  The first cap at
    # which any tied candidate reaches the minimum cost with both groups
    # populated is the optimal secondary objective; candidates arrive in
    # lexicographic order, so the first two are the canonical tie set.
    max_span = max(r.end - r.start for r in reads)
    chosen: list[int] = []
    chosen_cap = 0
    for cap in range(0, max_span + 1):
        if have_families:
            found = feasible_units_under_cap(
                uc0,
                uc1,
                um0,
                um1,
                uf0,
                uf1,
                weights,
                n,
                tied,
                global_best,
                cap=cap,
            )
        else:
            found = feasible_under_cap(
                mm0,
                mm1,
                cost0,
                cost1,
                feas0,
                feas1,
                tied,
                global_best,
                cap=cap,
            )
        if found:
            chosen = sorted(found)[:2]
            chosen_cap = cap
            break
    if not chosen:  # pragma: no cover - pass 1 already guarantees feasibility
        return [], family_feasible_anywhere

    unit_weights = [int(weights[i]) for i in range(len(member_lists))]
    unit_masks = [
        sum(1 << (n - 1 - read_idx) for read_idx in members) for members in member_lists
    ]

    # Gather the first two distinct *solutions*.  Two assignments under the
    # same haplotype are distinct solutions (the swap-equivalent copy is the
    # complement-labelled pair, which is already quotiented out); candidates
    # and assignments are both visited in lexicographic order.
    picked: list[tuple[int, int, int]] = []  # (hap bits, max mm, assignment bits)
    for hb in chosen:
        if len(picked) >= 2:
            break
        if have_families:
            exact_list = solve_assignments(
                um0[hb].tolist(),
                um1[hb].tolist(),
                uc0[hb].tolist(),
                uc1[hb].tolist(),
                uf0[hb].tolist(),
                uf1[hb].tolist(),
                unit_weights,
                unit_masks,
                n,
                global_best,
                chosen_cap,
                limit=2 - len(picked),
            )
        else:
            exact_list = solve_assignments(
                mm0[hb].tolist(),
                mm1[hb].tolist(),
                cost0[hb].tolist(),
                cost1[hb].tolist(),
                feas0[hb].tolist(),
                feas1[hb].tolist(),
                unit_weights,
                unit_masks,
                n,
                global_best,
                chosen_cap,
                limit=2 - len(picked),
            )
        for maxmm, assign_bits in exact_list:
            picked.append((hb, maxmm, assign_bits))

    solutions: list[Solution] = []
    for hb, maxmm, assign_bits in picked:
        hap = _hap_from_bits(hb, n_sites)
        assignments = _bits_to_tuple(assign_bits, n)
        mm_counts: list[int] = []
        mm_costs: list[int] = []
        mm_pos: list[tuple[int, ...]] = []
        for r, g in zip(reads, assignments):
            count = 0
            spent = 0
            positions: list[int] = []
            for k, site in enumerate(range(r.start, r.end)):
                mismatch = (r.obs[k] != hap[site]) if g == 0 else (r.obs[k] == hap[site])
                if mismatch:
                    count += 1
                    spent += r.costs[k]
                    positions.append(site)
            mm_counts.append(count)
            mm_costs.append(spent)
            mm_pos.append(tuple(positions))
        solutions.append(
            Solution(
                haplotype=hap,
                assignments=assignments,
                total_cost=global_best,
                max_mismatches=maxmm,
                mismatch_counts=tuple(mm_counts),
                mismatch_costs=tuple(mm_costs),
                mismatch_positions=tuple(mm_pos),
            )
        )
    return solutions, family_feasible_anywhere


def phase(payload: object) -> dict:
    """Validate, solve and build the API response payload."""
    n_sites, reads, families = parse_input(payload)
    solutions, family_feasible_anywhere = enumerate_solutions(n_sites, reads, families)
    if not solutions:
        if not family_feasible_anywhere:
            # No partial assignment is produced on this path; the message
            # names only the family constraint, never a half-built grouping.
            raise PhaseError(
                "FAMILY_NO_SOLUTION",
                "no canonical haplotype lets every molecule family be assigned "
                "to one group as a whole while each member stays within its "
                "mismatch allowance",
            )
        raise PhaseError(
            "NO_SOLUTION",
            "no complementary haplotype pair admits an assignment with at "
            "least two reads in each group within the mismatch allowances",
        )

    def serialize(sol: Solution) -> dict:
        groups: list[list[str]] = [[], []]
        per_read = []
        for r, g, mc, mco, pos in zip(
            reads,
            sol.assignments,
            sol.mismatch_counts,
            sol.mismatch_costs,
            sol.mismatch_positions,
        ):
            groups[g].append(r.id)
            per_read.append(
                {
                    "id": r.id,
                    "group": g,
                    "mismatch_count": mc,
                    "mismatch_cost": mco,
                    "mismatch_positions": list(pos),
                }
            )

        family_blocks: list[dict] = []
        for mol, members in families:
            g = sol.assignments[members[0]]
            total_count = sum(sol.mismatch_counts[i] for i in members)
            total_cost = sum(sol.mismatch_costs[i] for i in members)
            positions = sorted({p for i in members for p in sol.mismatch_positions[i]})
            family_blocks.append(
                {
                    "molecule_id": mol,
                    "group": g,
                    "members": [reads[i].id for i in members],
                    "mismatch_count": total_count,
                    "mismatch_cost": total_cost,
                    "mismatch_positions": positions,
                }
            )

        return {
            "haplotype": list(sol.haplotype),
            "complement": [1 - b for b in sol.haplotype],
            "groups": {"haplotype": groups[0], "complement": groups[1]},
            "assignments": list(sol.assignments),
            "per_read": per_read,
            "families": family_blocks,
            "total_mismatch_cost": sol.total_cost,
            "max_per_read_mismatches": sol.max_mismatches,
            "mismatch_positions": sorted({p for tup in sol.mismatch_positions for p in tup}),
        }

    response: dict = {
        "n_sites": n_sites,
        "unique": len(solutions) == 1,
        "solutions": [serialize(s) for s in solutions],
    }
    response["solution"] = response["solutions"][0]
    if len(solutions) > 1:
        response["note"] = (
            "two distinct solutions tie on (total_mismatch_cost, "
            "max_per_read_mismatches); they may differ in haplotype, in the "
            "per-read assignment, or in a molecule family's group. The first "
            "two canonical solutions are returned"
        )
    return response
