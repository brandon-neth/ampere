# Possible Optimizations for Ampere Loading and Attribution

Identified by line-level profiling (`line_profiler`) of
`Ensemble.from_trace_paths_parquet` and `Ensemble.attribute` on a 2-node
trace (`locgroup_dist_block_PARQUET_2nodes_64threads_sort_trial0`).

---

## Performance History

| Run | Loading | Attribution | Notes |
|-----|---------|-------------|-------|
| `profiling.out` (baseline) | 18.0 s | 44.8 s | Original code |
| `profiling2.out` | 14.8 s | 44.8 s | Pre-cast metrics upfront; `_apply_filter_to_dict` index computed once |
| `profilingA.out` | **6.4 s** | 46.4 s | **Plan A implemented** (lazy `Metric` construction) |

---

## Loading Phase Optimizations

---

## Background: what `Metric.__init__` does per instance

For each of the 86 unique metric names the loader calls `Metric(name, times,
values, config)`.  The constructor performs the following Arkouda server
round-trips (RTTs):

| Step | Operation | RTTs |
|------|-----------|------|
| (opt) cast | `ak.cast(times/values, float64)` | 0–2 |
| sort | `ak.argsort(times)` | 1 |
| gather | `times[perm]`, `values[perm] * scale` | 2 |
| scalars | `self.times[0]`, `self.times[-1]` | 2 |
| integration (INSTANTANEOUS only) | `dt`, `energy_steps`, `ak.zeros`, `ak.cumsum`, `ak.concatenate` | 5 |

~8–10 RTTs × 86 metrics ≈ **688–860 serial server round-trips** just for
construction.  In the observed workload only **one** metric
(`A2rocm_smi:::energy_count:device=6`) is ever passed to
`AttributionEngine.compute`; the other 85 are constructed and then discarded.

---

## Plan A — Lazy construction  ✅ *implemented (`profilingA.out`)*

**Core idea:** defer all Arkouda work until a field is first accessed.

`__init__` stores only the raw (unprocessed) input arrays and the config.
Two private methods gate the work:

- `_ensure_sorted()` — runs `argsort`, gathers `times` / `raw_values`,
  fetches `t_min` / `t_max`.  Guarded by `self._times is not None`.
- `_ensure_integrated()` — runs `dt`, `energy_steps`, `cumsum`.
  Guarded by `self._cum_values is not None`; calls `_ensure_sorted` first.

Public attributes that previously triggered server work become `@property`
accessors that call the appropriate guard method.

**Expected gain:** reduces construction RTTs from ~688 to ~10 (only the one
attributed metric pays the full cost).  The `Metric()` line's 52 %/7.8 s
should drop to near-zero for the 85 unused metrics.

**Risk:** any code that touches `metric.times`, `.t_min`, `.t_max`,
`.raw_values`, or `.cum_values` during the load phase (e.g. a diagnostic
print or an invariant check) triggers eager evaluation and cancels the
benefit.  All attribute access sites must be audited.

---

## Plan B — Compound sort via `ak.coargsort`

**Core idea:** replace the `GroupBy(m_names)` sort + 86 per-metric
`ak.argsort(times)` sorts with a single compound sort on `(m_names, times)`.

```python
perm = ak.coargsort([m_names, times_col])
sorted_names   = m_names[perm]
sorted_times   = times_col[perm]
sorted_values  = values_col[perm]
# GroupBy on already-sorted names is O(1) — just finds segment boundaries
g = ak.GroupBy(sorted_names)
segs = g.segments.to_ndarray().tolist()
```

Each metric's slice `sorted_times[start:end]` is then already time-ordered,
so `Metric.__init__` can skip its `argsort` + two indexed gathers when given
a `_presorted=True` flag.

**Expected gain:** saves 3 RTTs × 86 = **258 RTTs**.  Synergises with Plan A:
when the lazy sort does fire it can skip the `ak.argsort` entirely.

**Implementation notes:** `ak.coargsort` is available in Arkouda ≥ 2023.x.
The follow-up `GroupBy` on already-sorted data still incurs one sort
internally; an alternative is to derive segment boundaries from
`np.where(np.diff(sorted_names_np) != 0)` after a `to_ndarray()` call.

---

## Plan C — Batch integration via vectorized cumsum

**Core idea:** after the compound sort (Plan B), all metrics' data sits in one
contiguous Arkouda array ordered by `(metric, time)`.  The integration step
can be done in bulk instead of 5 RTTs per metric.

```python
# dt across the full array; zero cross-group gaps
dt_all = sorted_times[1:] - sorted_times[:-1]
boundary_mask = ak.zeros(dt_all.size, dtype=ak.bool)
boundary_mask[segs[1:] - 1] = True
dt_all = ak.where(boundary_mask, 0.0, dt_all)

# energy steps in bulk
energy_steps_all = sorted_values[:-1] * dt_all

# within-group prefix sum (if ak.GroupBy.scan is available)
_, cum_all = g.scan('sum', energy_steps_all)
# otherwise: ak.cumsum(energy_steps_all) minus group-start offsets
```

This computes all 86 metrics' cumulative values in **~5–8 RTTs total**
instead of 5 × 86 = 430.  The tricky parts are the cross-boundary dt
zeroing and prepending a zero per group in the output cumsum.

If `GroupBy.scan` is not available in the deployed Arkouda version the
within-group cumsum can be emulated: run `ak.cumsum` on the full array, then
subtract the cumulative value at the start of each group from all elements in
that group (one `GroupBy.broadcast` call).

---

## Plan D — Pull metrics data to local numpy, process offline

**Core idea:** the code comment already notes metrics are "small".  Pull the
full combined metrics array to the driver with two `to_ndarray()` calls, do
all per-metric operations (sort, dt, cumsum) in numpy at zero RTT cost, and
push results back to the Arkouda server lazily on first attribution use.

```python
times_np  = sorted_times.to_ndarray()   # 1 RTT
values_np = sorted_values.to_ndarray()  # 1 RTT

for i, m_name in enumerate(unique_metrics):
    s, e = segs[i], (segs[i+1] if i+1 < len(segs) else total)
    idx   = np.argsort(times_np[s:e])
    t_np  = times_np[s:e][idx]
    v_np  = values_np[s:e][idx] * config.scale_factor
    cum   = np.concatenate([[0.0], np.cumsum(v_np[:-1] * np.diff(t_np))])
    # store t_np, cum as numpy; push to Arkouda only when attributed
```

**Tradeoff:** `get_delta_vectorized` and `get_statistics_vectorized` call
`ak_interp1d` which requires server-side arrays, so a push-back
(`ak.array(t_np)`) is needed on first use — 2 RTTs per accessed metric.
With 1 metric attributed this is 2 RTTs total vs 688 currently.

Best combined with Plan A (lazy push-back): numpy processing at construction,
Arkouda transfer only when the metric is actually used in attribution.

---

## Priority and combinations (loading phase)

| Plan | Expected RTT reduction | Implementation effort | Status |
|------|----------------------|-----------------------|--------|
| **A — Lazy construction** | ~680 RTTs (pay only for attributed metric) | Low | ✅ Done |
| **B — Compound sort** | ~258 RTTs | Medium | |
| **C — Batch integration** | ~420 RTTs | High | |
| **D — Numpy pull + lazy push** | ~680 RTTs | Medium | |

Plans A and D address the same root cause from different angles; implement one
or the other, not both.  Plan B is a clean prerequisite for Plan C and a
useful complement to either A or D.

---

## Attribution Phase Optimizations

After loading improvements, attribution dominates at **~46 s**.  All costs are
in `AttributionEngine.compute` using the **Arkouda backend** (the pandas fast
path is bypassed because Arkouda is connected).

### Attribution bottlenecks (`profilingA.out`)

| Bottleneck | % of `compute` | Abs. time | Root cause |
|---|---|---|---|
| `_compute_coverage_ak` in exclusive depth loop (52 calls) | 42% | ~19 s | One Arkouda call per unique depth per rank |
| `metric.get_delta_vectorized` on full `breaks` array | 15% | ~6.6 s | Server-side interpolation over all unique timestamps |
| `_compute_coverage_ak` for shared active counts (4 calls) | 12% | ~5.5 s | One call per participating rank |
| `ak.searchsorted` × 2 for l_idx / r_idx (4 ranks) | 9% | ~4.3 s | Per-rank index computation |

Inside `_compute_coverage_ak` (21 s cumulative, 56 calls):

| Line | Operation | % of fn | Abs. |
|------|-----------|---------|------|
| `ak.searchsorted(breaks, starts)` | 26% | ~5.5 s | |
| `ak.searchsorted(breaks, ends)` | 24% | ~5.0 s | |
| `ak.GroupBy(idxs)` | 15% | ~3.2 s | |

`_exclusive_pandas` shows 0 s — it is **never called** because the exclusive
strategy uses the Arkouda depth-loop path instead.

---

### Plan E — Pull data to numpy; use the existing pandas fast path unconditionally  *(highest priority)*

**Core idea:** the pandas fast path for `shared` mode (lines 499–589 of
`__init__.py`) is already written, correct, and handles `exclusive` via
`_exclusive_pandas`.  It is only skipped because `get_backend() == 'pandas'`
is False when Arkouda is connected.  Making it run regardless of backend
would eliminate essentially every Arkouda RTT in the attribution computation.

**Algorithm change:** add a data-pull block before the `# ---- Arkouda
backend ----` section that converts rank and metric arrays to numpy, then
falls through to the existing pandas path logic:

```python
# --- Pull rank data to numpy for local processing (avoids hundreds of RTTs) ---
rank_s_np = [r.starts.to_ndarray()  for r in ranks]   # 1 RTT per rank
rank_e_np = [r.ends.to_ndarray()    for r in ranks]
rank_d_np = [r.depths.to_ndarray()  for r in ranks]

# metric data (already lazy — fires _ensure_sorted + _ensure_integrated once)
mt_np  = metric.times.to_ndarray()       # 1 RTT
cum_np = metric.cum_values.to_ndarray()  # 1 RTT

# ... then run the pandas path logic with these numpy arrays
# ... and push results back with ak.array(val) per rank
```

With 4 ranks and 1 metric: **~13 RTTs total** instead of the current ~700.
All computation — timeline construction, `np.unique`, `np.bincount`, interp,
`_exclusive_pandas` — runs locally on the driver at numpy speed.

**Why `_exclusive_pandas` is better than the depth loop:**  
The Arkouda exclusive path iterates over each unique call depth D and calls
`_compute_coverage_ak` once per depth (52 calls = 52 × ~7 RTTs).
`_exclusive_pandas` uses the identity `exclusive[k] = inclusive[k] -
sum(inclusive[children_of_k])`, computed in O(K log K) with a single
numpy sort — no depth iteration, no Arkouda round-trips.

**Expected gain:** 42 % + 15 % + 12 % + 9 % = **~78 % of attribution
time** is driven by Arkouda calls that the numpy path eliminates.
Attribution should drop from ~46 s to an estimated **5–12 s** (dominated by
the numpy computation speed and the cost of pulling rank data).

**Scalability:** ⚠️ **Does not scale past driver memory.**  Every rank's
`starts`, `ends`, `depths`, and `names` arrays must fit on the driver node.  A
trace with K function calls across R ranks requires O(K × R × 3 arrays) of
local memory (starts, ends, depths).  For the 2-node profiling trace this is
manageable; at 64+ nodes with deep call graphs it may not be.  Mitigation
options:

- **Size guard:** check `sum(r.starts.size for r in ranks)` before pulling;
  fall back to Plan F if it exceeds a configurable threshold (e.g. 50 M rows).
- **Rank-level batching:** pull and process one rank at a time so only one
  rank's arrays live in driver memory simultaneously.  Eliminates the
  multi-rank sharing of `breaks` / `active_counts`, but is still far fewer
  RTTs than the Arkouda depth loop.
- **Column projection:** only pull `starts`, `ends`, `depths` (3 arrays);
  `names` and `metadata` stay on the server and are accessed once at result
  construction time.

---

### Plan F — Vectorize the exclusive depth loop (Arkouda-only fallback)

**Use case:** when Plan E's numpy pull isn't feasible (data too large for
driver memory) but exclusive attribution is still required.

**Core idea:** replace the per-depth `_compute_coverage_ak` loop with a
single Arkouda pass that computes `max_depth_per_interval` without iterating
over depth levels.

The current loop effectively evaluates: for each interval, what is the
maximum depth of any call that covers it?  This can be reframed as a
"weighted coverage" problem — but Arkouda lacks a native max-by-interval
primitive.

A practical approach: compute coverage for **all depths simultaneously** using
a single "depth-weighted" diff array:

```python
# Weight each call's contribution by its depth
depth_weighted_diff = ak.zeros(breaks.size, dtype=ak.int64)
# For each call: add depth at start, subtract depth at end
# Then the running max gives max active depth per interval.
```

Because depth values aren't additive this requires sorting calls by
`(start_time, -depth)` so that deeper calls "dominate" — then a cumulative
max over the sorted diff gives the winning depth per interval in one pass.
This replaces O(D) `_compute_coverage_ak` calls with O(1) Arkouda operations.

**Implementation complexity:** high — requires careful index arithmetic to
correctly handle nested call stacks and gap intervals.  Plan E should be
preferred where memory allows.

**Scalability:** ✅ **Fully scalable.**  All arrays remain on the Arkouda
server; the driver only exchanges messages, not bulk data.  This is the
correct path for traces that exceed driver memory, and the intended use case
for the Arkouda backend.  The O(D) depth loop is the algorithmic inefficiency
— fixing it here (replacing with a single-pass max-depth computation) gives
the same asymptotic complexity as Plan E for large data without memory
constraints.

---

### Plan G — Cache the `breaks` array across attribution calls

**Observation:** `breaks = ak.unique(ak.concatenate([metric.times, ...rank
starts/ends...]))` rebuilds the global timeline from scratch for every
`attribute()` call.  The rank timestamps are identical regardless of which
metric is being attributed; only `metric.times` changes.

**Approach:** cache `breaks` (and the rank sort indices into it) on the
`Ensemble` or `Run` object after the first attribution call.  Subsequent
calls for different metrics only need to merge the new `metric.times` into
the cached rank timeline — a much smaller `ak.unique` over a pre-sorted base.

**Expected gain:** saves `ak.unique(merged)` + `ak.searchsorted` × 2 per
rank on every attribution beyond the first.  Roughly **1–2 s** per additional
metric.  Has no effect when only one metric is attributed (the current case),
but becomes significant in multi-metric workflows.

**Scalability:** ✅ **Fully scalable.**  The cache is an Arkouda server-side
array.  Merging new metric timestamps into a cached `breaks` is a small
Arkouda `ak.unique` call — memory and time scale only with the number of
new metric timestamps, not with the total rank data size.

---

### Plan H — Replace `_compute_coverage_ak` with a numpy equivalent

If Plan E is implemented, this is automatic (the numpy pull path never calls
`_compute_coverage_ak`).  Documented here as a standalone option for the
Arkouda-only case.

The Arkouda function performs:
`2 × searchsorted → filter → GroupBy + aggregate → scatter → cumsum`

The numpy equivalent is O(N) and eliminates every server round-trip:

```python
def _compute_coverage_np(starts_np, ends_np, breaks_np):
    l = np.maximum(np.searchsorted(breaks_np, starts_np, side='right') - 1, 0)
    r = np.searchsorted(breaks_np, ends_np, side='left')
    valid = r > l
    diff  = np.zeros(len(breaks_np), dtype=np.int64)
    np.add.at(diff, l[valid],  1)
    np.add.at(diff, r[valid], -1)
    return np.cumsum(diff)[:-1]
```

Used in place of `_compute_coverage_ak` it reduces the 56-call cost (21 s)
to pure numpy (~ms range).

**Scalability:** ⚠️ **Same constraint as Plan E** — requires `starts`,
`ends`, and `breaks` to be on the driver.  Not usable standalone for
large-scale data; only meaningful as a component of Plan E.

---

## Overall priority

| Plan | Phase | Expected gain | Scales past 1 node? | Effort | Status |
|------|-------|--------------|---------------------|--------|--------|
| Pre-cast + filter fix | Loading | −3.2 s (−18 %) | ✅ Yes | Low | ✅ Done |
| **A — Lazy Metric** | Loading | −8.4 s (−47 %) | ✅ Yes | Low | ✅ Done |
| **E — Numpy pull attribution** | Attribution | ~−36 s (est. −78 %) | ⚠️ Driver-memory bound | Medium | |
| **F — Vectorize depth loop** | Attribution | ~−19 s | ✅ Yes | High | |
| **G — Cache `breaks`** | Attribution | ~−2 s per extra metric | ✅ Yes | Low | |
| **B — Compound sort** | Loading | ~−1 s | ✅ Yes | Medium | |
| **C — Batch integration** | Loading | ~−0.5 s after A | ✅ Yes | High | |

### Scalability vs. performance trade-off

The Arkouda backend exists specifically to handle traces that are too large
for a single node.  The current attribution bottleneck (the depth loop)
arises from an **algorithmic choice** — iterating over depths — not from a
fundamental requirement to use Arkouda.

**Recommended strategy:**

1. **For small/medium traces (fits on driver):** implement Plan E with a
   size guard.  Gets ~78 % speedup immediately with modest effort.  The numpy
   computation is single-threaded on the driver but avoids all Arkouda
   latency, which dominates at this scale.

2. **For large traces (exceeds driver memory):** Plan F is the correct path.
   It keeps data distributed on Arkouda workers and fixes the depth-loop
   algorithm specifically.  Higher effort but preserves the distributed
   execution model that justifies using Arkouda in the first place.

3. **Adaptive dispatch:** implement both and switch at runtime based on total
   row count.  A threshold of ~10–50 M rows per rank is a reasonable starting
   point; tune empirically on the target system's driver memory.

```python
TOTAL_ROWS = sum(r.starts.size for r in ranks)
if TOTAL_ROWS < LOCAL_THRESHOLD:
    return _compute_local(metric, ranks, ...)   # Plan E
else:
    return _compute_arkouda(metric, ranks, ...) # Plan F (once implemented)
```

