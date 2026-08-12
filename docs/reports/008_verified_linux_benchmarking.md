# Report 008: Fixing a compiler-parity validity gap, and what changes when it's fixed

## 0. Why this report exists

Every prior report's FILO2 comparison has an unaddressed validity problem, first flagged in report 003 ("FILO2 was run via its prebuilt binary... same caveat as prior reports about unverified compiler-flag parity") but never actually fixed. This report fixes it, and the corrected numbers are materially different from what was previously published — reported here in full, not smoothed over.

**What was actually wrong**, confirmed by inspecting both build trees directly:

- Our solver's Windows build (`src/build`) is a Visual Studio 17 2022 project — built with **MSVC** (`cl.exe`).
- FILO2's Windows build (`baselines/filo2/build`) is configured via `CMAKE_CXX_COMPILER:STRING=C:/msys64/ucrt64/bin/g++.exe` — built with **g++**, and its `-O3 -march=native -flto=auto` flags (GCC syntax) were correctly applied.

So this was never "our solver was missing a flag" (report 006 already fixed that, for the MSVC build specifically). Every FILO2-vs-solver timing comparison published so far compared **two different compilers' codegen** on top of two different codebases — a materially worse problem, since compiler backend differences alone can easily be a 20-50% swing in wall-clock time on numerically heavy code, independent of anything either codebase does. A stale `baselines/filo2/build_wsl` directory also existed from an earlier attempt, but its `CMAKE_BUILD_TYPE` cache entry was empty (not `Release`), so its `-O3` flags — set only under `CMAKE_CXX_FLAGS_RELEASE` in FILO2's `CMakeLists.txt` — were silently never applied either. It could not be reused as-is.

**Also addressed here, per direct instruction**: BKS (best-known-solution) comparison alongside the FILO2 comparison, as a standing requirement for every benchmark run going forward. Investigation found this was mostly already built — `tools/bench.py`'s `gap_pct` column has been scoring against vendored BKS `.sol` files since report 005 — it just hasn't been surfaced as a headline metric in report prose, which has led with "gap to FILO2" only. This report reports both, and that becomes the standing format.

## 1. The fix: same compiler, same OS, same machine, verified

Both binaries were rebuilt from scratch under WSL (Ubuntu 24.04, g++ 13.3.0, cmake 3.28.3, same physical machine — 28 logical cores available), with the compiler invocation lines captured directly rather than trusting CMake's own report of what it configured (this is exactly the check that would have caught both the original MSVC problem and the empty-build-type `build_wsl` problem):

**Our solver** (`src/build_wsl`, `-DCMAKE_BUILD_TYPE=Release`):
```
/usr/bin/c++ -O3 -DNDEBUG -std=gnu++17 -O3 -march=native -c Stage2_ILS.cpp ...
```

**FILO2** (`baselines/filo2/build_wsl`, `-DCMAKE_BUILD_TYPE=Release -DENABLE_GUI=OFF`):
```
/usr/bin/c++ -Wall -Wextra -Wpedantic -Wuninitialized -O3 -DNDEBUG -O3 -march=native -flto=auto -c main.cpp ...
```

Both `-O3` and `-march=native` are confirmed present in both invocation lines, both compiled by the same `/usr/bin/c++` (g++ 13.3.0). One incidental portability bug was found and fixed along the way: `src/main.cpp` unconditionally `#include`d `<crtdbg.h>`, an MSVC-only header, which meant the codebase could not be built with any non-MSVC compiler at all until it was guarded behind `#if defined(_MSC_VER)`. This is a one-line, behavior-neutral fix (the debug-CRT code it guards was already conditional on `_MSC_VER`; only the `#include` itself was not) — it does not change anything about the Windows/MSVC build.

## 2. Method

Both binaries were re-run on the same two instances every prior report's headline numbers cite — Valle-D-Aosta (20,000 customers) and Lazio (999,999 customers) — using the settled configurations from report 006:

- **Valle-D-Aosta**: solver `-p 4 --stage2-ms 40000 --stage3-ms 1000 --stage5-ms 60000`, seeds 1-5; FILO2 with its default 100,000-iteration budget, same seeds.
- **Lazio**: solver `-p 16 --stage2-ms 200000 --stage3-ms 2000 --stage5-ms 20000`, seeds 1-3; FILO2 default, same seeds.

The solver side reused `tools/bench.py` unmodified (already verifier-integrated and BKS-scoring — no new tooling). FILO2 has no equivalent harness in this repo, so a small wrapper (`tools/run_filo2_wsl.sh`) was added: it runs the binary per seed, reads FILO2's own reported cost/elapsed-seconds from its `.out` file, and independently re-scores every `.sol` output against the same vendored BKS files via `tools/score_sol.py` (`baselines/filo2/results/i-bks/{Valle-D-Aosta,Lazio}.sol`) — the same "recompute, don't trust the reported number" discipline every other measurement in this repo uses. Every solver run was additionally checked feasible by `src/verifier.py`; every FILO2 run's `.sol` was independently recomputed feasible by `score_sol.py`. All raw output is kept under `results/bench/008_wsl_*/`.

## 3. Results

### Valle-D-Aosta (20,000 customers, 5 seeds)

| | Solver | FILO2 |
|---|---|---|
| Wall time | 103.4-103.6s (mean **103.5s**) | 87.7-194.0s (mean **140.6s**) |
| Cost | 22,010,089-22,118,255 (mean 22,065,436) | 21,720,934-21,761,544 (mean 21,738,306) |
| Gap to BKS (21,679,514) | 1.52-2.02% (mean **1.780%**) | 0.19-0.38% (mean **0.271%**) |
| Gap to FILO2 (mean cost) | **+1.505%** | — |

### Lazio (999,999 customers, 3 seeds)

| | Solver | FILO2 |
|---|---|---|
| Wall time | 261.8-263.1s (mean **262.4s**) | 319.1-365.4s (mean **340.1s**) |
| Cost | 3,181,280,794-3,182,339,142 (mean 3,181,798,006) | 3,156,655,265-3,158,699,091 (mean 3,157,466,737) |
| Gap to BKS (3,145,381,332) | 1.14-1.18% (mean **1.158%**) | 0.36-0.42% (mean **0.384%**) |
| Gap to FILO2 (mean cost) | **+0.771%** | — |

## 4. What this changes versus prior reports — stated plainly

**The cost-gap numbers roughly hold up.** Report 006 claimed +1.632% (VDA) / +0.741% (Lazio) against FILO2; the verified-Linux, multi-seed remeasurement gives **+1.505% / +0.771%** — close enough that the qualitative story ("solver is a bit more expensive than FILO2, closing over successive reports") survives the compiler-parity fix largely intact.

**The speed claim does not.** Every prior report's headline (2.3x-3.9x faster than FILO2) was built on the old, unverified FILO2 Windows numbers: 237s at Valle-D-Aosta and 641s at Lazio, both single-run point estimates from report 003. Under a fairly built, same-compiler, same-machine, multi-seed comparison, FILO2's own wall-clock time is **140.6s at Valle-D-Aosta and 340.1s at Lazio — roughly 40-47% faster than what was previously reported for it.** The corrected speedup is:

| Instance | Old headline (report 006) | Corrected (this report) |
|---|---|---|
| Valle-D-Aosta | 2.29x faster | **1.36x faster** |
| Lazio | 2.43x faster | **1.30x faster** |

The solver is still faster than FILO2 on both instances, but by a much narrower margin than previously claimed — **this is the single most important correction in this report, and it should be treated as superseding the speed claims in reports 003, 004, and 006**, not as a footnote alongside them.

**A second, related finding**: FILO2's wall time has real spread across seeds at Valle-D-Aosta (87.7s to 194.0s — more than 2x between the fastest and slowest seed), while the solver's is tight (103.4-103.6s). This is consistent with the two solvers' underlying designs: FILO2 runs to a **fixed iteration count** regardless of wall time, so a seed that leads it through more expensive move evaluations or a different acceptance trajectory genuinely does more or less total computational work; our solver runs to a **fixed wall-clock budget** by construction (report 004), so its timing is tight by design but its iteration count (and therefore how much search fits in the budget) is what varies instead. Neither is a bug — it's a direct consequence of iteration-budgeted vs. time-budgeted search — but it means a single FILO2 run, as every prior report used, was never a reliable point estimate, independent of the compiler question.

## 5. Root cause of the old inflated speedup claim

Best available explanation, consistent with the evidence: report 003's original FILO2 timings (237s / 641s) were **single Windows runs of an MSYS2-g++-built binary**, not verified-flag, multi-seed, same-OS measurements. The corrected numbers here differ from that baseline in three compounding ways at once — different OS (WSL/Linux vs. native Windows), multi-seed averaging instead of a single run, and (per §4) real seed-to-seed variance that a single run cannot separate from a true baseline. This report cannot cleanly attribute the ~40-47% gap between old and new FILO2 timings to any one of those three factors alone; what it can say is that the new numbers are the first that are verified, repeated, and measured under identical conditions to the solver's own numbers, and are the ones that should be trusted going forward.

## 6. Standing rules going forward

1. **WSL (`g++`, `-O3 -march=native`, `-DCMAKE_BUILD_TYPE=Release`, verified via the actual compile invocation line, not just CMake's config summary) is now the benchmarking baseline for any timing claim against FILO2.** The Windows/MSVC build remains valid for day-to-day development but must not be used to produce a published speed comparison again.
2. **Every future benchmark run reports gap-to-BKS alongside gap-to-FILO2**, not gap-to-FILO2 alone. `tools/bench.py` already computes this (`gap_pct` in every CSV); the change is a reporting-discipline one — headline both numbers, as done in §3 above.
3. **A timing claim against FILO2 needs ≥3 seeds on both sides**, not a single run of either — §4's variance finding shows a single FILO2 run is not a safe stand-in for its true average, exactly as report 005 already established for the solver's own numbers.

## 7. Caveats

- **Tier 1 (34 X-set instances x 5 seeds) was not re-run under WSL this round** — scope was deliberately limited to Valle-D-Aosta and Lazio, the two instances every report's headline numbers actually cite. The byte-identical-cost claims from reports 005/006 are not re-validated on Linux here; they remain Windows/MSVC-only findings.
- Sample sizes here (5 seeds at Valle-D-Aosta, 3 at Lazio) are the same as prior reports' Tier-2/Tier-3 conventions, but are still small enough that the FILO2 variance noted in §4 (87.7-194.0s) is a real, not fully characterized, spread — a larger seed count would tighten the confidence interval on FILO2's true mean time.
- No algorithm or search-behavior code changed in this report — only the `<crtdbg.h>` portability fix (§1), which is compiler-guard-only and does not affect the Windows/MSVC build's behavior.
- This report used g++ on Linux for both sides. It does not establish whether MSVC vs. g++ alone (holding OS constant) would reproduce the old inflated gap — that specific factorization was not isolated (see §5).
