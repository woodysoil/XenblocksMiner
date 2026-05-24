# Long-Running Hash Optimization Goal

Execute `docs/HASH_OPTIMIZATION_GOAL.md` as the persistent operating brief.

Optimize XenblocksMiner Hash API CUDA hashing throughput for the fixed mining workload:

- `t = 1`
- `s = 1` / `p = 1`
- `m = diff` / `difficulty`

Primary objective: minimize time per valid hash attempt and maximize warm steady-state Hash API throughput while preserving exact `argon2id-xen` semantics.

Work autonomously in benchmark-driven cycles:

1. Inspect git state and recent benchmark commits.
2. Establish or reuse a current baseline from ignored local benchmark reports.
3. Pick one measurable bottleneck.
4. Make the smallest useful code, harness, test, or documentation change.
5. Validate correctness before accepting any speed result.
6. Benchmark before and after with identical settings.
7. Commit small validated slices with English commit messages.
8. Repeat until the target is reached or a documented plateau is proven.

Current target: improve throughput by at least 1000% over the measured baseline where feasible, or continue until repeated benchmark and profiling evidence show the implementation is near the practical hardware limit.

Do not ask for approval for routine non-destructive checks, local builds, tests, benchmarks, scoped edits, or commits. Stop only for blockers listed in `docs/HASH_OPTIMIZATION_GOAL.md`.

Keep all code, comments, docs, tests, benchmark scenario names, API names, and commit messages in English. Never commit local absolute paths, usernames, hostnames, secrets, wallet addresses, raw benchmark dumps, or private machine details.
