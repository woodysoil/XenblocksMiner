# Long-Running Hash Optimization Goal

Continuously execute `docs/HASH_OPTIMIZATION_GOAL.md`.

Optimize XenblocksMiner Hash API hashing performance for the fixed mining workload:

- `t = 1`
- `s = 1` / `p = 1`
- `m = diff` / `difficulty`

Primary objective: minimize time per accepted hash attempt and maximize steady-state CUDA Hash API throughput while preserving exact `argon2id-xen` semantics.

First make the backend structure clean enough for isolated Hash API optimization. Then run benchmark-driven optimization cycles on the local GPU and keep the design portable for future RTX 3050-class and higher-end GPUs.

Operate autonomously: run non-destructive checks, builds, tests, benchmarks, and commits without asking for approval unless a blocker listed in `docs/HASH_OPTIMIZATION_GOAL.md` is reached.

Keep code, docs, tests, benchmark scenario names, and commit messages in English. Make small validated commits. Never commit local absolute paths, usernames, hostnames, secrets, wallet addresses, raw benchmark dumps, or private machine details.

Current target: improve throughput by at least 1000% over the measured baseline where feasible, or continue until benchmark and profiling evidence show a practical plateau.
