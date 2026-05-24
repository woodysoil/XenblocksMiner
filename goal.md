# Long-Running Codex Goal

Continuously execute `docs/HASH_OPTIMIZATION_GOAL.md`.

Optimize XenblocksMiner Hash API CUDA hashing throughput for the fixed mining workload:

- `t = 1`
- `p = 1` / `s = 1`
- `m = difficulty`

Preserve real `argon2id-xen` semantics. Do not approximate hashes, skip required work, weaken target matching, or report synthetic successes.

Run benchmark-driven optimization cycles without asking for approval unless a blocker listed in `docs/HASH_OPTIMIZATION_GOAL.md` is reached. Keep code, docs, tests, benchmark scenario names, and commit messages in English. Make small validated commits. Never commit local absolute paths, usernames, hostnames, secrets, wallet addresses, or private machine details.

Current target: improve throughput by at least 1000% over the measured baseline where feasible, or continue until benchmark and profiling evidence show a practical plateau.
