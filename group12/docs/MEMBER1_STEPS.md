# Member 1: status and what is left

## Done and pushed (branch `member1`, github.com/parikhwal291/OptiPIM-grp12)

- Build on WSL2, Gurobi 13.0.3.
- 10 OptiPIM mappings: C1, C2, C3, M1, M2 on HBM-PIM and SIMDRAM, 32 banks
  (`group12/mappings/<target>/<op>/optipim.trace` + Gurobi logs).
- `group12/results/solve_times.csv` (1.1 to 181 s per operator).
- Analytical-model scores: `group12/results/analytical_check_<target>.csv`.
- Scripts: `prepare_operators.sh`, `run_optipim.sh`, `run_all.sh`, `sim_check.sh`,
  `score_analytical.py`, `gate2_fresh_clone.sh`. Docs: `group12/docs/SETUP.md`,
  `group12/docker/Dockerfile`, `group12/docs/report_member1.tex`.

## What we found (use in report and slides)

1. **Gurobi 13 presolve bug.** C1 and C2 on HBM-PIM at 32 banks were reported
   "infeasible" in 0.02 s. The model solves at 8, 16, 17 and 20 banks. With
   `Presolve 0` it solves to a proven optimum, so `run_optipim.sh` disables
   presolve for HBM-PIM. C1: objective 360,149 at 32 banks vs 707,519 at 16.
2. **C1 vs authors' heuristic** (analytical model): 870,912 vs 29,546,496, a
   34x reduction, almost all from input loading. The paper says about 100x; units
   of its figure not verified.
3. **Solver objective is not the model cost.** On HBM-PIM the model is 2-3x
   (conv) and 70-110x (matmul) above the ILP objective; SIMDRAM agrees within
   5-32%. Cause not found (guess: one-time weight loading).
4. **Model vs simulator** (`group12/results/sim_check.csv`): HBM-PIM within 2.1% (C3, M1, M2). SIMDRAM model underestimates by 38-76% (C3 +37.6%, M1 +41.2%, M2 +75.8%). C1, C2 not simulated (too slow).
5. **Your C1 mapping differs from the paper's Fig. 12b** (ours uses all 32
   banks: Q=2, C=16; paper: C=16 on 16 banks). Both are valid; we did not
   score the paper's mapping.
6. The weight-parallel heuristic baseline is shipped only as trace files; Member 2
   writes the generator.

## Left, in order

| When | Step | Command / note |
| --- | --- | --- |
| now | ask a teammate to review the pull request | GitHub: PR member1 -> main |
| done | simulator runs for C3, M1, M2 | `nohup bash group12/scripts/sim_check.sh optipim M1 M2 C3 > group12/results/sim_check.log 2>&1 &` then `cat group12/results/sim_check.csv`; C1 and C2 take long, skip or leave to Member 3 |
| Mon 26 Oct (Gate 2) | rerun from a fresh clone | `bash group12/scripts/gate2_fresh_clone.sh https://github.com/parikhwal291/OptiPIM-grp12.git` (all 10 should print `same`; different mapping with equal objective is fine) |
| by Fri 30 Oct | root README points to `group12/docs/SETUP.md`; fill FILL IN lines there | edit by hand |
| by Fri 30 Oct | test the Docker build on a machine with 40 GB free (optional) | SETUP.md section 5 |
| Mon 2 Nov | slide "Tools and setup": toolchain, Gurobi version, machine, solve-time table, the presolve story | from report_member1.tex |
| Tue 3 Nov | final consistency pass on the whole report | same names C1..M2, units, every figure from the CSVs |
| any time | Member 4 sends a padded shape | `bash group12/scripts/run_optipim.sh <ID> hbmpim` |

## If something breaks

| Symptom | Fix |
| --- | --- |
| `libgurobi120.so: cannot open` | `source ~/.bashrc`; check the symlink in `$GUROBI_HOME/lib` |
| `No Gurobi license found` | rerun `grbgetkey` on the university network/VPN |
| numpy/pandas missing | `source optipim_env/bin/activate` |
| a run says `fail_rc...` | read `group12/mappings/<target>/<op>/optipim.log` |
| "Model is infeasible" in 0.02 s | presolve bug: make sure `gurobi.env` with `Presolve 0` is in the working dir (the script does this for HBM-PIM) |
