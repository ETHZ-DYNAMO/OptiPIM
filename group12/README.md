
## Member 1 results and settings (Gate 1)

- All 10 OptiPIM mappings (C1, C2, C3, M1, M2 on HBM-PIM and SIMDRAM, 32 banks) are in `group12/mappings/<target>/<op>/optipim.trace`, with Gurobi logs next to them. Solve times: `group12/results/solve_times.csv`.
- Arch files: `data/hbm_pim.json` for HBM-PIM, `data/dram.json` for SIMDRAM.
- Gurobi 13.0.3 (paper: 11.0.3), `libgurobi120.so` symlinked to `libgurobi130.so`.
- Gurobi 13 presolve wrongly reports the 3x3 convolutions (C1, C2) infeasible on HBM-PIM at 32 banks. `run_optipim.sh` therefore runs HBM-PIM with `Presolve 0` (temporary `gurobi.env`); SIMDRAM keeps the default. With presolve off, C1 reaches a proven optimum of 360,149 at 32 banks (707,519 at 16 banks).
- Run everything: `bash group12/scripts/prepare_operators.sh && bash group12/scripts/run_all.sh`. Full build and Docker steps: `group12/docs/SETUP.md`.
