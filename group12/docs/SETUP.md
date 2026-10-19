# Group 12: build and run the OptiPIM flow

This is how Member 1 built and ran the flow. Anyone on Linux, WSL2 or Docker
can follow it. Windows and Mac teammates who only plot results do not need
the build: everything they need (mapping files, CSVs) is committed.

## Machine we used

| Item | Value |
| --- | --- |
| OS | Windows + WSL2, Ubuntu 22.04 (repo in `~/OptiPIM-grp12`, never under `/mnt/c`) |
| Compiler | g++ / clang from Ubuntu 22.04 (g++ 11.4.0) |
| Gurobi | 13.0.3, named-user academic licence (paper used 11.0.3) |
| CPU / RAM given to WSL | 6 logical cores, 11 GB |
| Build time | about 4 h 15 min |

## 1. Packages

```bash
sudo apt-get update
sudo apt-get install -y g++ clang lld ccache cmake ninja-build python3 python3-venv \
    libboost-regex-dev git curl gzip libreadline-dev unzip tmux
```

## 2. Gurobi 13.0.3

```bash
sudo tar xzf gurobi13.0.3_linux64.tar.gz -C /opt
echo 'export GUROBI_HOME="/opt/gurobi1303/linux64"'           >> ~/.bashrc
echo 'export PATH="${PATH}:${GUROBI_HOME}/bin"'              >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"' >> ~/.bashrc
source ~/.bashrc
grbgetkey <your-key>                       # on the university network or VPN
# The repo's cmake/modules/FindGUROBI.cmake only looks for gurobi110/gurobi120,
# so point the 12.0 name at the 13.0 library:
sudo ln -s $GUROBI_HOME/lib/libgurobi130.so $GUROBI_HOME/lib/libgurobi120.so
gurobi_cl $GUROBI_HOME/examples/data/coins.lp   # must end with "Optimal objective"
```

## 3. Build (once, hours)

```bash
git clone https://github.com/parikhwal291/OptiPIM-grp12.git ~/OptiPIM-grp12
cd ~/OptiPIM-grp12
make python_env && source optipim_env/bin/activate
make compile_optipim            # if "Killed": bash build_llvm.sh -t 2 && bash build_optipim.sh
ls build/bin/pim-opt simulator/build/ramulator2
make test && cat Result/Group.txt   # smoke test
```

## 4. Run our experiments

```bash
source optipim_env/bin/activate
bash group12/scripts/prepare_operators.sh   # C1..M2 -> group12/operators/, 32 banks
bash group12/scripts/run_all.sh             # 5 operators x 2 targets with OptiPIM
bash group12/scripts/sim_check.sh optipim   # replay each mapping in Ramulator2
```

Outputs:

- `group12/mappings/<hbmpim|simdram>/<op>/optipim.trace` mapping (authors' trace format)
- `group12/mappings/.../optipim.log` full pim-opt and Gurobi output
- `group12/results/solve_times.csv` wall and Gurobi solve time and objective per run
- `group12/results/sim_check.csv` Ramulator2 `memory_system_cycles` per mapping

One operator: `bash group12/scripts/run_optipim.sh C1 hbmpim`.
Gurobi 13 presolve wrongly reports the 3x3 convolutions infeasible at 32 banks, so the script runs pim-opt with `Presolve 0` (via a temporary `gurobi.env`). Use `PRESOLVE=1` to turn that off.
Another arch file: `ARCH=data/hbm_pim.json bash group12/scripts/run_optipim.sh C1 simdram`.

## Settings

| Setting | Value |
| --- | --- |
| pim-opt flags | `--data-layout-pass --trans-coeff-method 4 --number-counting-method 0 --obj-method 0 --storage-method 0` |
| HBM-PIM | `--target-device-type 1`, `data/hbm_pim.json`, sim `simulator/hbmpim_config.yaml`, `alloc_method=new` |
| SIMDRAM | `--target-device-type 0`, `data/dram.json` (as in `validation/validation.py`), sim `simulator/simdram_config.yaml`, `alloc_method=simdram` |
| Knobs | `data/knobs.json` |
| Banks | 32 (the `nn_models/*/16banks` files with `num_banks` changed to 32) |

## 5. Docker (Mac or anyone without WSL)

Needs a Gurobi **WLS academic** licence file (`~/gurobi.lic`).

```bash
cp group12/docker/.dockerignore.example .dockerignore
docker build -f group12/docker/Dockerfile -t optipim-g12 .          # hours
docker run --rm -it -v $HOME/gurobi.lic:/opt/gurobi/gurobi.lic optipim-g12
# inside the container:
bash group12/scripts/run_all.sh
```

On Apple Silicon add `--platform linux/amd64` (emulated, slow).

## 6. Reproducibility check (Gate 2)

```bash
bash group12/scripts/gate2_fresh_clone.sh https://github.com/parikhwal291/OptiPIM-grp12.git
```

It clones the repo fresh, reuses the existing build, reruns all mappings and
diffs them against the committed ones. A different mapping with the same
objective is fine (ILPs can have several optimal answers).
