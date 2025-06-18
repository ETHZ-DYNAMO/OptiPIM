# OptiPIM

```

 ██████╗ ██████╗ ████████╗██╗██████╗ ██╗███╗   ███╗
██╔═══██╗██╔══██╗╚══██╔══╝██║██╔══██╗██║████╗ ████║
██║   ██║██████╔╝   ██║   ██║██████╔╝██║██╔████╔██║
██║   ██║██╔═══╝    ██║   ██║██╔═══╝ ██║██║╚██╔╝██║
╚██████╔╝██║        ██║   ██║██║     ██║██║ ╚═╝ ██║
 ╚═════╝ ╚═╝        ╚═╝   ╚═╝╚═╝     ╚═╝╚═╝     ╚═╝
                                                   
```
> The ASCII art is generated using [Text To Art Generator](https://patorjk.com/software/taag/#p=display&f=Graffiti&t=OptiPIM)
---

Welcome to the **OptiPIM** artifact evaluation repository! This README will guide you through the entire workflow of installing OptiPIM, running sample tests, and reproducing the figures (Figures 9, 10, 14, and 15) from our paper. The process is fully automated using the provided `Makefile`.

> [!NOTE]
> We are slowly changing the name from pim-opt to optipim.
---

## 1. Prerequisites

To successfully build and run these scripts, please ensure you have:

1. **Dependencies required by the project**
    Most of our dependencies are provided as standard packages on most Linux distributions. OptiPIM needs a working C/C++ toolchain (compiler, linker), `cmake` and `ninja` for building the project, Python (3.11), and standard command-line tools like `git`.
    ```bash
    sudo apt-get update
    sudo apt-get install -y g++ clang lld ccache cmake ninja-build python3 python3-venv libboost-regex-dev git curl gzip libreadline-dev unzip
    ```
   Before moving on to the next step, refresh your environment variables in your current terminal to make sure that all newly installed tools are visible in your `PATH`. Alternatively, open a new terminal.
2. **Python Environment**  
   For a clean, isolated setup, we recommend creating a dedicated Python virtual environment before running any experiments or installing dependencies: 
   
   (a) **Ensure Python > 3.10 is installed**:
    ```bash
    python3 --version
    ``` 

    (b) **Create a virtual environment and install all dependencies**:
    ```bash
    make python_env
    ```

    (c) **Every time before running any python scripts please activate the env**:
    ```bash
    source optipim_env/bin/activate
    ```
    Once activated, your shell prompt may change to indicate that you are now in the `optipim_env` environment.

    (d) **Deactivate the environment**:
    If you want to exit the virtual environment, run:
    ```bash
    deactivate
    ``` 
3. **Gurobi**

    OptiPIM uses Gurobi to optimze the mapping for different PIM devices.
    
    (a) **Download Gurobi**

    Gurobi is available for Linux [here](https://www.gurobi.com/downloads/gurobi-software/) (log in required). The resulting downloaded file will be `gurobiXX.X.X_linux64.tar.gz`.

    (b) **Obtain a license**
    
    Gurobi offers free [academic license](https://www.gurobi.com/academia/academic-program-and-licenses/).
    - If you choose to use `Named-User Academic` license, please follow step (c).
    - If you are using `WLS Academic` license, you just need to put the license in `~/gurobi.lic` and you can skip step (c).

    (c) **Installation**

    To install Gurobi, first extract your downloaded file to your desired installation directory.

    Using the following command:

    ```bash
    # Replace x's with obtained license
    <your_installation_path>/gurobiXXXX/linux64/bin/grbgetkey xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    ```

    to pass your obtained license to Gurobi, which it stores in `~/gurobi.lic`.

    (d) **Configuring your environment**

    In addition to adding Gurobi to your path, OptiPIM's CMake requires the `GUROBI_HOME` environment variable to find headers and libraries.
    
    ```bash
    # Replace "gurobiXXXX" with the correct version
    export GUROBI_HOME="<your_gurobi_path>/gurobiXXXX/linux64"
    export PATH="${PATH}:${GUROBI_HOME}/bin"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$GUROBI_HOME/lib"
    ```

    These lines can be added to your shell initiation script, e.g. `~/.bashrc` or `~/.zshrc`, or used with any other environment setup method. Besides that please also add the name of your gurobi version to **line 8** in [FindGUROBI.cmake](./cmake/modules/FindGUROBI.cmake). You just need to add the first three digit (i.e. You are using `gurobi1201`, please add `gurobi120` at the end of **line 8**).
4. **Sufficient Disk Space**  
   - Building LLVM can require several gigabytes of disk space.
   - Running the analytical model validation may require a large disk space, if you run too many processes in parallel.  

Make sure all these dependencies are properly installed before proceeding.

---
## 2. Installation and Setup via Makefile

The `Makefile` provides targets for all major steps. Below is a rundown of these primary commands:

### 2.1 Compile OptiPIM
```bash
make compile_optipim
```
1. Clone and build the LLVM project.
2. Build the main OptiPIM executable and libraries.
3. Build the cycle-accurate simulator.
3. Create a `build/` directory containing the build artifacts.

> [!WARNING]
> Please make sure that there is no `#` in any of the folders in your path, as CMake does not allow `#` to appear in the compilation paths

---

## 3. Sample Test of OptiPIM
To verify your installation and that everything is working correctly, you can run:
```bash
make test
```
This command invokes `test.sh`, which runs a brief sanity check on the OptiPIM setup. It will:
  1. Use preconfigured test inputs.
  2. Show how the tool processes them.
  3. Provide any debug or informational logs indicating success or any failures.

---

## 4. Analyze Mapping Results with the Simulator
To analyze the performance of the generated mapping using the simulator
```bash
simulator/build/ramulator2 --config_file <arch_config_file> \
                           --param Frontend.path=<trace_file> \
                           --param Frontend.PimCodeGen.alloc_method=<alloc_method>
```
This command invokes the simulator to evaluate the generated mapping, needed parameters:
  1. `arch_config_file`:  
    - **hbm_pim**: `simulator/hbmpim_config.yaml`  
    - **simdram**: `simulator/simdram_config.yaml`
  2. `trace_file`: The path of the generated mapping
  3. `alloc_method`:  
    - **hbm_pim**: `new`  
    - **simdram**: `simdram`

---

## 5. Generating Figures from the Paper

> [!NOTE]
> The figures you generate may look slightly different from those in the paper. For **Figure 9**, we use a large number of randomly generated design points to verify the analytical model, leading to some run-to-run variation. For the other figures, we employ multi-threading to handle different design points in parallel, which can introduce minor discrepancies in the final results. Despite these differences, the overall trends remain consistent with those reported in the paper.

### 5.1 Figure 9
```bash
make fig9
```
1. Runs the script `fig9.sh`.
    - This will build the cycle-accurate simulator.
    - Randomly generate all simulation input files.
    - Run cycle accurate simulation and our analytical model.
2. Produces `fig9.pdf` under `exp_results/fig9/`.

> **Note**: We have provided pre-extracted data files (`validation_hbmpim.csv` and `validation_simdram.csv`) in `exp_results/fig9` for convenience. If you simply want to view the final validation figure without re-running the entire experiment, you can execute the plotting script (`validation/correlation_2.py`) directly. However, please be aware that running `make fig9` from scratch will overwrite these pre-extracted files.

---

## 6. Contact and Support

If you encounter any issues or have questions, please contact us!
We hope this artifact evaluation package helps you reproduce our results smoothly and provides a clear pathway to experimenting with OptiPIM!

---
**Thank you for trying out OptiPIM!**
