# Makefile to automate the entire artifact evaluation flow

# Variable Definition
SHELL = /usr/bin/env bash
SETTIME := $(eval TIMESTAMP ?= $(shell date +%Y%m%d_%H%M%S))

#####################################################################################
# OptiPIM Installation
#####################################################################################
.PHONY: compile_llvm compile_optipim python_env

# Step shortcuts
compile_llvm: ./llvm-project
compile_optipim: compile_llvm ./build/bin/pim-opt
python_env: ./optipim_env

# Actuall steps
./llvm-project: 
	@git clone https://github.com/llvm/llvm-project.git; \
	cd llvm-project; git checkout 39048b69b85e530b9b8a4226d9043a0bd340fe8a; cd ..; bash build_llvm.sh;

./build/bin/pim-opt:
	@bash build_optipim.sh;

./optipim_env:
	@python3 -m venv optipim_env;source optipim_env/bin/activate;pip install -r requirements.txt

#####################################################################################
# Figure 9
#####################################################################################
.PHONY: fig9

# Step shortcuts
fig9: ./exp_results/fig9/fig9.pdf

# Actuall step to draw fig9
./exp_results/fig9/fig9.pdf:
	@bash fig9.sh

#####################################################################################
# Figure 10
#####################################################################################
.PHONY: fig10

# Step shortcuts
fig10: ./exp_results/fig10/fig10.pdf

# Actuall step to draw fig10
./exp_results/fig10/fig10.pdf:
	@cd experiments_scripts/gen_testcases; python create_new_testcases.py; cd ../Fig_10; python run_exp.py; python draw.py

#####################################################################################
# Figure 14
#####################################################################################
.PHONY: fig14

# Step shortcuts
fig14: ./exp_results/fig14/fig14.pdf

# Actuall step to draw fig14
./exp_results/fig14/fig14.pdf:
	@cd experiments_scripts/Fig_14; bash gen_mlir_files.sh; python run_exp.py;\
	cd ../..; bash fig14.sh

#####################################################################################
# Figure 15
#####################################################################################
.PHONY: fig15

# Step shortcuts
fig15: ./exp_results/fig15/fig15.pdf

# Actuall step to draw fig15
./exp_results/fig15/fig15.pdf:
	@cd experiments_scripts/Fig_15; bash gen_mlir_files.sh; python run_exp.py;\
	cd ../..; bash fig15.sh

#####################################################################################
# Helper Functions
#####################################################################################
.PHONY: test

# Sample test run of OptiPIM, detailed comfiguration can be found in test.sh
test:
	@mkdir Debug; mkdir Result; bash test.sh
