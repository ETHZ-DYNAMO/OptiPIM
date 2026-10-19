# Reproducible build of the OptiPIM flow for group 12 (Ubuntu 22.04, x86-64).
# Build (from the repo root, takes hours, needs ~40 GB disk):
#   docker build -f group12/docker/Dockerfile -t optipim-g12 .
# Run (Gurobi needs a WLS academic licence file; a named-user licence is tied to one machine):
#   docker run --rm -it -v $HOME/gurobi.lic:/opt/gurobi/gurobi.lic optipim-g12
#   then inside: bash group12/scripts/run_all.sh
# Apple Silicon Macs: add --platform linux/amd64 to both commands (slow, emulated).
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      g++ clang lld ccache cmake ninja-build python3 python3-venv python3-pip \
      libboost-regex-dev git curl wget ca-certificates gzip libreadline-dev unzip \
    && rm -rf /var/lib/apt/lists/*

# Gurobi 13.0.3, same version we used on WSL. The repo's FindGUROBI.cmake only
# looks for gurobi110/gurobi120, so we add the same libgurobi120.so symlink.
ARG GRB_VER=13.0.3
ARG GRB_DIR=gurobi1303
RUN wget -q https://packages.gurobi.com/${GRB_VER%.*}/gurobi${GRB_VER}_linux64.tar.gz \
    && tar xzf gurobi${GRB_VER}_linux64.tar.gz -C /opt && rm gurobi${GRB_VER}_linux64.tar.gz \
    && ln -s /opt/${GRB_DIR}/linux64/lib/libgurobi130.so /opt/${GRB_DIR}/linux64/lib/libgurobi120.so
ENV GUROBI_HOME=/opt/${GRB_DIR}/linux64
ENV PATH="${PATH}:${GUROBI_HOME}/bin"
ENV LD_LIBRARY_PATH="${GUROBI_HOME}/lib"
ENV GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic

WORKDIR /optipim
COPY . .
RUN make python_env
# LLVM + pim-opt + Ramulator2. Lower -t if the build machine runs out of memory.
RUN bash -c "source optipim_env/bin/activate && make compile_optipim"
RUN test -x build/bin/pim-opt && test -x simulator/build/ramulator2

ENV VIRTUAL_ENV=/optipim/optipim_env
ENV PATH="/optipim/optipim_env/bin:${PATH}"
CMD ["/bin/bash"]
