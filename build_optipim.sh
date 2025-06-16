#!/bin/bash

# Kill the whole scipt on Ctrl+C
trap "exit" INT

# Directory where the script is being ran from (must be directory where script
# is located!)
SCRIPT_CWD=$PWD

#### Helper functions ####

# Display list of possible options and exit
print_help_and_exit () {
    echo -e \
        "./build.sh [options]
        List of options:
            --disable-build-opt | -o  : don't use clang/lld/ccache to speed up builds
            --force-cmake | -f        : force cmake reconfiguration in current project (will not change subprojects)
            --release | -r            : build in \"Release\" mode (default is \"Debug\")
            --threads | -t            : number of concurrent threads to build on with ninja (by default, ninja spawns one thread per physical core)
            --help | -h               : display help message            
            "    
    
    exit
}

# Helper function to print large section title text
echo_section() {
    echo ""
    echo "# ===----------------------------------------------------------------------=== #"
    echo "# $1"
    echo "# ===----------------------------------------------------------------------=== #"
    echo ""
}

# Helper function to print subsection title text
echo_subsection() {
    echo "# ===--- $1 ---==="
}

# Helper function to exit script on failed command
exit_on_fail() {
    if [[ $? -ne 0 ]]; then
        if [[ ! -z $1 ]]; then
            echo -e "\n$1"
        fi
        echo ""
        echo_subsection "Build failed!"
        exit 1
    fi
}

# Helper function to create build directory and cd to it
create_build_directory() {
    cd "$SCRIPT_CWD" && mkdir -p $1 && cd $1
}

# Create symbolic link from the bin/ directory to an executable file built by
# the repository. The symbolic link's name is the same as the executable file.
# The path to the executable file must be passed as the first argument to this
# function and be relative to the repository's root. The function assumes that
# the bin/ directory exists and that the current working directory is the
# repository's root. 
create_symlink() {
    local src=$1
    local dst="bin/$(basename $1)"
    echo "$dst -> $src"
    ln -f --symbolic ../$src $dst
}

# Determine whether cmake should be re-configured by looking for a
# CMakeCache.txt file in the current working directory.
should_run_cmake() {
  if [[ -f "CMakeCache.txt" && $FORCE_CMAKE -eq 0 ]]; then
    echo "CMake configuration found, will not re-configure cmake"
    echo "Run script with -f or --force-cmake flag to re-configure cmake"
    echo ""
    return 1
  fi 
  return 0
}

# Run ninja using the number of threads provided as argument, if any. Otherwise,
# let ninja pick the number of threads to use  
run_ninja() {
  if [[ $NUM_THREADS -eq 0 ]]; then
    ninja
  else
    ninja -j "$NUM_THREADS"
  fi 
}

#### Parse arguments ####

# Loop over command line arguments and update script variables
CMAKE_FLAGS_SUPER="-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_LLD=ON" 
CMAKE_FLAGS_LLVM="$CMAKE_FLAGS_SUPER -DLLVM_CCACHE_BUILD=OFF" 
ENABLE_TESTS=0
FORCE_CMAKE=0
PARSE_NUM_THREADS=0
NUM_THREADS=0
BUILD_TYPE="Debug"

for arg in "$@"; 
do
    if [[ $PARSE_NUM_THREADS -eq 1 ]]; then
      NUM_THREADS=$arg
      PARSE_NUM_THREADS=0
    else
      case "$arg" in 
          "--disable-build-opt" | "-o")
              CMAKE_FLAGS_LLVM=""
              CMAKE_FLAGS_SUPER=""
              ;;
          "--force-cmake" | "-f")
              FORCE_CMAKE=1
              ;;
          "--release" | "-r")
              BUILD_TYPE="Release"
              ;;
          "--threads" | "-t")
              PARSE_NUM_THREADS=1
              ;;
          "--help" | "-h")
              print_help_and_exit
              ;;
          *)
              echo "Unknown argument \"$arg\", printing help and aborting"
              print_help_and_exit
              ;;
      esac
    fi
done

# Path to build directories
MLIR_BUILD_DIR="llvm-project/build"
PIMOPT_BUILD_DIR="build"

#### Build the project (submodules and superproject) ####

echo "####################################"
echo "############# OPTIPIM ##############"
echo "####################################"

# CMAKE
echo_section "Building OPTIPIM"
create_build_directory "$PIMOPT_BUILD_DIR"

if should_run_cmake ; then
    cmake ..
    exit_on_fail "Failed to cmake OPTIPIM"
fi

# Build
make
exit_on_fail "Failed to build OPTIPIM"
    
#### Build the simulator ####
echo_section "Building simulator"
cd ..
cd simulator
if [ -d "build" ]; then
    rm -rf build
fi
mkdir build
cd build
cmake ..
make -j
cd ../..

echo -e "\n\033[1;32m╔════════════════════════════════════════════════════════════════╗\033[0m"
echo -e "\033[1;32m║    ✨ Thank you for trying out OptiPIM                         ║\033[0m"
echo -e "\033[1;32m║    ✨ Please let us know if you have any questions!            ║\033[0m"
echo -e "\033[1;32m╚════════════════════════════════════════════════════════════════╝\033[0m\n"


