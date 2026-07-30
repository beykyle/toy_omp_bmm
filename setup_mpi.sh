#!/bin/bash --login
# Rebuild mpi4py against the MPI that is actually installed on this machine.
#
# Why this is needed: `uv sync` installs the PyPI mpi4py wheel, which bundles its own MPICH.
# That is fine for the notebooks, the smoke test, and any serial (--no-mpi) run, but a bundled
# MPICH cannot talk to a resource manager's OpenMPI/PMIx, so the multi-rank sampling jobs hang
# or abort.  Rebuilding from source against the local MPI fixes that.
#
# Run once, on a LOGIN node, after `uv sync`:
#     bash setup_mpi.sh
#
# Which MPI gets used is controlled by MPI_MODULE in site.env:
#   - set   -> that environment module is loaded (the cluster case);
#   - empty -> whatever `mpicc` is already on PATH (system package, conda, Homebrew, ...).
# Either way the build itself is MPI-implementation agnostic: OpenMPI, MPICH, Intel MPI and
# MVAPICH all work.  If your site's MPI is MPICH-ABI compatible you can skip this script
# entirely and keep the stock wheel.
set -eu

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$PROJECT/.venv/bin/python"

if [ ! -x "$PY" ]; then
    echo "ERROR: $PY not found. Run 'uv sync' first." >&2
    exit 1
fi

# shellcheck disable=SC1091
[ -f "$PROJECT/site.env" ] && source "$PROJECT/site.env"
MPI_MODULE="${MPI_MODULE:-}"

if [ -n "$MPI_MODULE" ] && command -v module >/dev/null 2>&1; then
    echo "loading MPI module: $MPI_MODULE"
    module purge
    module load "$MPI_MODULE"
elif [ -n "$MPI_MODULE" ]; then
    echo "WARNING: MPI_MODULE='$MPI_MODULE' is set in site.env but this machine has no" >&2
    echo "         'module' command; falling back to the MPI on PATH." >&2
fi

if ! command -v mpicc >/dev/null 2>&1; then
    echo "ERROR: no mpicc on PATH." >&2
    echo "       Either set MPI_MODULE in site.env to your site's MPI module, or install an" >&2
    echo "       MPI development package (e.g. 'apt install libopenmpi-dev', 'brew install" >&2
    echo "       open-mpi', 'conda install -c conda-forge openmpi mpi4py')." >&2
    exit 1
fi
echo "building mpi4py against: $(command -v mpicc)"

# setuptools<81 still provides the distutils-era hooks mpi4py's setup.py uses; --no-build-isolation
# makes the build see the MPI loaded above rather than a clean throwaway environment.
uv pip install --python "$PY" "setuptools<81" "Cython>=3.0.1"
uv pip install --python "$PY" --reinstall --no-binary mpi4py --no-build-isolation mpi4py

"$PY" -c "from mpi4py import MPI; print('mpi4py OK, links:', MPI.Get_library_version().split(',')[0])"
echo "mpi4py is now built against $(command -v mpicc)"
