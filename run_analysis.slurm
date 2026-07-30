#!/bin/bash --login
# Headless in-place execution of one notebook.  Outputs are written back into the .ipynb.
#
#     ./submit.sh analysis                      # chuq_analysis.ipynb (account/partition from site.env)
#     ./submit.sh analysis chuq_coverage.ipynb
#
# or directly, if you prefer to pass SLURM options yourself:
#     sbatch -A <account> -p <partition> run_analysis.slurm chuq_coverage.ipynb
#
# The #SBATCH lines below are defaults only: submit.sh overrides them from site.env, and any
# option on the sbatch command line wins over a directive here.  Account and partition are
# deliberately absent -- they are site-specific and come from site.env.
#SBATCH --nodes 1
#SBATCH --cpus-per-task 8
#SBATCH --time 02:00:00
#SBATCH --mem 32G
#SBATCH --job-name chuqS-analysis
#SBATCH --output analysis-%J.SLURMout

set -eu

# Locating the project root under SLURM: sbatch COPIES this script into its spool directory
# (/var/lib/slurmd/job<ID>/slurm_script), so $BASH_SOURCE points there, not at the repo.  Use
# the explicit variable submit.sh exports, then SLURM's submission directory, and only fall
# back to self-location when running this script directly outside SLURM.
PROJECT="${CHUQ_PROJECT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"

if [ ! -f "$PROJECT/chuq_io.py" ]; then
    echo "ERROR: '$PROJECT' does not look like the chuq_split project (no chuq_io.py)." >&2
    echo "       Submit with ./submit.sh, or set CHUQ_PROJECT to the repository root." >&2
    exit 1
fi
cd "$PROJECT"

if command -v module >/dev/null 2>&1; then
    module purge
fi

# Do NOT unset X43I_DATAPATH here. It looks like dead weight (nothing in this repo queries
# EXFOR), but x4i3 -- imported transitively by rxmc -- treats an unset variable as "the bundled
# database is missing" and DOWNLOADS a ~100 MB EXFOR tarball into site-packages at import time.
# Inside a batch job that is a silent, repeated network fetch. See README, "Data provenance".

if [ ! -f "$PROJECT/.venv/bin/activate" ]; then
    echo "ERROR: $PROJECT/.venv not found. Run 'uv sync' on a login node first." >&2
    exit 1
fi
source "$PROJECT/.venv/bin/activate"

# Notebook analysis is threaded (numpy/BLAS), unlike the rank-parallel sampling jobs.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

NOTEBOOK="${1:-chuq_analysis.ipynb}"
echo "executing $NOTEBOOK in $PROJECT"
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 "$NOTEBOOK"
