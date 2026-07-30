#!/bin/bash --login
# JupyterLab server on a compute node, for viewing or re-running the notebooks interactively.
#
#     ./submit.sh jupyter        # account/partition from site.env
#
# Then follow the `ssh -L` line printed in the job log to reach it from your laptop.
#
# The #SBATCH lines below are defaults only: submit.sh overrides them from site.env.  Account
# and partition are deliberately absent -- they are site-specific and come from site.env.
#SBATCH --nodes 1
#SBATCH --cpus-per-task 8
#SBATCH --time 18:00:00
#SBATCH --mem 48G
#SBATCH --job-name chuqS-nb
#SBATCH --output jupyter-%J.log

set -eu

# See run_analysis.slurm: sbatch copies this script to its spool dir, so self-location by
# $BASH_SOURCE does not find the repository.
PROJECT="${CHUQ_PROJECT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"

if [ ! -f "$PROJECT/chuq_io.py" ]; then
    echo "ERROR: '$PROJECT' does not look like the chuq_split project (no chuq_io.py)." >&2
    echo "       Submit with ./submit.sh, or set CHUQ_PROJECT to the repository root." >&2
    exit 1
fi
cd "$PROJECT"

PORT="${JUPYTER_PORT:-9999}"
NODE="$(hostname -s)"
USER_NAME="$(whoami)"

if command -v module >/dev/null 2>&1; then
    module purge
fi

if [ ! -f "$PROJECT/.venv/bin/activate" ]; then
    echo "ERROR: $PROJECT/.venv not found. Run 'uv sync' on a login node first." >&2
    exit 1
fi
source "$PROJECT/.venv/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cat <<EOF

==========================================================================
 JupyterLab on ${NODE}:${PORT}
     ssh -N -L ${PORT}:${NODE}:${PORT} ${USER_NAME}@<your-login-node>
 then open the http://127.0.0.1:${PORT}/lab?token=... URL printed below.
==========================================================================

EOF

jupyter-lab --no-browser --port="${PORT}" --ip="${NODE}" --notebook-dir="$PROJECT"
