#!/bin/bash
# Submit a job with the account/partition/resources from site.env, so no SLURM setting is
# hardcoded in a job script.
#
#     ./submit.sh sample low                  # nested sampling, low-energy model
#     ./submit.sh sample high
#     ./submit.sh analysis                    # execute chuq_analysis.ipynb headlessly
#     ./submit.sh analysis chuq_coverage.ipynb
#     ./submit.sh jupyter                     # JupyterLab on a compute node
#
# Everything it passes comes from site.env; edit that file, not this one.
set -eu

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT"

if [ ! -f site.env ]; then
    echo "ERROR: site.env not found in $PROJECT" >&2
    exit 1
fi
# shellcheck disable=SC1091
source site.env

usage() { sed -n '2,12p' "$0" >&2; exit 1; }
[ $# -ge 1 ] || usage

# Only pass -A / -p when site.env actually defines them, so sites without accounts work.
# CHUQ_PROJECT is exported into the job because sbatch copies the job script into its spool
# directory, where the script can no longer locate the repository from its own path.
opts=(--export="ALL,CHUQ_PROJECT=$PROJECT")
[ -n "${ACCOUNT:-}" ] && opts+=(-A "$ACCOUNT")

case "$1" in
    sample)
        [ $# -eq 2 ] || { echo "usage: ./submit.sh sample low|high" >&2; exit 1; }
        variant="$2"
        launcher="calibrations/chuq_${variant}/dynesty.slurm"
        if [ ! -f "$launcher" ]; then
            echo "ERROR: $launcher not found. Run 'python build_configs.py' first." >&2
            exit 1
        fi
        # The sampling launcher already carries its site settings: build_configs.py rendered
        # them in from site.env when it wrote the file.
        exec sbatch "$launcher"
        ;;
    analysis)
        [ -n "${PARTITION:-}" ] && opts+=(-p "$PARTITION")
        [ -n "${ANALYSIS_CPUS:-}" ] && opts+=(--cpus-per-task "$ANALYSIS_CPUS")
        [ -n "${ANALYSIS_MEM:-}" ] && opts+=(--mem "$ANALYSIS_MEM")
        [ -n "${ANALYSIS_WALLTIME:-}" ] && opts+=(--time "$ANALYSIS_WALLTIME")
        exec sbatch "${opts[@]}" run_analysis.slurm "${2:-chuq_analysis.ipynb}"
        ;;
    jupyter)
        [ -n "${PARTITION:-}" ] && opts+=(-p "$PARTITION")
        exec sbatch "${opts[@]}" run_jupyter.slurm
        ;;
    *)
        usage
        ;;
esac
