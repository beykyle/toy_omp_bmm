"""Build the two CHUQ energy-split calibrations and their nested-sampling launchers.

Splits the Ca-40(p,el) dXS data into three equal-point-count energy terciles, trains one chuq
model on the LOW tercile and one on the HIGH tercile, and holds the middle tercile out as the
test region compared in ``chuq_analysis.ipynb``.  Both models use the same chuq form and the
same Democratic-informed prior; they differ only in training data and in which imaginary
channel is active.  No likelihood rescaling (``likelihood_scaling = 1.0``), standard
``UnknownModelError`` fractional-error nuisance.

Writes, per variant, ``calibrations/chuq_{low,high}/config.pkl`` and a ``dynesty.slurm``
rendered from ``dynesty.slurm.template`` + ``site.env``, plus ``calibrations/fit_info.json``.

**Existing results are never overwritten.**  If ``config.pkl`` already exists the variant is
skipped (before the expensive MLE), so re-running this script -- or re-executing
``chuq_fit.ipynb`` -- cannot destroy a finished calibration.  Pass ``--force`` to rebuild.

Does not submit by default; it prints the ``sbatch`` lines::

    python build_configs.py                # build configs + launchers, print sbatch lines
    python build_configs.py --submit       # also sbatch them
    python build_configs.py --only low     # just one variant (fit_info.json is merged)
    python build_configs.py --force        # rebuild even if config.pkl exists
"""

from __future__ import annotations

import argparse
import json
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import dill

import chuq_io
import chuq_model

# Wall-clock and geometry come from site.env; see chuq_io.load_site_config.
JOB_NAME_PREFIX = "chuqS"


def write_executable(path, text):
    """Write a file and make it executable (it is a job script)."""
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def render_launcher(variant, site):
    """Fill ``dynesty.slurm.template`` with this site's settings for one variant."""
    if site["MPI_MODULE"]:
        module_block = (
            "# Load the site MPI module (MPI_MODULE in site.env).  Guarded so the script\n"
            "# still runs on machines without an environment-module system.\n"
            "if command -v module >/dev/null 2>&1; then\n"
            "    module purge\n"
            f"    module load {site['MPI_MODULE']}\n"
            "fi"
        )
    else:
        module_block = (
            "# MPI_MODULE is empty in site.env: using whatever MPI is already on PATH."
        )

    account_line = (
        f"#SBATCH -A {site['ACCOUNT']}"
        if site["ACCOUNT"]
        else "# (no ACCOUNT set in site.env)"
    )

    return chuq_io.SLURM_TEMPLATE.read_text().format(
        account_line=account_line,
        partition=site["PARTITION_LONG"] or site["PARTITION"],
        job=f"{JOB_NAME_PREFIX}-{variant}",
        ntasks=site["NTASKS"],
        mem_per_cpu=site["MEM_PER_CPU"],
        walltime=site["WALLTIME"],
        module_block=module_block,
        launcher=site["MPI_LAUNCHER"],
        proj=str(chuq_io.PROJECT),
        name=variant,
    )


def describe_split(bands, n_total, e_split_low, e_split_high):
    """Print how many datasets and points landed in each energy band."""
    print(
        f"N={n_total} pts over {sum(len(b) for b in bands.values())} datasets, "
        f"terciles E0={e_split_low:.1f}, E1={e_split_high:.1f} MeV"
    )
    ranges = {
        "low": f"[<= {e_split_low:.1f}]",
        "test": f"({e_split_low:.1f}, {e_split_high:.1f}]",
        "high": f"(> {e_split_high:.1f}]",
    }
    for band in chuq_io.BAND_NAMES:
        datasets = bands[band]
        points = chuq_io.total_points(datasets)
        note = "  [held out]" if band == "test" else ""
        print(
            f"  {band:4s} {ranges[band]:16s}: {len(datasets):2d} sets, "
            f"{points:4d} pts ({points / n_total:.0%}){note}"
        )


def main(submit=False, only=None, force=False):
    """Build the calibration configs and launchers.

    Args:
        submit: also ``sbatch`` the generated launchers.
        only: build just ``"low"`` or ``"high"``; ``fit_info.json`` is merged, not rewritten.
        force: rebuild a variant even if its ``config.pkl`` already exists.

    Returns:
        The list of launcher paths for the variants that were built.
    """
    site = chuq_io.load_site_config()
    # Read the template up front so a missing/renamed template fails before the slow MLE fits.
    if not chuq_io.SLURM_TEMPLATE.exists():
        raise FileNotFoundError(f"Launcher template not found: {chuq_io.SLURM_TEMPLATE}")

    datasets = chuq_io.load_datasets()
    e_split_low, e_split_high, n_total = chuq_io.tercile_bounds(datasets)
    bands = chuq_io.split_bands(datasets, e_split_low, e_split_high)
    describe_split(bands, n_total, e_split_low, e_split_high)

    # With --only, merge into the existing fit_info.json so the other variant's fields survive.
    info_path = chuq_io.CALIBRATIONS / "fit_info.json"
    info = json.loads(info_path.read_text()) if only and info_path.exists() else {}
    info.update(
        E0=e_split_low,
        E1=e_split_high,
        N=n_total,
        n_low=len(bands["low"]),
        n_test=len(bands["test"]),
        n_high=len(bands["high"]),
        pts_low=chuq_io.total_points(bands["low"]),
        pts_test=chuq_io.total_points(bands["test"]),
        pts_high=chuq_io.total_points(bands["high"]),
    )

    launchers, built_any = [], False
    for variant in chuq_io.VARIANT_NAMES:
        if only and variant != only:
            continue
        out_dir = chuq_io.run_dir(variant)
        launcher_path = out_dir / "dynesty.slurm"
        config_path = out_dir / "config.pkl"

        # Overwrite guard: never clobber a finished calibration by accident.
        if config_path.exists() and not force:
            print(
                f"  [{variant}] {config_path.relative_to(chuq_io.PROJECT)} already exists "
                "-- not overwritten (pass --force to rebuild)"
            )
            if not launcher_path.exists():
                write_executable(launcher_path, render_launcher(variant, site))
                print(f"         (re-rendered missing {launcher_path.name})")
            launchers.append(str(launcher_path))
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        prior_mean, prior_std, prior_lo, prior_hi = chuq_model.democratic_prior(variant)
        evidence = chuq_model.build_evidence(
            bands[variant], chuq_model.build_model(variant)
        )
        mle = chuq_model.mle_initial_guess(
            evidence, prior_mean, prior_lo, prior_hi, variant
        )
        truncation_hits = chuq_model.check_truncation(
            mle.x, prior_lo, prior_hi, variant
        )
        n_free = chuq_model.n_omp(variant)
        print(
            f"  [{variant}] {n_free} free  MLE -log L = {mle.fun:.1f}, "
            f"e^gamma = {np.exp(mle.x[n_free]):.3f}"
            + (
                f"  TRUNCATION HITS: {truncation_hits}"
                if truncation_hits
                else "  (no truncation hits)"
            )
        )
        config = chuq_model.build_calibration_config(
            variant,
            evidence,
            prior_mean,
            prior_std,
            prior_lo,
            prior_hi,
            mle.x,
            likelihood_scaling=1.0,
        )
        with open(config_path, "wb") as stream:
            dill.dump(config, stream)

        info[f"free_names_{variant}"] = chuq_model.free_names(variant)
        info[f"frozen_{variant}"] = chuq_model.VARIANTS[variant]["frozen"]
        info[f"prior_mean_{variant}"] = prior_mean.tolist()
        info[f"prior_std_{variant}"] = prior_std.tolist()
        info[f"prior_lo_{variant}"] = prior_lo.tolist()
        info[f"prior_hi_{variant}"] = prior_hi.tolist()
        info[f"mle_x_{variant}"] = mle.x.tolist()
        info[f"mle_negloglike_{variant}"] = float(mle.fun)

        write_executable(launcher_path, render_launcher(variant, site))
        launchers.append(str(launcher_path))
        built_any = True
        print(f"  built {out_dir.name}/config.pkl + dynesty.slurm")

    # Only rewrite fit_info.json if something was actually rebuilt -- otherwise a no-op run
    # would touch the record of a finished calibration.
    if built_any:
        info_path.parent.mkdir(parents=True, exist_ok=True)
        info_path.write_text(json.dumps(info, indent=2))
        print(f"\nwrote {info_path}")
    else:
        print("\nnothing rebuilt; calibrations/fit_info.json left untouched")

    if submit:
        for launcher in launchers:
            result = subprocess.run(
                ["sbatch", launcher], capture_output=True, text=True
            )
            print(
                "  sbatch",
                Path(launcher).parent.name,
                "->",
                result.stdout.strip() or result.stderr.strip(),
            )
    else:
        # Printed relative to the project root: these lines end up in chuq_fit.ipynb's stored
        # output, and an absolute path there would be meaningless to anyone else.
        print("\n(--submit not passed) submit them with:")
        for launcher in launchers:
            print("  sbatch", Path(launcher).relative_to(chuq_io.PROJECT))
    return launchers


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--submit", action="store_true", help="sbatch the launchers after building"
    )
    parser.add_argument(
        "--only",
        choices=chuq_io.VARIANT_NAMES,
        help="build only this variant (fit_info.json is merged, not rewritten)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild even if config.pkl already exists (overwrites a finished calibration)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    sys.exit(0 if main(args.submit, args.only, args.force) is not None else 1)
