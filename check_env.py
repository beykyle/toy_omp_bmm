"""Preflight check: is this checkout ready to run?

    python check_env.py

Verifies the interpreter, the imports, the bundled data, the site configuration, MPI, and which
calibration artifacts are present.  Run it first on a new machine -- it turns the failure modes
that would otherwise surface as opaque tracebacks (a stale ``X43I_DATAPATH``, a missing dataset,
mpi4py linked against the wrong MPI) into one readable report.

Exit status is 0 if everything essential passed, 1 otherwise.  Warnings do not fail the run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

OK, WARN, FAIL = "ok  ", "WARN", "FAIL"
_results = []


def report(status, label, detail=""):
    _results.append(status)
    print(f"  [{status}] {label}" + (f"\n         {detail}" if detail else ""))


def check_python():
    version = sys.version_info
    detail = f"{sys.version.split()[0]} at {sys.executable}"
    if version[:2] == (3, 11):
        report(OK, "Python 3.11", detail)
    elif version >= (3, 11):
        report(
            WARN,
            f"Python {version.major}.{version.minor} (expected 3.11)",
            detail
            + "\n         .python-version pins 3.11; uv.lock selects a different arviz major"
            "\n         version on 3.12+, which changes the analysis API.",
        )
    else:
        report(FAIL, f"Python {version.major}.{version.minor} is too old", "need >= 3.11")


def check_x4_database():
    """The one environment variable that can break -- or silently stall -- every import here.

    Nothing in this study queries EXFOR, but `rxmc` imports `exfor_tools`, which imports
    `x4i3`, which resolves its database at import time. All three outcomes matter.
    """
    raw = os.environ.get("X43I_DATAPATH")
    if raw is None:
        report(
            WARN,
            "X43I_DATAPATH unset",
            "x4i3 will treat its bundled database as missing and DOWNLOAD a ~100 MB EXFOR"
            "\n         tarball into site-packages on the first import (one-off, then cached)."
            "\n         Nothing here needs EXFOR, but the import chain forces the check."
            "\n         Point it at an unpacked EXFOR directory to skip the download.",
        )
        return
    path = Path(raw)
    if not path.exists():
        report(
            FAIL,
            "X43I_DATAPATH points at a path that does not exist",
            f"{raw}\n         x4i3 raises at IMPORT time, so this breaks every script,"
            "\n         notebook and MPI worker here -- even though none of them needs EXFOR."
            "\n         Fix: point it at a real unpacked EXFOR directory, or unset it and let"
            "\n         x4i3 download its own copy (see data/ca40/README.md).",
        )
        return
    if not list(path.glob("X4-20*")):
        report(
            FAIL,
            "X43I_DATAPATH exists but has no X4-20* tag file",
            f"{raw}\n         x4i3 raises at import time unless the directory contains a tag"
            "\n         file named like 'X4-2025-12-31'.",
        )
        return
    report(OK, "X43I_DATAPATH set and valid", f"{raw}\n         (no download will occur)")


def check_imports():
    try:
        import chuq_io  # noqa: F401
    except Exception as exc:  # pragma: no cover - diagnostic path
        report(FAIL, "import chuq_io", repr(exc))
        return None
    report(OK, "import chuq_io")

    try:
        import chuq_model
    except Exception as exc:
        report(
            FAIL,
            "import chuq_model (rxmc / jitr / exfor_tools)",
            f"{exc!r}\n         Run `uv sync`. If this is a FileNotFoundError from x4i3,"
            "\n         see the X43I_DATAPATH check above.",
        )
        return None
    report(
        OK,
        "import chuq_model",
        f"{chuq_model.n_omp('low')} free params per variant + 1 nuisance",
    )
    return chuq_model


def check_data():
    import chuq_io

    if chuq_io.DIFF_XS.exists():
        size_mb = chuq_io.DIFF_XS.stat().st_size / 1e6
        try:
            datasets = chuq_io.load_datasets()
            e_low, e_high, n_total = chuq_io.tercile_bounds(datasets)
            report(
                OK,
                f"dataset loads ({size_mb:.1f} MB)",
                f"{len(datasets)} datasets, {n_total} points; "
                f"terciles at {e_low:.1f} / {e_high:.1f} MeV",
            )
        except Exception as exc:
            report(FAIL, "dataset present but failed to load", repr(exc))
    else:
        report(
            FAIL,
            "curated dataset missing",
            f"{chuq_io.DIFF_XS}\n         It ships in the repo at "
            "data/ca40/data/. Set $CHUQ_SPLIT_DATA if you keep it elsewhere.",
        )


def check_site_config():
    import chuq_io

    if not chuq_io.SITE_ENV.exists():
        report(WARN, "site.env missing", "SLURM/MPI defaults will be used")
        return
    site = chuq_io.load_site_config()
    report(
        OK,
        "site.env parsed",
        f"account={site['ACCOUNT'] or '(none)'} partition={site['PARTITION'] or '(none)'} "
        f"ntasks={site['NTASKS']}\n         launcher={site['MPI_LAUNCHER']} "
        f"module={site['MPI_MODULE'] or '(none, using PATH)'}",
    )
    if not shutil.which("sbatch"):
        report(
            WARN,
            "sbatch not found",
            "No SLURM here. The calibration can still be run by hand with the launcher"
            "\n         command in calibrations/chuq_*/dynesty.slurm.",
        )


def check_mpi():
    try:
        from mpi4py import MPI
    except Exception as exc:
        report(
            WARN,
            "mpi4py not importable",
            f"{exc!r}\n         Only the sampling jobs need it; notebooks and smoke_test do not."
            "\n         Fix with `bash setup_mpi.sh`.",
        )
        return
    library = MPI.Get_library_version().split(",")[0].strip()
    try:
        import schwimmbad  # noqa: F401

        has_pool = True
    except Exception:
        has_pool = False
    if not has_pool:
        report(WARN, "schwimmbad missing", "black-box-bayes --require-mpi needs it")
        return
    launcher = "mpiexec" if shutil.which("mpiexec") else None
    report(
        OK,
        "mpi4py + schwimmbad",
        f"{library}\n         launcher on PATH: {launcher or 'none found (srun?)'}",
    )
    if shutil.which("mpicc"):
        try:
            built_against = subprocess.run(
                ["mpicc", "--version"], capture_output=True, text=True, timeout=10
            ).stdout.splitlines()[0]
            report(OK, "mpicc available", built_against)
        except Exception:
            pass


def check_calibrations():
    import chuq_io

    for variant in chuq_io.VARIANT_NAMES:
        run = chuq_io.run_dir(variant)
        posterior = run / "dynesty_idata.nc"
        config = run / "config.pkl"
        if posterior.exists():
            report(
                OK,
                f"{variant}: posterior present",
                f"{posterior.relative_to(chuq_io.PROJECT)} "
                f"({posterior.stat().st_size / 1e6:.1f} MB) -- analysis notebooks will run",
            )
        else:
            report(
                WARN,
                f"{variant}: no posterior yet",
                "Run the calibration: python build_configs.py, then submit"
                f"\n         calibrations/chuq_{variant}/dynesty.slurm",
            )
        if not config.exists():
            report(
                WARN,
                f"{variant}: config.pkl absent",
                "Not shipped in the repo (42 MB, regenerable). Run `python build_configs.py`"
                "\n         to rebuild it -- existing results are never overwritten.",
            )


def main():
    print(f"chuq_split preflight check\n{'=' * 60}")
    print("\nInterpreter and environment")
    check_python()
    check_x4_database()
    print("\nImports")
    if check_imports() is None:
        print("\nStopping: the core imports failed; fix those first.")
        return 1
    print("\nData")
    check_data()
    print("\nSite configuration")
    check_site_config()
    print("\nMPI (needed only for the sampling jobs)")
    check_mpi()
    print("\nCalibration artifacts")
    check_calibrations()

    n_fail = _results.count(FAIL)
    n_warn = _results.count(WARN)
    print(f"\n{'=' * 60}")
    if n_fail:
        print(f"{n_fail} failure(s), {n_warn} warning(s) -- not ready to run.")
        return 1
    print(f"No failures, {n_warn} warning(s). Ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
