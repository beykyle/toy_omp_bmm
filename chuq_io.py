"""Paths, site configuration, and the loaders shared by every script and notebook here.

This module is the single place that knows *where things are*.  Nothing in this repository
should contain an absolute path: the project root is derived from this file's own location, so
a clone works from wherever it is checked out.

"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Paths -- everything is relative to this file
# --------------------------------------------------------------------------- #
PROJECT = Path(__file__).resolve().parent
CALIBRATIONS = PROJECT / "calibrations"
FIGURES = PROJECT / "figures"
SITE_ENV = PROJECT / "site.env"
SLURM_TEMPLATE = PROJECT / "dynesty.slurm.template"

# The curated Ca-40(p,el) dXS dataset, bundled under data/.  $CHUQ_SPLIT_DATA overrides it, so
# a large shared copy can be used instead of the in-repo one without editing any file.
DIFF_XS = Path(
    os.environ.get(
        "CHUQ_SPLIT_DATA", PROJECT / "data" / "ca40" / "data" / "diff_xs_observations.lz"
    )
)

# --------------------------------------------------------------------------- #
# Variant vocabulary
# --------------------------------------------------------------------------- #
# "low"  = trained on the low-energy tercile, surface-only absorption
# "high" = trained on the high-energy tercile, volume-only absorption
VARIANT_NAMES = ("low", "high")

# The three energy bands the data is cut into.  "test" is held out of both fits.
BAND_NAMES = ("low", "test", "high")

VARIANT_COLOR = {"low": "tab:blue", "high": "tab:red"}
VARIANT_FORM = {"low": "surface-only", "high": "volume-only"}


def band_role(variant, band):
    """How a band relates to a model: its own training band, the held-out band, or extrapolation."""
    if variant == band:
        return "train"
    return "held-out" if band == "test" else "extrap"


# --------------------------------------------------------------------------- #
# Site configuration (site.env)
# --------------------------------------------------------------------------- #
_SITE_DEFAULTS = {
    "ACCOUNT": "",
    "PARTITION": "",
    "PARTITION_LONG": "",
    "NTASKS": "65",
    "MEM_PER_CPU": "6G",
    "WALLTIME": "36:00:00",
    "ANALYSIS_CPUS": "8",
    "ANALYSIS_MEM": "32G",
    "MPI_MODULE": "",
    "MPI_LAUNCHER": "mpiexec -n $SLURM_NTASKS",
}

_ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$")


def load_site_config(path=None):
    """Parse ``site.env`` into a dict of strings.

    ``site.env`` is deliberately a plain ``KEY=value`` file rather than TOML or YAML so that the
    SLURM scripts can ``source`` the very same file this parser reads -- one definition of the
    cluster, used by both shell and Python.  Values keep any ``$VAR`` references verbatim; they
    are expanded later by the shell that runs the rendered job script, not here.
    """
    config = dict(_SITE_DEFAULTS)
    path = Path(path) if path is not None else SITE_ENV
    if not path.exists():
        return config
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _ASSIGNMENT.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        value = value.split(" #")[0].strip()  # trailing comment
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        config[key] = value
    return config


# --------------------------------------------------------------------------- #
# Dataset accessors
# --------------------------------------------------------------------------- #
def load_datasets(path=None):
    """Load the curated list of elastic dXS observations (the LZMA-pickled ``.lz`` file)."""
    import lzma

    import dill

    path = Path(path) if path is not None else DIFF_XS
    if not path.exists():
        raise FileNotFoundError(
            f"Curated dataset not found: {path}\n"
            "It should be bundled at data/ca40/data/diff_xs_observations.lz. "
            "If you keep it elsewhere, set $CHUQ_SPLIT_DATA to its path. "
            "See data/ca40/README.md for how it is produced."
        )
    with lzma.open(path, "rb") as stream:
        return dill.load(stream)


def elab(dataset):
    """Incident lab energy of one dataset, in MeV."""
    return dataset.constraint_workspace.kinematics.Elab


def n_points(dataset):
    """Number of angular points in one dataset."""
    return int(dataset.x.size)


def total_points(datasets):
    """Total number of angular points across a list of datasets."""
    return sum(n_points(d) for d in datasets)


def tercile_bounds(datasets):
    """Energy boundaries ``(e_split_low, e_split_high, n_total_points)`` of the three terciles.

    The terciles hold an equal number of *data points*, not an equal number of datasets and not
    equal energy width -- so the two fits are trained on comparable amounts of information.
    Boundaries fall on dataset energies (a dataset is never split), which is why the cut is
    only approximately 1/3--2/3.  ``searchsorted`` is clamped to the last index so that a
    degenerate dataset list cannot index past the end.
    """
    rows = sorted((elab(d), n_points(d)) for d in datasets)
    energies = np.array([r[0] for r in rows])
    counts = np.array([r[1] for r in rows])
    cumulative = np.cumsum(counts)
    n_total = int(counts.sum())
    last = len(energies) - 1
    e_low = energies[min(np.searchsorted(cumulative, n_total / 3.0), last)]
    e_high = energies[min(np.searchsorted(cumulative, 2.0 * n_total / 3.0), last)]
    return float(e_low), float(e_high), n_total


def split_bands(datasets, e_split_low, e_split_high):
    """Cut datasets into the three energy bands: ``[0, e_low]``, ``(e_low, e_high]``, ``(e_high, inf)``.

    Half-open on the right, so every dataset lands in exactly one band.  This is the single
    definition of the split -- ``build_configs.py`` and all three notebooks call it, so the
    training bands and the analysis bands cannot drift apart.
    """
    return {
        "low": [d for d in datasets if elab(d) <= e_split_low],
        "test": [d for d in datasets if e_split_low < elab(d) <= e_split_high],
        "high": [d for d in datasets if elab(d) > e_split_high],
    }


def band_of(dataset, e_split_low, e_split_high):
    """Which of the three energy bands one dataset belongs to."""
    if elab(dataset) <= e_split_low:
        return "low"
    return "test" if elab(dataset) <= e_split_high else "high"


def statistical_variance(dataset):
    """Per-point statistical variance of one dataset.

    Reaches into ``rxmc``'s private ``_obs`` because the object exposes no public accessor for
    the covariance; isolated here so a single edit tracks any upstream change.

    Only the **diagonal** is used.  That is a modelling choice, not an oversight: the
    calibration's ``UnknownModelError`` likelihood treats points as independent given the
    fractional model error, so correlated systematic/normalisation uncertainty is deliberately
    excluded here too.  Using the full covariance in the analysis while the fit used the
    diagonal would make the two inconsistent.
    """
    return np.diag(dataset._obs.statistical_covariance)


def statistical_sigma(dataset):
    """Per-point statistical 1-sigma of one dataset (square root of :func:`statistical_variance`)."""
    return np.sqrt(statistical_variance(dataset))


def predictive_variance(dataset, y_model, model_err_frac):
    """Total predictive variance: statistical noise plus the fractional model defect.

    ``var = sigma_stat^2 + (f * y_model)^2`` where ``f = exp(gamma)`` is the sampled fractional
    model error.  This mirrors ``rxmc.likelihood_model.UnknownModelError`` -- the likelihood the
    calibration actually maximised -- and **must not drift from it**.  If that class changes,
    change this with it, or every posterior-predictive band and coverage number in the analysis
    silently stops matching the fit.

    Broadcasts over leading axes, so ``y_model`` may be ``(n_points,)`` or
    ``(n_draws, n_points)`` and ``model_err_frac`` a scalar or a ``(n_draws, 1)`` column.
    """
    y_model = np.asarray(y_model, float)
    var_stat = statistical_variance(dataset)
    return var_stat + (np.asarray(model_err_frac, float) * y_model) ** 2


# --------------------------------------------------------------------------- #
# Calibration outputs
# --------------------------------------------------------------------------- #
_MISSING_RESULTS_HINT = (
    "\nThis file is produced by the calibration, which has not been run in this checkout.\n"
    "  1. python build_configs.py          # build the sampler configs\n"
    "  2. sbatch calibrations/chuq_low/dynesty.slurm\n"
    "     sbatch calibrations/chuq_high/dynesty.slurm\n"
    "See the README (section 'Run') for the full sequence."
)


def run_dir(variant):
    """Directory holding one calibration's outputs."""
    return CALIBRATIONS / f"chuq_{variant}"


def load_fit_info():
    """Load ``calibrations/fit_info.json`` (split bounds, priors, MLE seeds)."""
    path = CALIBRATIONS / "fit_info.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found." + _MISSING_RESULTS_HINT)
    with open(path) as stream:
        return json.load(stream)


def split_energies():
    """The two tercile boundary energies (MeV) recorded by the calibration."""
    info = load_fit_info()
    return info["E0"], info["E1"]


# Seed for the weighted -> equal-weight resampling below.  dynesty's `resample_equal` defaults
# to an UNSEEDED generator, which would make every analysis run produce slightly different
# numbers from the same stored posterior.  Fixing the seed makes the published figures and
# tables exactly reproducible from the committed .nc files.
POSTERIOR_RESAMPLE_SEED = 20260730


def load_equal_weight_posterior(variant, seed=POSTERIOR_RESAMPLE_SEED):
    """Equal-weight posterior samples for one variant, shape ``(n_samples, n_dims)``.

    Nested sampling does not produce posterior samples directly: dynesty returns points drawn
    from nested shells, each carrying an importance weight, and the raw ``idata.posterior``
    array is that *weighted* set.  Averaging it as-is would be wrong.  ``resample_equal`` draws
    from it in proportion to the weights, giving an unweighted sample that ordinary
    ``mean``/``percentile`` calls can be applied to.

    The resampling is seeded (see :data:`POSTERIOR_RESAMPLE_SEED`) so repeated runs of the
    analysis give identical numbers.  Pass a different ``seed`` to check that a conclusion is
    not an artefact of one particular resampling.

    The last column is the nuisance parameter ``log f_err``; the preceding ``n_omp(variant)``
    columns are the optical-model parameters, in ``chuq_model.free_names(variant)`` order.
    """
    import arviz as az
    from dynesty.utils import resample_equal

    path = run_dir(variant) / "dynesty_idata.nc"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found." + _MISSING_RESULTS_HINT)
    idata = az.from_netcdf(path)
    theta = np.asarray(idata.posterior["theta"].values)
    weights = np.asarray(idata.sample_stats["importance_weight"].values).reshape(-1)
    return resample_equal(
        theta.reshape(-1, theta.shape[-1]),
        weights / weights.sum(),
        rstate=np.random.default_rng(seed),
    )


def load_all_posteriors(seed=POSTERIOR_RESAMPLE_SEED):
    """``{variant: equal-weight samples}`` for both variants."""
    return {v: load_equal_weight_posterior(v, seed=seed) for v in VARIANT_NAMES}


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
FIGURE_DPI = 140


def use_paper_style():
    """Apply the shared matplotlib style used by every figure in this repository."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.autolayout": False,
        }
    )


def save_figure(fig, name):
    """Save a figure into ``figures/`` and return its path."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    return path


def plot_dataset(ax, dataset, label="data", color="k", ms=4, **kwargs):
    """Draw one dataset's angular distribution with statistical error bars, sorted by angle.

    Returns the angle sort order, so the caller can plot bands on the same abscissa.
    """
    theta_deg = np.rad2deg(np.asarray(dataset.x))
    order = np.argsort(theta_deg)
    ax.errorbar(
        theta_deg[order],
        np.asarray(dataset.y)[order],
        statistical_sigma(dataset)[order],
        ls="none",
        marker=".",
        ms=ms,
        color=color,
        label=label,
        **kwargs,
    )
    ax.set_yscale("log")
    return order


def visualization_angles_deg(dataset):
    """The dense angular grid (degrees) used for smooth model curves."""
    return np.rad2deg(np.asarray(dataset.visualization_workspace.angles))
