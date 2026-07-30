"""CHUQ (CH89) global optical potential for Ca-40 (p, el), for the energy-split study.

Uses the *actual* ``jitr.optical_potentials.chuq`` model (Pruitt et al. 2023, PRC 107, 014602;
Varner et al. 1991), fit as ``dXS/dRuth`` only, with spin-orbit frozen (so the analyzing power Ay
is dropped).  Two copies of THIS model are calibrated on two disjoint energy bands (see
``build_configs.py``) and compared in a held-out middle band.

The full chuq form has 22 parameters, sampler order::

    V0 Ve Vt r0 r0_0 a0  Wv0 Wve0 Wvew rw rw_0 aw  Ws0 Wst Wse0 Wsew  Vso rso rso_0 aso  rc rc_0

This module builds TWO variants of the chuq model, one per training band, with **different
absorption forms** (the physics each band can actually constrain):

- ``"low"``  -- **surface-only** absorption: imaginary volume frozen off (``Wv0 = 0``, and the now
  inert ``Wve0, Wvew`` frozen).  Free: ``V0 Ve r0 a0 rw aw  Ws0 Wse0 Wsew``.
- ``"high"`` -- **volume-only** absorption: imaginary surface frozen off (``Ws0 = 0``, and the now
  inert ``Wse0, Wsew`` frozen).  Free: ``V0 Ve r0 a0 Wv0 Wve0 Wvew rw aw``.

In chuq's central form the imaginary volume and surface **share** the geometry ``(rw, aw)``,
so ``rw, aw`` are free in both (surface geometry for ``low``, volume geometry for ``high``).

Common to both variants, the following are frozen (single isotope, spin-orbit frozen):

- **spin-orbit** ``Vso, rso, rso_0, aso`` -- (this also freezes Ay);
- **asymmetry terms** ``Vt, Wst`` -- multiply ``alpha=(N-Z)/A`` - non-indentifiable with isoscalar
    depth for a single isotope -> frozen;
- **radial intercepts** ``r0_0, rw_0, rc_0`` -- non-identifiable with r0, rw, rc for a single
    A -> frozen;
- **Coulomb radius** ``rc`` -- the elastic dXS/Ruth ratio is nearly insensitive to it
  (``|dlogL| < 1`` across ``rc in [1.1, 1.5]`` vs ~100-1000x that for ``r0``) -> frozen.

So each variant samples **9 free physics params + 1 model-defect nuisance = 10 dims**.  Frozen values
are the CHUQ **Democratic** posterior means (except ``Wv0``/``Ws0`` = 0 exactly).  The prior on the
free params is a truncated normal centred at the Democratic posterior mean with the Democratic
posterior std, fixed a priori -- the prior is chosen before seeing this dataset and is not tuned
to it, so the MLE is a starting point for sampling rather than a MAP estimate to report.
``likelihood_scaling = 1.0`` (no rescaling).

Debugging note: the ``-inf`` guards in :class:`CalibrationConfigBB` swallow exceptions during
sampling, which is what you want for a 36-hour run but not while developing.  Set the
environment variable ``CHUQ_DEBUG=1`` to make them re-raise instead.
"""

from __future__ import annotations

import os

import numpy as np
from scipy import optimize, stats

# rxmc uses np.trapz but newer numpy versions renamed it to np.trapezoid
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

from jitr.optical_potentials import chuq
from rxmc.config import CalibrationConfig, ParameterConfig
from rxmc.constraint import Constraint
from rxmc.elastic_diffxs_model import ElasticDifferentialXSModel
from rxmc.evidence import Evidence
from rxmc.likelihood_model import UnknownModelError
from rxmc.params import Parameter

# Set CHUQ_DEBUG=1 to turn the sampling-time "-inf instead of an exception" guards back into
# real tracebacks.  Leave it unset for production runs: a single bad proposal must not kill a
# 65-rank job.
DEBUG = bool(os.environ.get("CHUQ_DEBUG"))

# Proton projectile as chuq's (A, Z) tuple.
PROTON = (1, 1)


class TruncatedNormalPrior:
    """Independent truncated-normal prior over a parameter vector.

    Each coordinate is ``N(mu_i, sigma_i)`` restricted to ``[lower_i, upper_i]``, with no
    correlations.  Provides everything the samplers here need:
    :meth:`priors_as_scipy_objects` (for ``rxmc``'s ParameterConfig), :meth:`prior_transform`
    (dynesty's unit-cube map), :meth:`logpdf`, :meth:`rvs`, and ``.dim``.

    Args:
        mu: per-coordinate means.
        sigma: per-coordinate standard deviations (before truncation).
        lower: per-coordinate lower truncation bounds.
        upper: per-coordinate upper truncation bounds.
        seed: seed for the generator backing :meth:`rvs`.
    """

    def __init__(self, mu, sigma, lower, upper, seed=123):
        self.mu = np.asarray(mu, float)
        self.sigma = np.asarray(sigma, float)
        self.lower = np.asarray(lower, float)
        self.upper = np.asarray(upper, float)
        self.dim = len(self.mu)
        self.rng = np.random.default_rng(seed)
        # scipy's truncnorm takes its bounds in units of sigma about mu, not in the
        # parameter's own units -- these are those standardized bounds, not the raw ones.
        self.a_std = (self.lower - self.mu) / self.sigma
        self.b_std = (self.upper - self.mu) / self.sigma

    def priors_as_scipy_objects(self):
        """One frozen ``scipy.stats.truncnorm`` per coordinate."""
        return [
            stats.truncnorm(
                a=self.a_std[i], b=self.b_std[i], loc=self.mu[i], scale=self.sigma[i]
            )
            for i in range(self.dim)
        ]

    def logpdf(self, theta):
        """Log prior density of one parameter vector, or of each row of a 2-D array."""
        theta = np.asarray(theta, float)
        single_vector = theta.ndim == 1
        theta = np.atleast_2d(theta)
        log_density = np.zeros(theta.shape[0])
        for i in range(self.dim):
            log_density += stats.truncnorm.logpdf(
                theta[:, i],
                a=self.a_std[i],
                b=self.b_std[i],
                loc=self.mu[i],
                scale=self.sigma[i],
            )
        return float(log_density[0]) if single_vector else log_density

    def rvs(self, n):
        """Draw ``n`` samples, shape ``(n, dim)``."""
        out = np.zeros((n, self.dim))
        for i in range(self.dim):
            out[:, i] = stats.truncnorm.rvs(
                a=self.a_std[i],
                b=self.b_std[i],
                loc=self.mu[i],
                scale=self.sigma[i],
                size=n,
                random_state=self.rng,
            )
        return out

    def prior_transform(self, u):
        """Map a unit-cube point to parameter space (per-coordinate inverse CDF).

        This is what makes ``u = 0.5`` the prior median in every coordinate, which the smoke
        test relies on.
        """
        u = np.asarray(u, float)
        theta = np.empty_like(u)
        for i in range(self.dim):
            theta[i] = stats.truncnorm.ppf(
                u[i],
                a=self.a_std[i],
                b=self.b_std[i],
                loc=self.mu[i],
                scale=self.sigma[i],
            )
        return theta


class CalibrationConfigBB(CalibrationConfig):
    """``rxmc.CalibrationConfig`` adapted for black-box-bayes ("BB").

    Adds the two things the black-box-bayes CLI expects but ``rxmc`` does not provide --
    ``parameter_names`` and ``prior_transform`` -- plus a ``-inf`` robustness guard so a single
    pathological proposal is rejected rather than killing a long MPI run.

    NOTE: the class name is part of the on-disk format.  ``config.pkl`` is a dill pickle that
    references this class *by name*, so renaming it makes every already-built config
    unloadable.  Same for the module-level ``extract_low`` / ``extract_high`` below.
    """

    @property
    def parameter_names(self):
        """Flat list of sampled parameter names, model params first then nuisance."""
        param_blocks = [self.model_config] + list(self.likelihood_configs)
        return [param.name for block in param_blocks for param in block.params]

    def log_likelihood(self, x):
        try:
            value = super().log_likelihood(x)
        except Exception:
            if DEBUG:
                raise
            return -np.inf
        return value if np.isfinite(value) else -np.inf

    def log_posterior(self, x):
        try:
            value = super().log_posterior(x)
        except Exception:
            if DEBUG:
                raise
            return -np.inf
        return value if np.isfinite(value) else -np.inf

    def prior_transform(self, u):
        """Map a unit-cube vector to parameter space, block by block (dynesty's interface)."""
        u = np.asarray(u, float)
        u_blocks = np.split(u, self.indices[:-1])
        param_blocks = [self.model_config] + list(self.likelihood_configs)
        transformed = []
        for block, u_block in zip(param_blocks, u_blocks):
            prior = block.prior
            if isinstance(prior, list):
                transformed.append(
                    np.array([dist.ppf(u_i) for dist, u_i in zip(prior, u_block)])
                )
            elif hasattr(prior, "prior_transform"):
                transformed.append(np.asarray(prior.prior_transform(u_block)))
            else:
                transformed.append(np.asarray(prior.ppf(u_block)))
        return np.concatenate(transformed)


# --------------------------------------------------------------------------- #
# Parameter bookkeeping
# --------------------------------------------------------------------------- #
# Full chuq sampler order (22).
FULL_NAMES = [
    "V0",
    "Ve",
    "Vt",
    "r0",
    "r0_0",
    "a0",
    "Wv0",
    "Wve0",
    "Wvew",
    "rw",
    "rw_0",
    "aw",
    "Ws0",
    "Wst",
    "Wse0",
    "Wsew",
    "Vso",
    "rso",
    "rso_0",
    "aso",
    "rc",
    "rc_0",
]

# Frozen in BOTH variants, at the CHUQ Democratic posterior means.  These literals are checked
# against `democratic_frozen()` at import (see the assertion at the end of this section), so
# they cannot silently drift if jitr's Democratic sample set is ever updated.
_FROZEN_SHARED = {
    "Vt": 13.9342,
    "Wst": 28.8585,  # asymmetry terms (alpha=0 for Ca-40)
    "Vso": 5.5308,
    "rso": 1.3013,
    "rso_0": -1.1292,
    "aso": 0.6083,  # spin-orbit
    "r0_0": -0.1950,
    "rw_0": -0.3975,
    "rc_0": 0.1286,  # radial intercepts (single isotope)
    "rc": 1.2472,  # Coulomb radius (dXS/Ruth insensitive)
}

# The two variants: distinct absorption forms. `Wv0`/`Ws0` = 0 exactly kills the inactive imaginary
# channel.  Its energy-dependence parameters are then multiplied by zero and have no effect at
# all -- they are frozen at their Democratic means only so the 22-vector stays well formed; do
# not read anything physical into those values.
VARIANTS = {
    "low": {  # surface-only absorption (low-E band)
        "free": ["V0", "Ve", "r0", "a0", "rw", "aw", "Ws0", "Wse0", "Wsew"],
        "frozen": {**_FROZEN_SHARED, "Wv0": 0.0, "Wve0": 34.3861, "Wvew": 24.9077},
    },
    "high": {  # volume-only absorption (high-E band)
        "free": ["V0", "Ve", "r0", "a0", "Wv0", "Wve0", "Wvew", "rw", "aw"],
        "frozen": {**_FROZEN_SHARED, "Ws0": 0.0, "Wse0": 19.3620, "Wsew": 41.0751},
    },
}

# (units, LaTeX symbol, hard support bounds) for every free-able parameter.
#
# The bounds are *physical support*, not a prior: they are where the truncated normal is cut
# off, chosen wide enough to contain the CHUQ Democratic posterior with room to spare while
# excluding unphysical or numerically degenerate regions -- negative diffusenesses, radii
# outside any sane nuclear surface, absorption depths that would make the potential
# non-perturbative.  The asymmetric-looking pairs are deliberate:
#   * Wvew (1, 90) vs Wsew (1, 150) -- the surface energy scale is the broader of the two in
#     the Democratic posterior, and the low model needs the extra headroom (see PRIOR_OVERRIDES);
#   * the (1.0, ...) floors on Wvew/Wsew avoid a division by ~zero energy scale in chuq's
#     energy dependence, which is why they are 1 and not 0.
# The Wsew >= 1 floor is load-bearing: it is exactly the bound the first low-model run piled up
# against, which is what PRIOR_OVERRIDES below exists to fix.
_META = {
    "V0": ("MeV", r"V_0", (30.0, 90.0)),
    "Ve": ("", r"V_e", (-1.0, 0.5)),
    "r0": ("fm", r"r_0", (0.9, 1.5)),
    "a0": ("fm", r"a_0", (0.3, 1.0)),
    "Wv0": ("MeV", r"W_{v0}", (0.0, 40.0)),
    "Wve0": ("MeV", r"W_{ve0}", (-80.0, 140.0)),
    "Wvew": ("MeV", r"W_{vew}", (1.0, 90.0)),
    "rw": ("fm", r"r_w", (0.9, 1.7)),
    "aw": ("fm", r"a_w", (0.3, 1.0)),
    "Ws0": ("MeV", r"W_{s0}", (0.0, 40.0)),
    "Wse0": ("MeV", r"W_{se0}", (-80.0, 140.0)),
    "Wsew": ("MeV", r"W_{sew}", (1.0, 150.0)),
}

# --------------------------------------------------------------------------- #
# The model-defect nuisance parameter
# --------------------------------------------------------------------------- #
# `rxmc`'s UnknownModelError adds a fractional model error f to every prediction:
# var_total = var_stat + (f * y_model)^2.  The sampled parameter is gamma = log f.
#
# Prior: log f ~ N(log 0.05, 0.3).  Median 5% model error, and +/-1 sigma spans roughly
# 3.7%-6.8% -- i.e. the prior asserts that a global optical potential reproduces elastic
# dXS/dRuth to a few percent, which is what the CH89-family literature reports, while being
# loose enough that the data can move it by a factor of ~2 either way.
NUISANCE_PRIOR = stats.norm(loc=np.log(0.05), scale=0.3)

# Width of the initial-walker proposal on gamma, in log units.  Small on purpose: the walkers
# start clustered at the MLE value of gamma and let the sampler expand from there.
NUISANCE_INIT_SCALE = 0.01

# Hard bounds on gamma = log f: model error between 0.1% and 100%.  The lower bound keeps the
# sampler from chasing f -> 0 (where the likelihood is dominated by statistical error alone and
# the gradient vanishes); the upper bound says a model error above 100% is not a fit at all.
NUISANCE_BOUNDS = (np.log(1e-3), np.log(1.0))

# --------------------------------------------------------------------------- #
# Optimizer / seeding defaults
# --------------------------------------------------------------------------- #
# The MLE is only a *starting point* for the sampler, never a reported result, so these are
# chosen for robustness rather than precision.
MLE_METHOD = "L-BFGS-B"
MLE_N_STARTS = 4  # 1 start at the prior mean + 3 random, to dodge a bad local basin
MLE_MAXITER = 500
MLE_SEED = 0  # seed for the random restarts

# A parameter counts as "at a bound" if it sits within this fraction of the bound's range.
TRUNCATION_RTOL = 1e-3

# Initial walker proposal = prior width / this.  Starting the walkers in a tight ball around
# the MLE converges faster than starting them prior-wide, without biasing the posterior (the
# initial distribution only affects burn-in, not the stationary distribution).
INIT_PROPOSAL_SHRINK = 10.0

CONFIG_SEED = 0  # seed for the prior/proposal RNGs baked into a built config


def free_names(variant):
    """Names of the free optical-model parameters of a variant, in sampler order."""
    return list(VARIANTS[variant]["free"])


def n_omp(variant):
    """Number of free optical-model parameters (excludes the nuisance parameter)."""
    return len(VARIANTS[variant]["free"])


def params_of(variant):
    """``rxmc.Parameter`` objects for a variant's free parameters (name, units, LaTeX, bounds)."""
    return [
        Parameter(name, float, _META[name][0], _META[name][1], _META[name][2])
        for name in free_names(variant)
    ]


def prior_bounds(variant):
    """``(lower, upper)`` support bounds for a variant's free parameters."""
    names = free_names(variant)
    lower = np.array([_META[name][2][0] for name in names])
    upper = np.array([_META[name][2][1] for name in names])
    return lower, upper


def _democratic_samples():
    """(208, 22) CHUQ Democratic posterior samples, columns in FULL_NAMES order."""
    return chuq.get_samples_democratic()


def democratic_frozen():
    """Recompute the shared-frozen values from the Democratic posterior means (provenance helper)."""
    means = _democratic_samples().mean(0)
    return {name: float(means[FULL_NAMES.index(name)]) for name in _FROZEN_SHARED}


def _check_frozen_values_current():
    """Fail loudly if the hardcoded frozen values no longer match jitr's Democratic posteriors.

    ``_FROZEN_SHARED`` is written out as literals so the fit is reproducible without re-deriving
    them, but that means a jitr update could silently change what "the Democratic mean" is while
    this file kept the old numbers.  Checking at import makes that impossible to miss.
    """
    recomputed = democratic_frozen()
    mismatched = {
        name: (value, recomputed[name])
        for name, value in _FROZEN_SHARED.items()
        # literals are quoted to 4 decimals, so compare at that resolution
        if abs(value - recomputed[name]) > 5e-5
    }
    if mismatched:
        raise AssertionError(
            "chuq_model._FROZEN_SHARED no longer matches chuq.get_samples_democratic().\n"
            "The installed jitr version has different Democratic posteriors than the ones this\n"
            "study was frozen against, so results would not be comparable. Mismatches "
            "(hardcoded -> current):\n"
            + "\n".join(
                f"    {name}: {old} -> {new:.4f}" for name, (old, new) in mismatched.items()
            )
        )


_check_frozen_values_current()


# Post-hoc prior overrides (variant -> {param: (mean, std)}).  The Democratic prior N(41, 20) on
# Wsew leaves the low model bimodal: the low-E window (<~20 MeV) doesn't constrain the surface-depth
# energy dependence, so ~58% of posterior mass piles into a spurious near-zero-width mode at the
# Wsew >= 1 truncation bound (physical mode: 57.6 +/- 4.9; see calibrations/chuq_low_wideWsew/).
# Tighten to N(60, 10) to select the physical mode; truncation bounds unchanged.
PRIOR_OVERRIDES = {"low": {"Wsew": (60.0, 10.0)}}


def democratic_prior(variant):
    """Prior on a variant's free params from the Democratic posteriors: (mean, std, lo, hi).

    Truncated-normal centred at the Democratic posterior mean with the Democratic posterior std,
    truncated to the physical support bounds in ``_META``.  Fixed a priori, except for the
    ``PRIOR_OVERRIDES`` above (tightened after the first run exposed an unconstrained direction).
    """
    samples = _democratic_samples()
    names = free_names(variant)
    columns = [FULL_NAMES.index(name) for name in names]
    mean = samples[:, columns].mean(0)
    std = samples[:, columns].std(0)
    for param_name, (override_mean, override_std) in PRIOR_OVERRIDES.get(
        variant, {}
    ).items():
        i = names.index(param_name)
        mean[i] = override_mean
        std[i] = override_std
    lower, upper = prior_bounds(variant)
    return mean, std, lower, upper


# --------------------------------------------------------------------------- #
# Model wiring
# --------------------------------------------------------------------------- #
def _extract(variant, workspace, *free_values):
    """Map a variant's free params (+ frozen constants) to chuq's central/coulomb/spin-orbit args."""
    variant_spec = VARIANTS[variant]
    # Guard against a config built with a different VARIANTS definition than the one now
    # imported -- e.g. a stale config.pkl after someone edited the free/frozen split.  Without
    # this, `zip` would silently truncate: if the free list SHRANK, the extra sampled value is
    # dropped and the now-frozen constant is used in its place, giving wrong physics with no
    # error anywhere.  Not an `assert`, because `python -O` would strip it.
    if len(free_values) != len(variant_spec["free"]):
        raise ValueError(
            f"chuq_model.VARIANTS['{variant}'] expects {len(variant_spec['free'])} free "
            f"parameters {variant_spec['free']}, but was called with {len(free_values)}.\n"
            "This usually means calibrations/chuq_*/config.pkl was built against a different "
            "model definition. Rebuild it: python build_configs.py --force"
        )
    param_values = dict(zip(variant_spec["free"], free_values))
    param_values.update(variant_spec["frozen"])
    target = tuple(workspace.reaction.target)  # (A, Z)
    e_lab = workspace.kinematics.Elab
    full_param_vector = [param_values[name] for name in FULL_NAMES]
    coulomb, central, spin_orbit = chuq.calculate_params(
        PROTON, target, e_lab, *full_param_vector
    )
    # ElasticDifferentialXSModel expects (central_bundle, spin_orbit); central_plus_coulomb(r, cen, coul)
    return (central, coulomb), spin_orbit


# Module-level per-variant wrappers so dill can pickle the built config by reference.
# Do not rename: `config.pkl` files on disk refer to these by name (see CalibrationConfigBB).
def extract_low(ws, *free):
    return _extract("low", ws, *free)


def extract_high(ws, *free):
    return _extract("high", ws, *free)


_EXTRACTORS = {"low": extract_low, "high": extract_high}


def build_model(variant, quantity="dXS/dRuth"):
    """Build the ``rxmc`` elastic-scattering model for one variant."""
    return ElasticDifferentialXSModel(
        quantity,
        interaction_central=chuq.central_plus_coulomb,
        interaction_spin_orbit=chuq.spin_orbit,
        calculate_interaction_from_params=_EXTRACTORS[variant],
        params=params_of(variant),
        model_name=f"chuq_split_{variant}_{quantity.replace('/', '')}",
    )


def build_evidence(observations, model, averaging=True):
    """Wrap datasets + model in an ``Evidence`` with the fractional model-error likelihood."""
    constraint = Constraint(observations, model, UnknownModelError(averaging=averaging))
    return Evidence(parametric_constraints=[constraint])


# --------------------------------------------------------------------------- #
# MLE seed + config
# --------------------------------------------------------------------------- #
def mle_initial_guess(
    evidence,
    prior_mean,
    prior_lo,
    prior_hi,
    variant,
    method=MLE_METHOD,
    n_starts=MLE_N_STARTS,
    maxiter=MLE_MAXITER,
    seed=MLE_SEED,
):
    """Maximum-likelihood parameter vector, used only to seed the sampler.

    Runs ``n_starts`` bounded optimizations -- the first from the prior mean, the rest from
    random points in the support box -- and keeps the best finite result.  Multiple starts
    matter because the likelihood surface has shallow secondary basins in the absorption
    parameters.

    Returns:
        The best ``scipy.optimize.OptimizeResult``; ``.x`` is ``[*omp_params, log_f_err]``.
        If every start failed to produce a finite objective the last result is returned, so the
        caller still gets a usable vector -- check ``check_truncation`` and the reported
        ``-log L`` before trusting it.
    """
    n_free = n_omp(variant)
    lower = np.concatenate([np.asarray(prior_lo, float), [NUISANCE_BOUNDS[0]]])
    upper = np.concatenate([np.asarray(prior_hi, float), [NUISANCE_BOUNDS[1]]])
    bounds = list(zip(lower, upper))

    def negative_log_likelihood(theta):
        try:
            log_like = evidence.log_likelihood(theta[:n_free], [np.array([theta[n_free]])])
        except Exception:
            if DEBUG:
                raise
            return np.inf
        return -log_like if np.isfinite(log_like) else np.inf

    rng = np.random.default_rng(seed)
    start_at_prior_mean = np.concatenate(
        [np.asarray(prior_mean, float), [np.log(0.05)]]
    )
    starts = [start_at_prior_mean] + [
        rng.uniform(lower, upper) for _ in range(max(0, n_starts - 1))
    ]
    best = None
    for start in starts:
        result = optimize.minimize(
            negative_log_likelihood,
            start,
            method=method,
            bounds=bounds,
            options={"maxiter": maxiter},
        )
        if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
            best = result
    return best if best is not None else result


def check_truncation(mle_x, prior_lo, prior_hi, variant, rtol=TRUNCATION_RTOL):
    """Names of parameters whose MLE sits against a support bound.

    A hit means the data want a value the support box forbids, so the posterior will pile up at
    the bound rather than being informative there -- worth knowing before spending 36 hours
    sampling.  Reported, not raised: it is diagnostic, and a hit in one parameter does not
    invalidate the rest of the fit.
    """
    names = free_names(variant)
    lower = np.asarray(prior_lo, float)
    upper = np.asarray(prior_hi, float)
    theta_omp = np.asarray(mle_x, float)[: len(names)]
    bound_range = np.maximum(upper - lower, 1e-12)
    hits = []
    for i, name in enumerate(names):
        if (theta_omp[i] - lower[i]) <= rtol * bound_range[i]:
            hits.append(f"{name}@lower({lower[i]:g})")
        elif (upper[i] - theta_omp[i]) <= rtol * bound_range[i]:
            hits.append(f"{name}@upper({upper[i]:g})")
    return hits


def walker_seed(mle_x, prior_mean, prior_lo, prior_hi, variant, rtol=TRUNCATION_RTOL):
    """MLE vector with any bound-hugging coordinate reset to the prior mean.

    Seeding walkers exactly on a truncation bound is pathological: half of every proposal falls
    outside the support and is rejected, so the chain barely moves. Backing those coordinates off
    to the prior mean costs nothing (the initial distribution does not affect the stationary one).

    Returns:
        ``(seed_vector, names_that_were_reset)``.
    """
    names = free_names(variant)
    seed_theta = np.array(mle_x, float, copy=True)
    lower = np.asarray(prior_lo, float)
    upper = np.asarray(prior_hi, float)
    prior_mean = np.asarray(prior_mean, float)
    bound_range = np.maximum(upper - lower, 1e-12)
    reset_params = []
    for i, name in enumerate(names):
        at_lower = (seed_theta[i] - lower[i]) <= rtol * bound_range[i]
        at_upper = (upper[i] - seed_theta[i]) <= rtol * bound_range[i]
        if at_lower or at_upper:
            seed_theta[i] = prior_mean[i]
            reset_params.append(name)
    return seed_theta, reset_params


def build_calibration_config(
    variant,
    evidence,
    prior_mean,
    prior_std,
    prior_lo,
    prior_hi,
    mle_x,
    likelihood_scaling=1.0,
    init_shrink=INIT_PROPOSAL_SHRINK,
    seed=CONFIG_SEED,
):
    """Assemble the picklable :class:`CalibrationConfigBB` the sampler consumes.

    Args:
        variant: ``"low"`` or ``"high"``.
        evidence: from :func:`build_evidence`, carrying the training datasets.
        prior_mean, prior_std, prior_lo, prior_hi: from :func:`democratic_prior`.
        mle_x: seed vector from :func:`mle_initial_guess`.
        likelihood_scaling: 1.0 here -- this study does no k/N likelihood rescaling.
        init_shrink: initial walker proposal width = prior std / this.
        seed: RNG seed for the prior and proposal distributions.
    """
    n_free = n_omp(variant)
    prior_mean = np.asarray(prior_mean, float)
    prior_std = np.asarray(prior_std, float)
    prior_lo = np.asarray(prior_lo, float)
    prior_hi = np.asarray(prior_hi, float)
    mle_x = np.asarray(mle_x, float)
    seed_theta, _ = walker_seed(mle_x, prior_mean, prior_lo, prior_hi, variant)

    model_prior = TruncatedNormalPrior(
        prior_mean, prior_std, prior_lo, prior_hi, seed=seed
    )
    init_prior = TruncatedNormalPrior(
        seed_theta[:n_free], prior_std / init_shrink, prior_lo, prior_hi, seed=seed + 1
    )
    model_config = ParameterConfig(
        params=params_of(variant),
        prior=model_prior.priors_as_scipy_objects(),
        initial_proposal_distribution=init_prior.priors_as_scipy_objects(),
    )
    nuisance_params = evidence.parametric_constraints[0].likelihood.params
    nuisance_config = ParameterConfig(
        params=nuisance_params,
        prior=[NUISANCE_PRIOR],
        initial_proposal_distribution=[
            stats.norm(loc=float(mle_x[n_free]), scale=NUISANCE_INIT_SCALE)
        ],
    )
    return CalibrationConfigBB(
        evidence=evidence,
        model_config=model_config,
        likelihood_configs=[nuisance_config],
        likelihood_scaling=likelihood_scaling,
    )


# Labels for the channels :func:`energy_dependent_depths` returns, in order.  Lives here rather
# than in the analysis notebook so that a change to the potential's channel structure needs no
# edit outside this module -- the notebook sizes its panel grid from `len(...)`.
DEPTH_CHANNEL_LABELS = ("V", r"$W_v$ (volume)", r"$W_s$ (surface)")


def energy_dependent_depths(theta_omp, e_lab, variant, target=(40, 20)):
    """Central potential depths ``(V, W_volume, W_surface)`` in MeV at one energy.

    The returned channels correspond one-to-one with :data:`DEPTH_CHANNEL_LABELS`.

    Evaluates chuq's energy dependence for one posterior sample, which is what the depth-vs-E
    bands in ``chuq_analysis.ipynb`` are made of.

    Args:
        theta_omp: the free optical-model parameters of one sample (nuisance column excluded).
        e_lab: incident lab energy in MeV.
        variant: ``"low"`` or ``"high"``.
        target: ``(A, Z)`` of the target; Ca-40 by default.
    """
    variant_spec = VARIANTS[variant]
    param_values = dict(zip(variant_spec["free"], np.asarray(theta_omp, float)))
    param_values.update(variant_spec["frozen"])
    full_param_vector = [param_values[name] for name in FULL_NAMES]
    _, central, _ = chuq.calculate_params(
        PROTON, tuple(target), e_lab, *full_param_vector
    )
    # chuq's central bundle is ordered (real, imaginary volume, imaginary surface).
    return central[0], central[1], central[2]


def builtin_prediction(obs, quantity="dXS/dRuth", visualize=False):
    """dXS/dRuth (or Ay) for one obs from the published CH89 default chuq parameters (benchmark)."""
    target = tuple(obs.reaction.target)
    e_lab = obs.constraint_workspace.kinematics.Elab
    coulomb, central, spin_orbit = chuq.Global().get_params(PROTON, target, e_lab)
    workspace = obs.visualization_workspace if visualize else obs.constraint_workspace
    cross_section = workspace.xs(
        chuq.central_plus_coulomb,
        chuq.spin_orbit,
        args_central=(central, coulomb),
        args_spin_orbit=spin_orbit,
    )
    return (
        cross_section.Ay
        if quantity == "Ay"
        else cross_section.dsdo / workspace.rutherford
    )
