"""Serial (no-MPI) sanity check: each built config round-trips a finite log-posterior.

    python smoke_test.py

Loads ``calibrations/chuq_{low,high}/config.pkl``, evaluates it at the prior median, and checks
that the log-likelihood and log-posterior are finite.  This mirrors what a
``black-box-bayes --no-mpi`` round-trip would do, without needing MPI -- so it catches a broken
model, a broken prior, or an unpicklable config before a 65-rank job does.

Run it after ``python build_configs.py``.

Note the check is only meaningful because ``CalibrationConfigBB`` converts exceptions into
``-inf``: a model that raises shows up here as non-finite, i.e. as a FAIL, rather than as a
traceback.  Set ``CHUQ_DEBUG=1`` to see the underlying exception instead.
"""

import sys

import numpy as np
import dill

import chuq_io
import chuq_model  # noqa: F401  (needed so dill can resolve the pickled config's references)

# The prior median in every coordinate: prior_transform is a per-coordinate inverse CDF, so the
# centre of the unit cube maps to the median of each marginal prior.  A generic interior point
# would do, but this one is reproducible and never near a truncation bound.
UNIT_CUBE_CENTRE = 0.5


def check_matches_current_model(variant, config):
    """Confirm the pickled config was built from the model definition now in chuq_model.

    A `config.pkl` stores its parameter block, but calls back into the live
    ``chuq_model.VARIANTS`` at evaluation time.  If someone edits the free/frozen split and does
    not rebuild, those two disagree -- and the failure is not always loud, so check it directly
    rather than relying on a wrong answer to look wrong.  Uses the parameter names already in
    the config, so it works on configs built before this check existed.
    """
    expected = chuq_model.free_names(variant)
    try:
        stored = [param.name for param in config.model_config.params]
    except AttributeError:
        print(f"       (config exposes no model_config.params; skipping model-match check)")
        return True
    if stored == expected:
        return True
    print(
        f"       MODEL MISMATCH: config.pkl was built with {stored}\n"
        f"                       chuq_model.VARIANTS['{variant}'] now says {expected}\n"
        f"                       Rebuild it: python build_configs.py --force --only {variant}"
    )
    return False


def check(variant):
    """Evaluate one built config at the prior median; return True if it is usable."""
    config_path = chuq_io.run_dir(variant) / "config.pkl"
    if not config_path.exists():
        print(f"[{variant}] {config_path} not found -- run `python build_configs.py` first")
        return False
    with open(config_path, "rb") as stream:
        config = dill.load(stream)

    n_dims = len(config.parameter_names)
    theta = config.prior_transform(np.full(n_dims, UNIT_CUBE_CENTRE))
    log_prior = (
        config.log_prior(theta) if hasattr(config, "log_prior") else float("nan")
    )
    log_likelihood = config.log_likelihood(theta)
    log_posterior = config.log_posterior(theta)

    # log_prior is reported but not asserted on: it is absent on some rxmc config versions, and
    # a finite log_posterior already implies a finite prior contribution.
    finite = np.isfinite(log_likelihood) and np.isfinite(log_posterior)
    print(f"[{variant}] dim={n_dims}  params={config.parameter_names}")
    print(f"       theta(prior median)={np.round(theta, 3)}")
    print(
        f"       log_prior={log_prior:.2f}  log_likelihood={log_likelihood:.2f}  "
        f"log_posterior={log_posterior:.2f}  -> {'OK' if finite else 'FAIL'}"
    )
    # Checked after the numbers are printed so a mismatch is reported alongside them.
    return finite and check_matches_current_model(variant, config)


if __name__ == "__main__":
    all_passed = all(check(variant) for variant in chuq_io.VARIANT_NAMES)
    print("\nSMOKE TEST:", "PASS" if all_passed else "FAIL")
    sys.exit(0 if all_passed else 1)
