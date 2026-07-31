# chuq_split — low-E vs high-E CHUQ calibration, compared in a held-out middle tercile (Ca-40 p,el)

Two calibrations of the `jitr.optical_potentials.chuq` (CH89/CHUQ) optical potential with **different
absorption forms**, each trained on a different energy band, then compared where neither were
trained. `rxmc` + `black-box-bayes` + **dynesty** MPI calibration workflow.

The curated data and both posteriors are committed, so the analysis reproduces from a clone
without re-running the 36-hour sampling jobs.

## Quickstart

```bash
git clone <this repo> && cd chuq_split
uv sync                    # Python 3.11 env from uv.lock (pinned by commit SHA)
python check_env.py        # preflight: interpreter, imports, data, MPI, results

# reproduce the analysis from the committed posteriors:
jupyter lab                # run chuq_analysis.ipynb and chuq_coverage.ipynb
```

To redo the calibration itself, see [Run](#run) below. Edit [`site.env`](site.env) first — it is
the only file that should need changing on a new machine.

## The split

The Ca-40(p,el) dσ/dΩ data (`data/ca40/data/diff_xs_observations.lz`, 5.4–201 MeV, bundled) is cut
into three **equal-point-count** energy terciles at `E0, E1` (≈20, 40 MeV; computed exactly in the
code — see `chuq_io.tercile_bounds`):

- **low** model — trained on `[0, E0]`,
- **high** model — trained on `(E1, 200]`,
- **test** region — `(E0, E1]`, held out of both fits and used only for comparison.

## The model

`chuq.calculate_params` has 22 params. Fit as `dXS/dRuth` only. The two models use **different
absorption forms** (each band's constrainable physics), derived from chuq by zeroing the inactive
imaginary channel:

- **low** = **surface-only**: free `V0 Ve r0 a0 rw aw  Ws0 Wse0 Wsew` (imag volume off: `Wv0=0`,
  `Wve0,Wvew` inert);
- **high** = **volume-only**: free `V0 Ve r0 a0 Wv0 Wve0 Wvew rw aw` (imag surface off: `Ws0=0`,
  `Wse0,Wsew` inert).

(In chuq the imaginary volume and surface share the `(rw, aw)` geometry, so `rw, aw` are free in
both.)

Frozen in **both** (single isotope, spin-orbit frozen): spin-orbit `Vso, rso, rso_0, aso` (freezes
Ay); asymmetry terms `Vt, Wst` (`(N−Z)/A = 0` for N=Z Ca-40 → degenerate); radial intercepts
`r0_0, rw_0, rc_0` (degenerate with `r0, rw` for a single isotope); Coulomb radius `rc` (dXS/Ruth
nearly insensitive: `|ΔlogL| < 1` across `rc∈[1.1,1.5]`).

Frozen values and the prior (truncated normal, `mean`/`std`) come from the CHUQ **Democratic**
posteriors (`chuq.get_samples_democratic()`), fixed a priori — see `chuq_model.py`. Each model:
9 free + 1 nuisance = **10 sampled dims**. The hardcoded frozen values are checked against the
installed jitr's Democratic posteriors at import, so they cannot silently drift.

**One prior override** (`chuq_model.PRIOR_OVERRIDES`): the low model's `Wsew` is tightened from the
Democratic N(41, 20) to **N(60, 10)** (truncation [1, 150] unchanged). The first low run was bimodal
in `Wsew` — the small low-E window doesn't constrain the surface-depth energy dependence, so ~58% of
posterior mass piled into a spurious near-zero-width mode at the `Wsew ≥ 1` bound (physical mode:
57.6 ± 4.9). That original-prior run is archived in `calibrations/chuq_low_wideWsew/`.

## Files

| file | role |
|---|---|
| `chuq_io.py` | **paths, site config, shared loaders.** The only module that knows where things are; the project root is derived from its own location, so nothing hardcodes a path. Also holds the band split, the posterior loader and the predictive variance model, which the notebooks share. |
| `chuq_model.py` | model, prior (`democratic_prior()`), MLE seed, config builders, energy-dependent depths. |
| `build_configs.py` | split the data, MLE-seed each model, build `calibrations/chuq_{low,high}/config.pkl` + a `dynesty.slurm` each, write `calibrations/fit_info.json`. **Never overwrites existing results** (`--force` to rebuild); does not submit (`--submit` to submit). |
| `site.env` | **all cluster-specific settings** — SLURM account/partition/resources, MPI module and launcher. Edit this and nothing else when moving machines. |
| `dynesty.slurm.template` | the sampling job script, with placeholders filled from `site.env` by `build_configs.py`. |
| `submit.sh` | submits any job with the settings from `site.env` (`./submit.sh sample low`, `analysis`, `jupyter`). |
| `setup_mpi.sh` | rebuild mpi4py against whatever MPI this machine provides. |
| `check_env.py` | preflight diagnostics — run this first on a new machine. |
| `smoke_test.py` | serial round-trip: finite log-posterior for each built config. |
| `chuq_fit.ipynb` | setup: data split, priors, prior-predictive, build configs. |
| `chuq_analysis.ipynb` | (after sampling) convergence, posteriors, depths vs E, held-out test comparison. |
| `chuq_coverage.ipynb` | (after sampling) posterior-predictive intervals + empirical PIT coverage in all three bands. |
| `chuq_reaction_xs.ipynb` | (after sampling) angle-integrated reaction cross section vs E over 0.01–200 MeV, both models, with each training region shaded. |
| `data/ca40/` | the curated dataset, plus its provenance — see [`data/ca40/README.md`](data/ca40/README.md). |
| `figures/` | every figure the notebooks produce. |
| `run_analysis.slurm`, `run_jupyter.slurm`, `pyproject.toml`, `uv.lock`, `.python-version` | environment and job plumbing. |

## What is committed, and what is regenerated

Committed per calibration (`calibrations/chuq_{low,high,low_wideWsew}/`):

| file | size | why |
|---|---|---|
| `dynesty_idata.nc` | ~4 MB | **the posterior samples** — everything the analysis reads |
| `dynesty_results.npz` | ~11 MB | evidence trajectory and sampler diagnostics |
| `dynesty-*.SLURMout` | ~18 KB | the actual run log, as provenance |
| `../fit_info.json` | 3 KB | split bounds, priors, MLE seeds |

Deliberately **not** committed (see `.gitignore`):

- `config.pkl` (~42 MB) — regenerate with `python build_configs.py`;
- `dynesty_checkpoint.pkl` (~55 MB) — sampler resume state, no scientific content;
- `dynesty_history.h5` (~125 MB) — raw live-point history; it exceeds GitHub's 100 MiB hard file
  limit, and the posteriors in `dynesty_idata.nc` supersede it;
- `dynesty.slurm` — generated from the template + `site.env`, and machine-specific.

Total repository payload is ~70 MB, so **no git-lfs is required**.

Notebooks are committed **with their outputs**: they are the published result, and a reader should
see the figures without running anything. `figures/` holds the same plots as standalone PNGs.

## Run

```bash
uv sync
bash setup_mpi.sh                 # once per machine (see MPI below)

python check_env.py               # confirm the environment is sane
python build_configs.py           # build both configs + launchers (does not submit, does not overwrite)
python smoke_test.py              # serial sanity check

./submit.sh sample low            # ~36 h on 65 ranks
./submit.sh sample high

./submit.sh analysis chuq_analysis.ipynb   # execute headlessly, in place
./submit.sh analysis chuq_coverage.ipynb
./submit.sh analysis chuq_reaction_xs.ipynb
./submit.sh jupyter                        # or view interactively
```

Dynesty settings: dynamic, `--nlive 1000 --dynesty-bound multi --dynesty-sample rwalk`, checkpoint
every 1800 s, auto-resume from a checkpoint if one is present. Change them in
`dynesty.slurm.template`, then re-run `python build_configs.py --force`.

## Porting to another system

Everything site-specific lives in **`site.env`**. Nothing else needs editing.

```sh
ACCOUNT=frib-nodes                     # empty => no -A flag is passed
PARTITION=general                      # short jobs (notebooks, Jupyter)
PARTITION_LONG=general-long            # the 36 h sampling jobs
NTASKS=65                              # 1 schwimmbad master + 64 likelihood workers
MEM_PER_CPU=6G
WALLTIME=36:00:00
MPI_MODULE=OpenMPI/4.1.5-GCC-12.3.0    # empty => use whatever MPI is on PATH
MPI_LAUNCHER="mpiexec -n $SLURM_NTASKS"
```

**MPI.** `uv sync` installs the PyPI mpi4py wheel, which bundles its own MPICH — fine for the
notebooks, the smoke test and any serial run, but it cannot talk to a resource manager's MPI, so
the sampling jobs would hang. `setup_mpi.sh` rebuilds mpi4py from source against the local MPI:

- with an environment-module system, set `MPI_MODULE` to your site's module;
- without one, leave `MPI_MODULE` empty and make sure `mpicc` is on `PATH` (`apt install
  libopenmpi-dev`, `brew install open-mpi`, `conda install -c conda-forge openmpi`);
- if your site's MPI is MPICH-ABI compatible, skip `setup_mpi.sh` entirely.

The build itself is implementation-agnostic — OpenMPI, MPICH, Intel MPI and MVAPICH all work.

**Launcher.** `mpiexec` suits OpenMPI+PMIx under SLURM. Many sites need `srun` instead:

```sh
MPI_LAUNCHER="srun --mpi=pmi2"
```

`black-box-bayes --require-mpi` needs a working `libmpi`, `schwimmbad`, and **at least 2 ranks**
(one master, ≥1 worker); it fails fast otherwise.

**No SLURM at all?** The sampler is an ordinary MPI program. Take the `black-box-bayes` command
from `calibrations/chuq_low/dynesty.slurm` and run it under your own launcher. The notebooks and
`smoke_test.py` need no scheduler.

**Different data location.** Set `$CHUQ_SPLIT_DATA` to a `diff_xs_observations.lz` elsewhere and
nothing else changes.

## Data provenance

The curated dataset is bundled and versioned — it is the reproducibility boundary. Regenerating it
requires the EXFOR masterfile at tag `X4-2025-12-31` (~2.2 GB, not redistributable here) and the
`X43I_DATAPATH` environment variable. Full details, including why the public fallback EXFOR
database would give a *different* dataset, are in [`data/ca40/README.md`](data/ca40/README.md).

⚠️ **`X43I_DATAPATH`** — the one environment variable that can bite you. `rxmc` imports
`exfor_tools`, which imports `x4i3`, which resolves an EXFOR database **at import time**. Nothing
in this study queries EXFOR, but the import chain forces the check anyway, and all three outcomes
matter:

| `X43I_DATAPATH` | what happens on `import chuq_model` |
|---|---|
| points at an unpacked EXFOR dir containing an `X4-20*` tag file | works, no download; a banner naming that directory is printed (it appears in the notebooks' stored output) |
| **unset** | x4i3 decides its bundled database is missing and **downloads a ~100 MB EXFOR tarball** into `site-packages` — one-off and then cached, but a surprise inside a batch job |
| set to a path that does not exist, or with no tag file | `FileNotFoundError` at import — **every** script, notebook and MPI worker rank fails |

`python check_env.py` reports which case you are in. Do not "clean up" by unsetting it in a job
script: that turns every job into a network fetch.

## License

MIT — see [LICENSE](LICENSE).
