# Ca-40 (p, elastic) data — curated inputs

These files are the **versioned inputs** to the calibration in the parent directory. They are
committed rather than regenerated: everything downstream of them runs anywhere, but producing
them requires a ~2.2 GB EXFOR masterfile that cannot be redistributed here.

## Files

| file | what it is |
|---|---|
| `data/diff_xs_observations.lz` | **the file the calibration uses.** `dill`-pickled, LZMA-compressed list of `rxmc` elastic dσ/dΩ observation objects, Ca-40(p,p) 5.4–201 MeV, with compound-nucleus corrections applied |
| `data/ay_observations.lz` | analyzing-power observations (not used by `chuq_split`; used by the spin-orbit variant of this study) |
| `data/obs_list.lz`, `data/obslist_low.lz`, `data/obslist_high.lz` | intermediate curation products kept for provenance |
| `data/talys_combined_data.csv` | TALYS compound-nucleus cross sections used for the low-energy correction |
| `curate_data.ipynb` | EXFOR query → filtering → the `.lz` files above |
| `compound_corrections.ipynb` | derivation of the TALYS compound correction |

## The reproducibility boundary

`diff_xs_observations.lz` is where reproducibility starts for an outside user. **Nothing in
the calibration or analysis touches EXFOR** — the `.lz` file is a self-contained pickle. You
only need what is below if you want to re-derive the curated dataset itself.

Re-running `curate_data.ipynb` requires:

1. **The EXFOR database at tag `X4-2025-12-31`**, unpacked with
   [`x4i3_tools`](https://github.com/afedynitch/x4i3) from the ~2.2 GB `exfor-2025.txt`
   masterfile, and pointed to by the environment variable `X43I_DATAPATH`:

   ```bash
   export X43I_DATAPATH=/path/to/unpack_exfor-2025/X4-2025-12-31
   ```

   Note the spelling: `X43I_DATAPATH`, not `X4I3_DATAPATH`.

   **The public fallback database is a different vintage.** If `X43I_DATAPATH` is unset,
   `x4i3` falls back to a bundled path and, failing that, the release
   `x4i3_X4-2023-04-29.tar.gz`. Curating against 2023-04-29 instead of 2025-12-31 yields a
   *different dataset* — more entries were added and some corrected in between.

2. `data/talys_combined_data.csv` (committed here), and a working directory of `data/ca40/`
   — `curate_data.ipynb` writes to the relative path `./data`.

Package versions used to produce these files:

```
exfor_tools  1.2   (git 1.2.dev4+gaf6261e74)
jitr         2.6
rxmc         0.1.dev298+g1dffe7e9d
EXFOR tag    X4-2025-12-31
```

## A warning about `X43I_DATAPATH`

`x4i3` resolves its database **at import time**, and `rxmc` imports `exfor_tools`, which imports
`x4i3`. The consequences reach well beyond the curation notebooks:

- **set to a valid unpacked EXFOR directory** (one containing an `X4-20*` tag file) — imports
  cleanly, no download. A banner naming that directory is printed on every import, which is why
  it shows up in the notebooks' stored output.
- **unset** — `x4i3` concludes its bundled database is incomplete and **downloads the
  `x4i3_X4-2023-04-29.tar.gz` archive (~100 MB) into `site-packages`** before the import returns.
  It is a one-off (the files are then cached in the venv) and it does make a fresh checkout work
  unattended, but it is a surprise if it happens inside a batch job. The only suppression x4i3
  offers is `"pytest" in sys.modules`.
- **set to a path that does not exist, or to one with no `X4-20*` tag file** — `x4i3` raises
  `FileNotFoundError` during `import chuq_model`, so *every* script, notebook and MPI worker rank
  in this repo dies on import even though none of them needs EXFOR.

So: point it at a real unpacked EXFOR directory if you have one, otherwise leave it unset and
accept the one-time download. **Do not unset it inside a job script** to silence the banner —
that converts every job into a network fetch. `python check_env.py` in the parent directory
reports which of the three cases you are in.
