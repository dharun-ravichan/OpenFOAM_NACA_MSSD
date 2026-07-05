# NACA Airfoil Camber Study — OpenFOAM Automation

Automated Python/OpenFOAM 13 pipeline for a NACA 4-digit airfoil parameter study:
(1) a discretization convergence study, and (2) a camber sweep quantifying the
effect of camber on Cl/Cd. Built for RWTH Aachen MSSD Project 2.

## Requirements

- Python 3.10+ with `numpy`, `pandas`, `matplotlib`
- [Docker Desktop](https://www.docker.com/products/docker-desktop/), running,
  with the `microfluidica/openfoam:13` image available
  (`docker pull microfluidica/openfoam:13`)

## Files

| File | Purpose |
|---|---|
| `foam_pipeline.py` | Shared library: NACA geometry generation, the mesher, OpenFOAM dictionary writers, and `run_case()` — runs one case end-to-end (mesh → `blockMesh` → `checkMesh` → `foamRun` → parse results) via Docker. |
| `convergence_study.py` | Sweeps surface point count N at fixed camber, produces `convergence_results.csv`. |
| `camber_study.py` | Sweeps camber M = 0-8 at fixed N, produces `camber_results.csv`. |
| `convergence_plot.py` | Generates the discretization convergence figures (`cl_cd_vs_N.png`, used as Figure 1 in the report) from cached `RESULT.json` files — no Docker, no re-running. |
| `check_sim_time_convergence_v2.py` | Scans a completed case's saved time series to find the simulation-time convergence point (used to produce Figure 2 in the report). |


## Running it

**1. Discretization convergence study** (sweeps N, fixed camber NACA 2412):
```bash
python convergence_study.py
```
Runs every case in `runs/conv_N####_2412/`, writes `convergence_results.csv`
and `convergence_plot.png`. Safe to interrupt and rerun — completed cases are
cached and skipped automatically (pass `--force-rerun` to redo everything).

**2. Simulation-time convergence check** (no new simulation, reads cached results):
```bash
python check_sim_time_convergence_v2.py --case runs/conv_N0300_2412 --tolerance-pct 0.01
```

**3. Camber parameter study** (sweeps M = 0-8, fixed N = 300):
```bash
python camber_study.py --n-points 300
```
Runs every case in `runs/camber_M#_*/`, writes `camber_results.csv` and
`camber_plot.png`.


## Notes on parameters

- Freestream: U = 30 m/s, ρ = 1.225 kg/m³, ν = 1.5e-5 m²/s (Re ≈ 2×10⁶)
- Selected discretization: N = 300 (cost/accuracy trade-off; see report Section 3)
- Selected simulation length: `end_time` = 0.6 s (validated in report Section 4)
- A handful of specific (N, camber) combinations fail `checkMesh` due to a
  mesher limitation at sparse leading-edge discretization — see report
  Section 6. This is expected and handled gracefully (cases are marked
  `mesh_failed` and skipped, not treated as errors).
