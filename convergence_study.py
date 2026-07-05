
import argparse
import sys
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

import foam_pipeline as fp


def parse_args():
    p = argparse.ArgumentParser(description="Discretization convergence study")
    p.add_argument("--airfoil", default="2412",
                    help="Fixed NACA 4-digit code held constant across the sweep (default: 2412)")
    p.add_argument("--points", type=int, nargs="+", default=[10, 20, 40, 80, 160, 320],
                    help="Surface discretization point counts to sweep (default: 10 20 40 80 160 320)")
    p.add_argument("--end-time", type=float, default=0.6,
                    help="Simulation end time in seconds, same for every case in this sweep (default: 0.6)")
    p.add_argument("--u-inf", type=float, default=30.0, help="Freestream velocity, m/s (default: 30.0)")
    p.add_argument("--rho-inf", type=float, default=1.225, help="Freestream density, kg/m^3 (default: 1.225)")
    p.add_argument("--nu", type=float, default=1.5e-5, help="Kinematic viscosity, m^2/s (default: 1.5e-5)")
    p.add_argument("--output-dir", default=".",
                    help="Root directory to run in / mount into Docker (default: current directory)")
    p.add_argument("--docker-image", default=fp.OPENFOAM_IMAGE,
                    help=f"OpenFOAM Docker image (default: {fp.OPENFOAM_IMAGE})")
    p.add_argument("--force-rerun", action="store_true",
                    help="Ignore any cached RESULT.json and re-run every case from scratch")
    p.add_argument("--convergence-threshold", type=float, default=0.01,
                    help="Relative change in Cl/Cd between consecutive N values below which we "
                         "consider it 'converged' (default: 0.01 = 1%%)")
    return p.parse_args()


def main():
    args = parse_args()
    root_dir = Path(args.output_dir).resolve()
    root_dir.mkdir(parents=True, exist_ok=True)

    print(f"Convergence study: NACA {args.airfoil}, points = {args.points}, "
          f"end_time = {args.end_time}s")
    print(f"Root dir (Docker mount): {root_dir}\n")

    results = []
    for n in args.points:
        case_name = f"conv_N{n:04d}_{args.airfoil}"
        result = fp.run_case(
            root_dir=root_dir,
            case_name=case_name,
            airfoil_code=args.airfoil,
            n_points=n,
            end_time=args.end_time,
            u_inf=args.u_inf,
            rho_inf=args.rho_inf,
            nu=args.nu,
            docker_image=args.docker_image,
            force_rerun=args.force_rerun,
            verbose=True,
        )
        result["n_points_sweep"] = n  # keep the sweep variable explicit regardless of status
        results.append(result)

    
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(root_dir / "convergence_results.csv", index=False)
        print()

    summary_df = pd.DataFrame(results)
    completed = summary_df[summary_df["status"] == "completed"].sort_values("n_points_sweep")
    failed = summary_df[summary_df["status"] != "completed"]

    if len(failed) > 0:
        print("WARNING: the following cases did not complete successfully:")
        for _, row in failed.iterrows():
            print(f"  - N={row['n_points_sweep']}: status={row['status']}")
        print(f"  Check runs/<case_name>/logs/ for details.\n")

    if len(completed) < 2:
        print("Not enough completed cases to plot/assess convergence. Exiting.")
        sys.exit(1)

    #Plot
    fig, (ax_cl, ax_cd) = plt.subplots(1, 2, figsize=(12, 5))

    ax_cl.plot(completed["n_points_sweep"], completed["cl_mean"], "o-", color="tab:blue")
    ax_cl.set_xlabel("Number of surface discretization points")
    ax_cl.set_ylabel("Lift coefficient (Cl)")
    ax_cl.set_title(f"Cl convergence — NACA {args.airfoil}, 0° AoA")
    ax_cl.set_xscale("log")
    ax_cl.grid(True, which="both", linestyle="--", alpha=0.5)

    ax_cd.plot(completed["n_points_sweep"], completed["cd_mean"], "o-", color="tab:red")
    ax_cd.set_xlabel("Number of surface discretization points")
    ax_cd.set_ylabel("Drag coefficient (Cd)")
    ax_cd.set_title(f"Cd convergence — NACA {args.airfoil}, 0° AoA")
    ax_cd.set_xscale("log")
    ax_cd.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.tight_layout()
    plot_path = root_dir / "convergence_plot.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved convergence plot: {plot_path}")

