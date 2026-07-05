import argparse
import sys
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

import foam_pipeline as fp


def parse_args():
    p = argparse.ArgumentParser(description="Camber (M) parameter study")
    p.add_argument("--camber-values", type=int, nargs="+", default=list(range(0, 9)),
                    help="NACA camber digit M values to sweep, 0-9 (default: 0 1 2 3 4 5 6 7 8)")
    p.add_argument("--p", type=int, default=4,
                    help="NACA position-of-max-camber digit P, held fixed across the sweep "
                         "(default: 4, i.e. max camber at 40%% chord)")
    p.add_argument("--tt", type=int, default=12,
                    help="NACA max-thickness two digits TT, held fixed across the sweep "
                         "(default: 12, i.e. 12%% chord thick)")
    p.add_argument("--n-points", type=int, default=125,
                    help="Surface discretization point count, fixed from the convergence "
                         "study (default: 125)")
    p.add_argument("--end-time", type=float, default=0.6,
                    help="Simulation end time in seconds, fixed from the simulation-time "
                         "convergence check (default: 0.6)")
    p.add_argument("--u-inf", type=float, default=30.0, help="Freestream velocity, m/s (default: 30.0)")
    p.add_argument("--rho-inf", type=float, default=1.225, help="Freestream density, kg/m^3 (default: 1.225)")
    p.add_argument("--nu", type=float, default=1.5e-5, help="Kinematic viscosity, m^2/s (default: 1.5e-5)")
    p.add_argument("--output-dir", default=".",
                    help="Root directory to run in / mount into Docker (default: current directory)")
    p.add_argument("--docker-image", default=fp.OPENFOAM_IMAGE,
                    help=f"OpenFOAM Docker image (default: {fp.OPENFOAM_IMAGE})")
    p.add_argument("--force-rerun", action="store_true",
                    help="Ignore any cached RESULT.json and re-run every case from scratch")
    return p.parse_args()


def naca_code(m, p, tt):
    # NACA 4-digit code: M (1 digit) P (1 digit) TT (2 digits)
    if not (0 <= m <= 9):
        raise ValueError(f"Camber digit M must be 0-9, got {m}")
    return f"{m}{p}{tt:02d}"


def main():
    args = parse_args()
    root_dir = Path(args.output_dir).resolve()
    root_dir.mkdir(parents=True, exist_ok=True)

    print(f"Camber study: P={args.p}, TT={args.tt}, N={args.n_points} points, "
          f"end_time={args.end_time}s")
    print(f"Camber values (M): {args.camber_values}")
    print(f"Root dir (Docker mount): {root_dir}\n")

    results = []
    for m in args.camber_values:
        airfoil_code = naca_code(m, args.p, args.tt)
        case_name = f"camber_M{m}_{airfoil_code}"
        result = fp.run_case(
            root_dir=root_dir,
            case_name=case_name,
            airfoil_code=airfoil_code,
            n_points=args.n_points,
            end_time=args.end_time,
            u_inf=args.u_inf,
            rho_inf=args.rho_inf,
            nu=args.nu,
            docker_image=args.docker_image,
            force_rerun=args.force_rerun,
            verbose=True,
        )
        result["camber_m"] = m  # keep the sweep variable explicit regardless of status
        result["airfoil_code"] = airfoil_code
        results.append(result)

        # Write the summary CSV after every single case
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(root_dir / "camber_results.csv", index=False)
        print()

    summary_df = pd.DataFrame(results)
    completed = summary_df[summary_df["status"] == "completed"].sort_values("camber_m")
    failed = summary_df[summary_df["status"] != "completed"]

    if len(failed) > 0:
        print("WARNING: the following cases did not complete successfully:")
        for _, row in failed.iterrows():
            print(f"  - M={row['camber_m']} ({row['airfoil_code']}): status={row['status']}")
        print(f"  Check runs/<case_name>/logs/ for details.\n")

    if len(completed) < 2:
        print("Not enough completed cases to plot. Exiting.")
        sys.exit(1)

    # --- Plot ---
    fig, (ax_cl, ax_cd) = plt.subplots(1, 2, figsize=(12, 5))

    ax_cl.plot(completed["camber_m"], completed["cl_mean"], "o-", color="tab:blue")
    ax_cl.set_xlabel("Camber (M, % chord)")
    ax_cl.set_ylabel("Lift coefficient (Cl)")
    ax_cl.set_title(f"Cl vs camber - NACA M{args.p}{args.tt:02d}, 0° AoA, N={args.n_points}")
    ax_cl.grid(True, linestyle="--", alpha=0.5)

    ax_cd.plot(completed["camber_m"], completed["cd_mean"], "o-", color="tab:red")
    ax_cd.set_xlabel("Camber (M, % chord)")
    ax_cd.set_ylabel("Drag coefficient (Cd)")
    ax_cd.set_title(f"Cd vs camber - NACA M{args.p}{args.tt:02d}, 0° AoA, N={args.n_points}")
    ax_cd.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    plot_path = root_dir / "camber_plot.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved camber plot: {plot_path}")

    print("\nSummary:")
    for _, row in completed.iterrows():
        print(f"  M={row['camber_m']:>2} ({row['airfoil_code']}): "
              f"Cl={row['cl_mean']:.4f}, Cd={row['cd_mean']:.4f}")

