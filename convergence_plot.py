import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot convergence study results from cached RESULT.json files")
    p.add_argument("--root-dir", default=".", help="Root directory containing runs/ (default: current directory)")
    p.add_argument("--airfoil", default="2412", help="Airfoil code used in the convergence sweep (default: 2412)")
    p.add_argument("--selected-n", type=int, default=None,
                    help="Optionally mark a chosen production N with a vertical line on the plots")
    return p.parse_args()


def load_results(root_dir: Path, airfoil: str) -> pd.DataFrame:
    rows = []
    pattern = f"conv_N*_{airfoil}"
    for case_dir in sorted(root_dir.glob(f"runs/{pattern}")):
        result_path = case_dir / "RESULT.json"
        if not result_path.exists():
            continue
        r = json.loads(result_path.read_text())
        # extract N from directory name: conv_N0125_2412 -> 125
        try:
            n = int(case_dir.name.split("_N")[1].split("_")[0])
        except (IndexError, ValueError):
            continue
        row = {"N": n, "status": r.get("status", "unknown")}
        if r.get("status") == "completed":
            row["cl_mean"] = r["cl_mean"]
            row["cd_mean"] = r["cd_mean"]
        rows.append(row)
    if not rows:
        raise SystemExit(f"No cached results found under {root_dir}/runs/{pattern}/RESULT.json")
    df = pd.DataFrame(rows).sort_values("N").reset_index(drop=True)
    return df


def main():
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    df = load_results(root_dir, args.airfoil)

    completed = df[df["status"] == "completed"].copy()
    failed = df[df["status"] != "completed"]

    print(f"Loaded {len(df)} cases ({len(completed)} completed, {len(failed)} failed) for NACA {args.airfoil}")
    if len(failed) > 0:
        print("Failed N values:", ", ".join(str(n) for n in failed["N"]))
    print()
    print(completed[["N", "cl_mean", "cd_mean"]].to_string(index=False))

    # Plot 1: Cl / Cd vs N
    fig1, (ax_cl, ax_cd) = plt.subplots(1, 2, figsize=(13, 5))

    ax_cl.plot(completed["N"], completed["cl_mean"], "o-", color="tab:blue", markersize=4)
    ax_cl.set_xscale("log")
    ax_cl.set_xlabel("Number of surface discretization points N")
    ax_cl.set_ylabel("Lift coefficient (Cl)")
    ax_cl.set_title(f"Cl vs N \u2014 NACA {args.airfoil}")
    ax_cl.grid(True, which="both", linestyle="--", alpha=0.5)

    ax_cd.plot(completed["N"], completed["cd_mean"], "o-", color="tab:red", markersize=4)
    ax_cd.set_xscale("log")
    ax_cd.set_xlabel("Number of surface discretization points N")
    ax_cd.set_ylabel("Drag coefficient (Cd)")
    ax_cd.set_title(f"Cd vs N \u2014 NACA {args.airfoil}")
    ax_cd.grid(True, which="both", linestyle="--", alpha=0.5)

    if args.selected_n is not None:
        for ax in (ax_cl, ax_cd):
            ax.axvline(args.selected_n, color="green", linestyle=":", linewidth=2,
                       label=f"selected N={args.selected_n}")
            ax.legend()


    if len(failed) > 0:
        y_marker = completed["cl_mean"].mean()
        ax_cl.plot(failed["N"], [y_marker] * len(failed), "rx", markersize=8,
                   label="mesh_failed")
        ax_cl.legend()

    fig1.tight_layout()
    fig1.savefig(root_dir / "cl_cd_vs_N.png", dpi=150)
    print(f"\nSaved: {root_dir / 'cl_cd_vs_N.png'}")


    # Plot 2: relative error vs N, referenced to the FINEST completed mesh
    finest_n = completed["N"].max()
    cl_ref = completed.loc[completed["N"] == finest_n, "cl_mean"].values[0]
    cd_ref = completed.loc[completed["N"] == finest_n, "cd_mean"].values[0]
    print(f"\nUsing finest mesh N={finest_n} as reference: Cl_ref={cl_ref:.5f}, Cd_ref={cd_ref:.5f}")

    rel_cl = (completed["cl_mean"] - cl_ref).abs() / abs(cl_ref) * 100
    rel_cd = (completed["cd_mean"] - cd_ref).abs() / abs(cd_ref) * 100

    fig2, ax = plt.subplots(figsize=(9, 6))
    ax.plot(completed["N"], rel_cl, "o-", color="tab:blue", markersize=4, label="Cl relative error (%)")
    ax.plot(completed["N"], rel_cd, "s-", color="tab:red", markersize=4, label="Cd relative error (%)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of surface discretization points N")
    ax.set_ylabel(f"Relative error vs. finest mesh (N={finest_n}) [%]")
    ax.set_title(f"Relative error vs N \u2014 NACA {args.airfoil}")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()

    if args.selected_n is not None:
        ax.axvline(args.selected_n, color="green", linestyle=":", linewidth=2)

    fig2.tight_layout()
    fig2.savefig(root_dir / "relative_error_vs_N.png", dpi=150)
    print(f"Saved: {root_dir / 'relative_error_vs_N.png'}")


    # Plot 3: windowed (binned) standard deviation of Cl, to visualize
    bins = [0, 80, 150, 500, np.inf]
    bin_labels = ["N<80", "80\u2264N<150", "150\u2264N<500", "N\u2265500"]
    completed["bin"] = pd.cut(completed["N"], bins=bins, labels=bin_labels, right=False)

    bin_stats = completed.groupby("bin")["cl_mean"].agg(["mean", "std", "count"]).reindex(bin_labels)
    print("\nBinned Cl scatter by N-range:")
    print(bin_stats.to_string())

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.bar(bin_labels, bin_stats["std"], color="tab:purple", alpha=0.7)
    for i, (label, row) in enumerate(bin_stats.iterrows()):
        ax3.text(i, row["std"], f"n={int(row['count'])}", ha="center", va="bottom", fontsize=9)
    ax3.set_ylabel("Std. dev. of Cl within N-range")
    ax3.set_title(f"Cl scatter by discretization range \u2014 NACA {args.airfoil}")
    ax3.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig3.tight_layout()
    fig3.savefig(root_dir / "windowed_scatter_vs_N.png", dpi=150)
    print(f"Saved: {root_dir / 'windowed_scatter_vs_N.png'}")