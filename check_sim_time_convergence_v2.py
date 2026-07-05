import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Pinpoint the simulation-time convergence point for one case")
    p.add_argument("--case", required=True, help="Path to the case directory, e.g. runs/conv_N0300_2412")
    p.add_argument("--tolerance-pct", type=float, default=1.0,
                    help="Permanent-deviation tolerance from the final value, in percent "
                         "(default: 1.0, matching the literature-standard GCI-adjacent tolerance)")
    return p.parse_args()


def find_t_conv(time, values, tolerance_pct):
    """Return the earliest time such that all subsequent values deviate from the
    final value by less than tolerance_pct percent. Returns None if never satisfied."""
    final_val = values.iloc[-1]
    if abs(final_val) < 1e-12:
        return None
    dev_pct = np.abs(values - final_val) / abs(final_val) * 100
    n = len(values)
    # scan backwards: find the first index i such that dev_pct[i:] is ALL below tolerance
    for i in range(n):
        if (dev_pct.iloc[i:] < tolerance_pct).all():
            return time.iloc[i], dev_pct
    return None, dev_pct


def main():
    args = parse_args()
    case_dir = Path(args.case)

    force_path = case_dir / "force_coeffs_timeseries.csv"
    resid_path = case_dir / "residuals_timeseries.csv"

    if not force_path.exists():
        raise SystemExit(f"Not found: {force_path}\n"
                          f"(Only cases with status='completed' write this file - check RESULT.json.)")

    force_df = pd.read_csv(force_path)
    resid_df = pd.read_csv(resid_path) if resid_path.exists() else None

    t_conv_cl, dev_cl = find_t_conv(force_df["time"], force_df["cl"], args.tolerance_pct)
    t_conv_cd, dev_cd = find_t_conv(force_df["time"], force_df["cd"], args.tolerance_pct)

    print(f"Case: {case_dir.name}")
    print(f"Tolerance: permanently within {args.tolerance_pct}% of final value")
    print(f"Total time range: {force_df['time'].iloc[0]:.4f}s to {force_df['time'].iloc[-1]:.4f}s "
          f"({len(force_df)} saved timesteps)")
    print()

    if t_conv_cl is not None:
        print(f"Cl: final value = {force_df['cl'].iloc[-1]:.5f}, "
              f"t_conv = {t_conv_cl:.4f}s (permanently within {args.tolerance_pct}% from this point on)")
    else:
        print(f"Cl: never permanently settles within {args.tolerance_pct}% over the tested duration - "
              f"consider a longer run.")

    if t_conv_cd is not None:
        print(f"Cd: final value = {force_df['cd'].iloc[-1]:.5f}, "
              f"t_conv = {t_conv_cd:.4f}s (permanently within {args.tolerance_pct}% from this point on)")
    else:
        print(f"Cd: never permanently settles within {args.tolerance_pct}% over the tested duration - "
              f"consider a longer run.")

    if t_conv_cl is not None and t_conv_cd is not None:
        t_conv_combined = max(t_conv_cl, t_conv_cd)
        print()
        print(f"==> Combined recommendation: end_time should be at least {t_conv_combined:.4f}s "
              f"(the later of the two individual convergence times).")

    # --- Plot ---
    fig, (ax_force, ax_resid) = plt.subplots(1, 2, figsize=(14, 5))

    ax_force.plot(force_df["time"], force_df["cl"], label="Cl", color="tab:blue")
    ax_force.plot(force_df["time"], force_df["cd"], label="Cd", color="tab:red")
    if t_conv_cl is not None:
        ax_force.axvline(t_conv_cl, color="tab:blue", linestyle="--", alpha=0.6, label=f"Cl t_conv={t_conv_cl:.3f}s")
    if t_conv_cd is not None:
        ax_force.axvline(t_conv_cd, color="tab:red", linestyle="--", alpha=0.6, label=f"Cd t_conv={t_conv_cd:.3f}s")
    ax_force.set_xlabel("Time (s)")
    ax_force.set_ylabel("Coefficient value")
    ax_force.set_title(f"Cl/Cd vs time - {case_dir.name}")
    ax_force.grid(True, linestyle="--", alpha=0.5)
    ax_force.legend(fontsize=8)

    if resid_df is not None:
        ax_resid.plot(resid_df["time"], resid_df["r_ux"], label="Ux", color="tab:blue", alpha=0.8)
        ax_resid.plot(resid_df["time"], resid_df["r_uy"], label="Uy", color="tab:green", alpha=0.8)
        ax_resid.plot(resid_df["time"], resid_df["r_p"], label="p", color="black", alpha=0.8)
        ax_resid.set_yscale("log")
        ax_resid.set_xlabel("Time (s)")
        ax_resid.set_ylabel("Residual (log scale)")
        ax_resid.set_title(f"Residuals vs time - {case_dir.name}")
        ax_resid.grid(True, which="both", linestyle="--", alpha=0.5)
        ax_resid.legend(fontsize=8)
    else:
        ax_resid.set_title("No residuals_timeseries.csv found")

    fig.tight_layout()
    out_path = case_dir / "sim_time_convergence.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")

