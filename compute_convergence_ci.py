#!/usr/bin/env python3
"""Compute convergence order p with 95% CI from log-log regression."""
import numpy as np
import sys

# Read convergence data
data_path = "/mnt/c/Users/admin/Desktop/1号 Hermes 课题/09_FDTD电磁波数值仿真教学/03_论文输出/figures/convergence_data.csv"
raw = np.loadtxt(data_path, delimiter=",", skiprows=1)
dxs = raw[:, 0]
errs = raw[:, 1]

# Exclude zero-error reference point (last row, nx=400, err=0)
non_zero = errs > 1e-15
dx_fit = dxs[non_zero]
err_fit = errs[non_zero]

n_points = len(dx_fit)
print(f"Fitting points: {n_points}")
print(f"dx values: {dx_fit}")
print(f"err values: {err_fit}")

log_dx = np.log(dx_fit)
log_err = np.log(err_fit)

# Linear regression: log(err) = p * log(dx) + log(C)
A = np.vstack([log_dx, np.ones_like(log_dx)]).T
coeffs, residuals, rank, s = np.linalg.lstsq(A, log_err, rcond=None)
p = coeffs[0]
log_C = coeffs[1]

# Residuals
residuals = log_err - (p * log_dx + log_C)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((log_err - np.mean(log_err))**2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

# Standard error of p (using Student's t distribution)
# MSE = sum(residuals^2) / (n - 2)
mse = ss_res / (n_points - 2)

# Variance of slope: MSE / sum((x_i - mean_x)^2)
x_mean = np.mean(log_dx)
x_var = np.sum((log_dx - x_mean)**2)
se_p = np.sqrt(mse / x_var)

# 95% CI using Student's t
from scipy import stats
t_crit = stats.t.ppf(0.975, n_points - 2)
ci_low = p - t_crit * se_p
ci_high = p + t_crit * se_p

print(f"\n=== Results ===")
print(f"p = {p:.4f}")
print(f"SE(p) = {se_p:.4f}")
print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"R² = {r2:.4f}")
print(f"n = {n_points}")

# Also compute without scipy (fallback)
# For 95% CI with n-2=5 df, t_0.975 ≈ 2.571
t_manual = 2.571  # t_{0.975}(5) = 2.571
ci_low2 = p - t_manual * se_p
ci_high2 = p + t_manual * se_p
print(f"\nManual (t=2.571): p = {p:.4f} ± {t_manual*se_p:.4f}")
print(f"  CI: [{ci_low2:.4f}, {ci_high2:.4f}]")

# Print the exact patch text
print(f"\n=== PATCH TEXT ===")
print(f"p = {p:.2f} ± {t_manual*se_p:.2f} (95% CI: [{ci_low2:.2f}, {ci_high2:.2f}], $R^2 = {r2:.3f}$)")
