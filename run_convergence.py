#!/usr/bin/env python3
"""运行收敛性测试 v2"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from fdtd_2d_tm import convergence_test, plot_convergence
import numpy as np

FIG_DIR = "/mnt/c/Users/admin/Desktop/1号 Hermes 课题/09_FDTD电磁波数值仿真教学/03_论文输出/figures"
os.makedirs(FIG_DIR, exist_ok=True)

dxs, errs, p, r2 = convergence_test()
print(f"\n=== 结果 ===")
print(f"收敛阶 p = {p:.4f}")
print(f"R² = {r2:.4f}")

np.savetxt(f"{FIG_DIR}/convergence_data.csv",
           np.column_stack([dxs, errs]),
           header="dx(m),rel_L2_error", delimiter=",")
print(f"数据保存至: {FIG_DIR}/convergence_data.csv")

plot_convergence(dxs, errs, p, r2, f"{FIG_DIR}/fig_convergence.png")
print(f"图保存至: {FIG_DIR}/fig_convergence.png")
