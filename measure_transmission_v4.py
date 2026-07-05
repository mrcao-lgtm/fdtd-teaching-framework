#!/usr/bin/env python3
"""介质板透射系数: 脉冲传输法 - 在脉冲穿过slab的瞬间测量振幅"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from fdtd_2d_tm import FDTD2DTM, c0

print("=" * 60)
print("介质板透射系数: 脉冲传输法")
print("=" * 60)

dx = 5e-3
nx, ny = 200, 150
source_x = 40
slab_s, slab_e = 100, 130

# 先算时间: 脉冲从源到slab前 + 穿过slab + 到slab后探头
dist_inc = (slab_s - 5 - source_x) * dx  # slab前5格
dist_trn = (slab_e + 10 - source_x) * dx  # slab后10格
# 在介质中波速减半
slab_width = (slab_e - slab_s) * dx
t_inc = dist_inc / c0
t_trn = (dist_inc + slab_width/2 + 10*dx) / c0  # 粗略估计

# modulated Gaussian脉冲参数
fc = 3e9
tau = 30 * dx / (c0 * 1.0)  # 30个时间步的物理宽度  
n_steps_est = int(t_trn * 2 * c0 / dx) + 100
n_steps = min(n_steps_est, 500)

# 有slab仿真, 单个探头在slab后
print(f"\n[1] 有slab仿真 (n_steps={n_steps})...")
f = FDTD2DTM(nx, ny, dx)
f.set_source(source_x, 75)
f.add_dielectric_slab(slab_s, slab_e, 4.0)
f._init_coeffs()
r_slb = f.run(n_steps, 'modulated', save_interval=5, progress=True, probe_pos=slab_e+10)
probe_trn = r_slb['probe_Ez']  # slab后探针

# 无slab参考 - 同探头位置
print(f"\n[2] 无slab参考仿真...")
f_ref = FDTD2DTM(nx, ny, dx)
f_ref.set_source(source_x, 75)
r_ref = f_ref.run(n_steps, 'modulated', 5, progress=True, probe_pos=slab_e+10)
probe_ref = r_ref['probe_Ez']

# 找脉冲包络峰值
t_vals = np.arange(n_steps) * f.dt * 1e9  # ns

# Hilbert包络
from scipy import signal
hilb_trn = signal.hilbert(probe_trn)
hilb_ref = signal.hilbert(probe_ref)
env_trn = np.abs(hilb_trn)
env_ref = np.abs(hilb_ref)

# 找峰值（跳过前50步瞬态）
search_start = 50
idx_trn = np.argmax(env_trn[search_start:]) + search_start
idx_ref = np.argmax(env_ref[search_start:]) + search_start

amp_trn = env_trn[idx_trn]
amp_ref = env_ref[idx_ref]
T_meas = amp_trn / amp_ref if amp_ref > 0 else 0

print(f"\n{'='*60}")
print("结果汇总")
print(f"{'='*60}")
print(f"  无slab参考峰值: {amp_ref:.6f} @ t={t_vals[idx_ref]:.2f}ns")
print(f"  有slab透射峰值: {amp_trn:.6f} @ t={t_vals[idx_trn]:.2f}ns")
print(f"  FDTD实测 T = {T_meas:.4f}")
print(f"  单界面Fresnel理论 T = {2/3:.4f} = 2/3")
print(f"  差异: {(T_meas - 2/3)/(2/3)*100:+.1f}%")
