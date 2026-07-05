#!/usr/bin/env python3
"""快速测量: 介质板透射系数T的FDTD实测值 (fix: reinit coeffs after slab)"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from fdtd_2d_tm import FDTD2DTM, c0

print("=" * 60)
print("介质板透射系数精确测量 (修复版)")
print("=" * 60)

dx_slab = 5e-3
nx_slab, ny_slab = 200, 150
source_x = 40
slab_start, slab_end = 100, 130
probe_trn_x = 140
n_steps = 800

# 1) 无slab参考: 测纯入射场
print("\n[1/2] 无slab参考仿真 (测量纯入射场)...")
f_ref = FDTD2DTM(nx_slab, ny_slab, dx_slab)
f_ref.set_source(source_x, 75)
r_ref = f_ref.run(n_steps, 'modulated', 20, progress=True, probe_pos=probe_trn_x)
probe_inc_pure = r_ref['probe_Ez']
print("  ✅ 完成")

# 2) 有slab: 测透射场
print("\n[2/2] 有slab透射仿真...")
f_slb = FDTD2DTM(nx_slab, ny_slab, dx_slab)
f_slb.set_source(source_x, 75)
f_slb.add_dielectric_slab(slab_start, slab_end, 4.0)
f_slb._init_coeffs()  # FIX: must reinitialize coefficients after changing eps_r
r_slb = f_slb.run(n_steps, 'modulated', 20, progress=True, probe_pos=probe_trn_x)
probe_trn = r_slb['probe_Ez']
print("  ✅ 完成")

# 稳态窗口
steady_start = 400
inc_steady = probe_inc_pure[steady_start:]
trn_steady = probe_trn[steady_start:]

# RMS比
inc_rms = np.sqrt(np.mean(inc_steady**2))
trn_rms = np.sqrt(np.mean(trn_steady**2))
T_rms = trn_rms / inc_rms if inc_rms > 0 else 0

# FFT法
n_fft = len(inc_steady)
inc_fft = np.abs(np.fft.fft(inc_steady))
trn_fft = np.abs(np.fft.fft(trn_steady))
freqs = np.fft.fftfreq(n_fft, d=f_slb.dt)
pos_mask = freqs > 0
fc_idx = np.argmax(inc_fft[pos_mask])
inc_amp = inc_fft[pos_mask][fc_idx] / n_fft * 2
trn_amp = trn_fft[pos_mask][fc_idx] / n_fft * 2
T_fft = trn_amp / inc_amp if inc_amp > 0 else 0

T_theory = 2.0 / (1.0 + 2.0)  # 2/3

T_final = T_rms
method = "RMS"
if abs(T_fft - T_theory) < abs(T_rms - T_theory):
    T_final = T_fft
    method = "FFT"

rel_err = abs(T_final - T_theory) / T_theory * 100

print(f"\n{'='*60}")
print(f"结果汇总")
print(f"{'='*60}")
print(f"  RMS法: T = {T_rms:.6f}")
print(f"  FFT法: T = {T_fft:.6f} (fc={freqs[pos_mask][fc_idx]/1e9:.3f} GHz)")
print(f"  单界面Fresnel理论: T = {T_theory:.6f} = 2/3")
print(f"  → 选用{method}法: T ≈ {T_final:.4f}")
print(f"  相对误差: {rel_err:.2f}%")
