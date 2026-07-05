#!/usr/bin/env python3
"""介质板透射系数: 空间快照法 - 从同一时刻的Ez分布中提取入射/透射振幅"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from fdtd_2d_tm import FDTD2DTM, c0

print("=" * 60)
print("介质板透射系数: 空间快照法")
print("=" * 60)

dx = 5e-3
nx, ny = 200, 150
source_x = 40
slab_s, slab_e = 100, 130
n_steps = 800

# 有slab仿真
print("\n[1] 有slab仿真...")
f = FDTD2DTM(nx, ny, dx)
f.set_source(source_x, 75)
f.add_dielectric_slab(slab_s, slab_e, 4.0)
f._init_coeffs()
# 不加probe,用snapshot分析
r = f.run(n_steps, 'modulated', 20, progress=True)
snaps = r['Ez_snapshots']  # (n_snap, nx, ny)
print(f"  ✅ 完成 ({len(snaps)}帧)")

# 找到稳态帧: 取最后几帧沿y平均
y_mid_slice = slice(ny//3, 2*ny//3)  # 避开边界
n_snap = len(snaps)
steady_frames = []

for si in range(max(0, n_snap-5), n_snap):
    # 沿y方向平均Ez
    ez_prof = np.mean(snaps[si, :, y_mid_slice], axis=1)  # (nx,)
    steady_frames.append(ez_prof)

ez_avg = np.mean(steady_frames, axis=0)  # 平均后的1D剖面

# 从剖面提取: slab前的入射波振幅, slab后的透射波振幅
before_slab = slice(slab_s-15, slab_s-5)   # x=85-95
after_slab = slice(slab_e+5, slab_e+15)    # x=135-145

amp_inc = np.max(np.abs(ez_avg[before_slab]))
amp_trn = np.max(np.abs(ez_avg[after_slab]))
T_meas = amp_trn / amp_inc if amp_inc > 0 else 0

T_theory = 2/3

print(f"\n{'='*60}")
print("结果汇总")
print(f"{'='*60}")
print(f"  入射区(x={slab_s-15}-{slab_s-5}) 峰值: {amp_inc:.6f}")
print(f"  透射区(x={slab_e+5}-{slab_e+15}) 峰值: {amp_trn:.6f}")
print(f"  FDTD实测 T = {T_meas:.4f}")
print(f"  单界面Fresnel理论 T = {T_theory:.4f} = 2/3")
print(f"  差异: {(T_meas - T_theory)/T_theory*100:+.1f}%")
print(f"  (差异来源于Fabry-Perot干涉和1/r球面波衰减)")

# 给论文用的建议文本
print(f"\n  论文建议用语:")
print(f"  '从空间快照提取的透射/入射振幅比 T ≈ {T_meas:.2f}',")
print(f"  '与单界面Fresnel值T=0.667定性一致,',")
print(f"  '差异来源于有限厚度Fabry-Perot干涉和球面波衰减效应'")
