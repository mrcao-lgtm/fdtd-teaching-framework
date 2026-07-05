#!/usr/bin/env python3
"""
generate_figures.py — FDTD仿真结果可视化与论文用图生成

用法: ./generate_figures.py 或 /path/to/hermes/python3 generate_figures.py
输出: 论文输出/figures/*.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import sys, os, json

sys.path.insert(0, os.path.dirname(__file__))
from fdtd_2d_tm import (
    FDTD2DTM, c0,
    run_free_space, run_reflection,
    run_dielectric, run_two_slit,
    run_waveguide, convergence_test
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '03_论文输出', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 颜色风格
COLORS = {
    'ez_cmap': 'RdBu_r',
    'pec': '#444444',
    'dielectric': '#88CC88',
    'probe': '#1f77b4',
    'analytical': '#d62728'
}

def _vmax_ez(Ez, p=99.5):
    """用百分位数色标，避免硬源奇异性压垮色彩"""
    return np.percentile(np.abs(Ez), p)

print("=" * 60)
print("FDTD电磁波仿真 — 论文用图生成")
print("=" * 60)

# ============================================================
# Fig 1: FDTD网格示意图 (Yee网格)
# ============================================================
def fig1_yee_grid():
    """Fig 1: Yee网格示意图"""
    print("\n[Fig 1] Yee网格示意图...")
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # 画网格线
    for i in range(6):
        ax.axvline(i, color='gray', linewidth=0.5, alpha=0.5)
        ax.axhline(i, color='gray', linewidth=0.5, alpha=0.5)
    
    # Ez在网格点 (i, j)
    for i in range(5):
        for j in range(5):
            ax.plot(i, j, 'o', color='#E74C3C', markersize=8, zorder=5)
    
    # Hx在 (i, j+1/2)
    for i in range(5):
        for j in range(4):
            ax.plot(i, j + 0.5, '^', color='#3498DB', markersize=8, zorder=5)
    
    # Hy在 (i+1/2, j)
    for i in range(4):
        for j in range(5):
            ax.plot(i + 0.5, j, 's', color='#2ECC71', markersize=8, zorder=5)
    
    # 标注
    ax.plot([], [], 'o', color='#E74C3C', label='Ez(i,j)', markersize=8)
    ax.plot([], [], '^', color='#3498DB', label='Hx(i,j+1/2)', markersize=8)
    ax.plot([], [], 's', color='#2ECC71', label='Hy(i+1/2,j)', markersize=8)
    
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 5.5)
    ax.set_xlabel('i (x-index)', fontsize=11)
    ax.set_ylabel('j (y-index)', fontsize=11)
    ax.set_title('Yee grid configuration for TM$_z$ mode', fontsize=12)
    ax.set_aspect('equal')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig1_yee_grid.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")

# ============================================================
# Fig 2: 自由空间波传播 (PML验证)
# ============================================================
def fig2_free_space():
    """Fig 2: 自由空间波传播"""
    print("\n[Fig 2] 自由空间波传播...")
    fdtd, result = run_free_space()
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    snapshots = [0, len(result['Ez_snapshots'])//4, len(result['Ez_snapshots'])//2,
                 3*len(result['Ez_snapshots'])//4, -1]
    
    for idx, ax in zip(snapshots, axes.flat[:5]):
        t_ns = result['times_snap'][idx] * 1e9
        absEz = np.abs(result['Ez_snapshots'][idx])
        vmax = np.percentile(absEz, 99.5)
        ax.imshow(result['Ez_snapshots'][idx].T, origin='lower',
                  cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
        ax.set_title(f't={t_ns:.1f} ns', fontsize=10)
        ax.set_xlabel('x (格点)')
        ax.set_ylabel('y (格点)')
    
    # 第6个子图: 探针信号
    ax = axes.flat[5]
    ax.plot(result['times'] * 1e9, result['probe_Ez'], color=COLORS['probe'])
    ax.set_xlabel('时间 (ns)')
    ax.set_ylabel('Ez (相对值)')
    ax.set_title('探针点时域波形')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('自由空间电磁波传播 (PML吸收边界验证)', fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig2_free_space.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")
    return fdtd, result


# ============================================================
# Fig 3: PEC反射
# ============================================================
def fig3_reflection():
    """Fig 3: PEC平板反射"""
    print("\n[Fig 3] PEC平板反射...")
    fdtd, result = run_reflection()
    
    # 选取反射波清晰的时刻
    snap_idx = len(result['Ez_snapshots']) // 2
    Ez = result['Ez_snapshots'][snap_idx]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # 左: 反射场景快照
    vmax = _vmax_ez(Ez)
    im = axes[0].imshow(Ez.T, origin='lower', cmap='RdBu_r',
                        vmin=-vmax, vmax=vmax, aspect='equal')
    # 标出PEC
    axes[0].axvline(3*fdtd.nx//4, color='k', linewidth=3, label='PEC')
    axes[0].set_title(f't={result["times_snap"][snap_idx]*1e9:.1f} ns')
    axes[0].set_xlabel('x (格点)')
    axes[0].set_ylabel('y (格点)')
    
    # 中: 中心切线的Ez分布
    center_y = fdtd.ny // 2
    axes[1].plot(Ez[:, center_y], 'b-', linewidth=1.5)
    axes[1].axvline(3*fdtd.nx//4, color='k', linewidth=2, linestyle='--', label='PEC')
    axes[1].set_xlabel('x (格点)')
    axes[1].set_ylabel('Ez')
    axes[1].set_title('x方向Ez分布 (y=中心)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 右: 探针信号
    axes[2].plot(result['times'] * 1e9, result['probe_Ez'], color=COLORS['probe'])
    axes[2].set_xlabel('时间 (ns)')
    axes[2].set_ylabel('Ez')
    axes[2].set_title('探针信号 (入射+反射)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig3_reflection.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")

# ============================================================
# Fig 4: 介质平板透射
# ============================================================
def fig4_dielectric():
    """Fig 4: 介质平板反射与透射"""
    print("\n[Fig 4] 介质平板透射...")
    fdtd, result = run_dielectric()
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # 入射前
    idx1 = len(result['Ez_snapshots']) // 8
    Ez1 = result['Ez_snapshots'][idx1]
    vmax = _vmax_ez(Ez1)
    axes[0, 0].imshow(Ez1.T, origin='lower', cmap='RdBu_r',
                      vmin=-vmax, vmax=vmax, aspect='equal')
    axes[0, 0].set_title(f'入射波 t={result["times_snap"][idx1]*1e9:.1f} ns')
    axes[0, 0].set_xlabel('x (格点)')
    axes[0, 0].set_ylabel('y (格点)')
    
    # 反射+透射
    idx2 = 3 * len(result['Ez_snapshots']) // 4
    Ez2 = result['Ez_snapshots'][idx2]
    vmax = _vmax_ez(Ez2)
    im = axes[0, 1].imshow(Ez2.T, origin='lower', cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, aspect='equal')
    # 介质区域
    slab_start = fdtd.nx // 2 - 15
    slab_end = slab_start + 30
    axes[0, 1].axvspan(slab_start, slab_end, color='green', alpha=0.15)
    axes[0, 1].set_title(f'反射+透射 t={result["times_snap"][idx2]*1e9:.1f} ns')
    axes[0, 1].set_xlabel('x (格点)')
    axes[0, 1].set_ylabel('y (格点)')
    
    # 中心x方向Ez分布
    center_y = fdtd.ny // 2
    axes[1, 0].plot(Ez2[:, center_y], 'b-', linewidth=1.5)
    axes[1, 0].axvspan(slab_start, slab_end, color='green', alpha=0.15,
                       label='介质(ε_r=4)')
    axes[1, 0].set_xlabel('x (格点)')
    axes[1, 0].set_ylabel('Ez')
    axes[1, 0].set_title('x方向Ez分布')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 时域探针
    axes[1, 1].plot(result['times'] * 1e9, result['probe_Ez'], color=COLORS['probe'])
    axes[1, 1].set_xlabel('时间 (ns)')
    axes[1, 1].set_ylabel('Ez')
    axes[1, 1].set_title('探针信号 (含入射+反射+透射)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig4_dielectric.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")

# ============================================================
# Fig 5: 双缝干涉
# ============================================================
def fig5_interference():
    """Fig 5: 双缝干涉"""
    print("\n[Fig 5] 双缝干涉...")
    fdtd, result = run_two_slit()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 最终时刻干涉图
    Ez_final = result['Ez_snapshots'][-1]
    vmax = _vmax_ez(Ez_final)
    im = axes[0].imshow(Ez_final.T, origin='lower', cmap='RdBu_r',
                        vmin=-vmax, vmax=vmax, aspect='equal')
    axes[0].set_title('双缝干涉 Ez场分布（稳态）')
    axes[0].set_xlabel('x (格点)')
    axes[0].set_ylabel('y (格点)')
    plt.colorbar(im, ax=axes[0], label='Ez (相对值)')
    
    # 远场y方向分布（干涉条纹）—— 取挡板后一定距离
    measure_x = fdtd.nx // 2
    axes[1].plot(Ez_final[measure_x, :], 'b-', linewidth=1.5)
    axes[1].set_xlabel('y (格点)')
    axes[1].set_ylabel('Ez (相对值)')
    axes[1].set_title(f'远场干涉条纹 (x={measure_x})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig5_interference.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")

# ============================================================
# Fig 6: 收敛性分析
# ============================================================
def fig6_convergence():
    """Fig 6: 收敛性分析"""
    print("\n[Fig 6] 收敛性分析...")
    dx_vals, errors = convergence_test()
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.loglog(dx_vals, errors, 'bo-', markersize=10, linewidth=2,
              label='FDTD数值误差')
    
    # 理论二阶收敛
    ref = errors[0] * (dx_vals / dx_vals[0])**2
    ax.loglog(dx_vals, ref, 'r--', linewidth=2, label='O(Δx²) 理论')
    
    # 标注斜率和点
    for i, (dx, err) in enumerate(zip(dx_vals, errors)):
        ax.annotate(f'  {err:.2e}', (dx, err), fontsize=9)
    
    ax.set_xlabel('空间步长 Δx (m)', fontsize=12)
    ax.set_ylabel('归一化数值误差', fontsize=12)
    ax.set_title('FDTD数值收敛性分析', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 插入小图: 典型Ez场分布
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_inset = inset_axes(ax, width="35%", height="35%", loc='lower left')
    
    fdtd_test = FDTD2DTM(100, 100, 1e-2)
    fdtd_test.set_source(50, 50)
    for t in range(200):
        fdtd_test.step_one(t, source_func=lambda t: fdtd_test.gaussian_pulse(t, tau=10, t0=30))
    
    vmax = _vmax_ez(fdtd_test.Ez)
    ax_inset.imshow(fdtd_test.Ez.T, origin='lower', cmap='RdBu_r',
                    vmin=-vmax, vmax=vmax)
    ax_inset.set_title('Ez分布 (100×100)', fontsize=8)
    ax_inset.axis('off')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig6_convergence.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")

# ============================================================
# Fig 7: 并行板波导
# ============================================================
def fig7_waveguide():
    """Fig 7: 平行板波导"""
    print("\n[Fig 7] 平行板波导...")
    fdtd, result = run_waveguide()
    
    # 选一个传播清晰的时刻
    snap_idx = len(result['Ez_snapshots']) // 2
    Ez = result['Ez_snapshots'][snap_idx]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    vmax = _vmax_ez(Ez)
    im = ax.imshow(Ez.T, origin='lower', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, aspect='equal',
                   extent=[0, fdtd.nx, 0, fdtd.ny])
    # 标出波导壁
    ax.axhline(2, color='k', linewidth=3)
    ax.axhline(fdtd.ny-3, color='k', linewidth=3)
    ax.set_xlabel('传播方向 x (格点)')
    ax.set_ylabel('y (格点)')
    ax.set_title('平行板波导中的电磁波传播')
    plt.colorbar(im, ax=ax, label='Ez')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig7_waveguide.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")

# ============================================================
# 保存仿真数据（论文引用用）
# ============================================================
def save_simulation_data():
    """保存关键仿真数据到JSON"""
    print("\n[数据] 保存仿真元数据...")
    data = {
        'FDTD_params': {
            'c0': c0,
            'CFL_number': 0.99 / np.sqrt(2),
            'pml_order': 3,
            'pml_reflection_coeff': '< -60 dB'
        },
        'scenarios': {
            'free_space': {
                'grid': '200×200',
                'dx_m': 1e-2,
                'steps': 500,
                'source': 'gaussian_pulse'
            },
            'reflection': {
                'grid': '200×200',
                'pec_position': 'x=3nx/4',
                'steps': 600
            },
            'dielectric_slab': {
                'grid': '300×200',
                'eps_r': 4.0,
                'slab_thickness': '30 cells (15mm)',
                'steps': 800
            },
            'two_slit': {
                'grid': '300×400',
                'slit_width_cells': 8,
                'slit_sep_cells': 40,
                'steps': 1200
            },
            'waveguide': {
                'grid': '400×100',
                'steps': 1000
            }
        },
        'convergence': dx_vals.tolist() if 'dx_vals' in dir() else [],
    }
    path = os.path.join(OUTPUT_DIR, 'simulation_data.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✅ {path}")


if __name__ == '__main__':
    # 先生成所有场景图
    fig1_yee_grid()
    fig2_free_space()
    fig3_reflection()
    fig4_dielectric()
    fig5_interference()
    fig7_waveguide()
    fig6_convergence()
    save_simulation_data()
    
    print("\n" + "=" * 60)
    print("🎉 所有论文用图已生成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)
