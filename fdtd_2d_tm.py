#!/usr/bin/env python3
"""
fdtd_2d_tm.py — 2D FDTD TM模式电磁波仿真引擎 (简化版，无CPML)
==============================================
使用Mur一阶吸收边界，确保数值稳定
面向: 大学物理电磁学数值教学
"""

import numpy as np
from typing import Optional, Callable
import matplotlib
matplotlib.use('Agg')  # WSL无GUI，必须用Agg后端
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='fdtd_2d_tm')

c0 = 299792458
mu0 = 4 * np.pi * 1e-7
eps0 = 1 / (mu0 * c0**2)


class FDTD2DTM:
    """2D FDTD TM模式求解器 (Ez, Hx, Hy) — Mur吸收边界"""

    def __init__(self, nx: int, ny: int, dx: float,
                 dt: Optional[float] = None,
                 source_type: str = 'point'):
        self.nx = nx
        self.ny = ny
        self.dx = dx

        if dt is None:
            self.dt = dx / (c0 * np.sqrt(2)) * 0.99
        else:
            self.dt = dt

        self.S = c0 * self.dt / self.dx  # Courant数

        # 材料参数
        self.eps_r = np.ones((nx, ny))
        self.mu_r = np.ones((nx, ny))
        self.sigma_e = np.zeros((nx, ny))
        self.sigma_m = np.zeros((nx, ny))

        # Yee网格场分量
        self.Ez = np.zeros((nx, ny))
        self.Hx = np.zeros((nx, ny))
        self.Hy = np.zeros((nx, ny))

        # Mur吸收边界用的边界缓存
        self.Ez_xmin = np.zeros(ny)
        self.Ez_xmax = np.zeros(ny)
        self.Ez_ymin = np.zeros(nx)
        self.Ez_ymax = np.zeros(nx)

        # 更新系数
        self._init_coeffs()

        # 源位置
        self.source_x = nx // 2
        self.source_y = ny // 2
        self.last_Ez = np.zeros_like(self.Ez)

    def _init_coeffs(self):
        self.Ceze = (1 - self.sigma_e * self.dt / (2 * eps0 * self.eps_r)) / \
                    (1 + self.sigma_e * self.dt / (2 * eps0 * self.eps_r))
        self.Cezh = (self.dt / (eps0 * self.eps_r * self.dx)) / \
                    (1 + self.sigma_e * self.dt / (2 * eps0 * self.eps_r))
        self.Chxh = (1 - self.sigma_m * self.dt / (2 * mu0 * self.mu_r)) / \
                    (1 + self.sigma_m * self.dt / (2 * mu0 * self.mu_r))
        self.Chxe = (self.dt / (mu0 * self.mu_r * self.dx)) / \
                    (1 + self.sigma_m * self.dt / (2 * mu0 * self.mu_r))
        self.Chyh = np.copy(self.Chxh)
        self.Chye = np.copy(self.Chxe)

    def add_dielectric_slab(self, x_start: int, x_end: int, eps_r: float):
        self.eps_r[x_start:x_end, :] = eps_r

    def add_dielectric_circle(self, cx: float, cy: float, r: float, eps_r: float):
        Y, X = np.ogrid[:self.ny, :self.nx]
        mask = ((X - cx)**2 + (Y - cy)**2) <= r**2
        self.eps_r[mask.T] = eps_r
        self._init_coeffs()

    def add_pec_block(self, x_start: int, x_end: int, y_start: int, y_end: int):
        self.sigma_e[x_start:x_end, y_start:y_end] = 1e10
        self._init_coeffs()

    def set_source(self, x: int, y: int):
        self.source_x = x
        self.source_y = y

    def gaussian_pulse(self, t: int, tau: float = 15.0, t0: float = 40.0) -> float:
        return np.exp(-((t - t0) / tau)**2)

    def modulated_gaussian(self, t: int, tau: float = 30.0,
                            fc_hz: float = 3e9) -> float:
        arg = ((t - 3 * tau) / tau)**2
        return np.exp(-arg) * np.sin(2 * np.pi * fc_hz * t * self.dt)

    def sinusoidal(self, t: int, freq_hz: float = 2e9) -> float:
        return np.sin(2 * np.pi * freq_hz * t * self.dt)

    def _mur_abc(self):
        """Mur一阶吸收边界"""
        nx, ny = self.nx, self.ny
        cdt = c0 * self.dt
        # x方向边界
        for j in range(ny):
            self.Ez[0, j] = self.Ez_xmin[j] + \
                (cdt - self.dx) / (cdt + self.dx) * (self.Ez[1, j] - self.Ez[0, j])
            self.Ez_xmin[j] = self.Ez[1, j]

        for j in range(ny):
            self.Ez[nx-1, j] = self.Ez_xmax[j] + \
                (cdt - self.dx) / (cdt + self.dx) * (self.Ez[nx-2, j] - self.Ez[nx-1, j])
            self.Ez_xmax[j] = self.Ez[nx-2, j]

        # y方向边界
        for i in range(nx):
            self.Ez[i, 0] = self.Ez_ymin[i] + \
                (cdt - self.dx) / (cdt + self.dx) * (self.Ez[i, 1] - self.Ez[i, 0])
            self.Ez_ymin[i] = self.Ez[i, 1]

        for i in range(nx):
            self.Ez[i, ny-1] = self.Ez_ymax[i] + \
                (cdt - self.dx) / (cdt + self.dx) * (self.Ez[i, ny-2] - self.Ez[i, ny-1])
            self.Ez_ymax[i] = self.Ez[i, ny-2]

    def step_one(self, t: int, source_func: Optional[Callable] = None,
                 amplitude: float = 1.0):
        """执行一个FDTD时间步 (只在内部格点更新Ez, 边界由Mur处理)"""
        nx, ny = self.nx, self.ny

        # 1. 更新Hx: Hx^n+1/2 at (i, j+1/2), j=0..ny-2
        dEz_dy = self.Ez[:, 1:] - self.Ez[:, :-1]  # 纯差分, 不除dx (Chxe已含1/dx因子)
        self.Hx[:, :-1] = self.Chxh[:, :-1] * self.Hx[:, :-1] - \
                          self.Chxe[:, :-1] * dEz_dy

        # 2. 更新Hy: Hy^n+1/2 at (i+1/2, j), i=0..nx-2
        dEz_dx = self.Ez[1:, :] - self.Ez[:-1, :]  # 纯差分
        self.Hy[:-1, :] = self.Chyh[:-1, :] * self.Hy[:-1, :] + \
                          self.Chye[:-1, :] * dEz_dx

        # 3. 更新Ez^n+1 at (i, j), 只在内部格点 i=1..nx-2, j=1..ny-2
        # dHy/dx at (i, j): (Hy(i+1/2,j) - Hy(i-1/2,j)) → 纯差分 (Cezh已含1/dx)
        dHy_dx = self.Hy[1:-1, :] - self.Hy[:-2, :]
        # dHx/dy at (i, j): (Hx(i,j+1/2) - Hx(i,j-1/2)) → 纯差分
        dHx_dy = self.Hx[:, 1:-1] - self.Hx[:, :-2]
        # 在(i,j)处: (dHy/dx - dHx/dy), 取交集 i=1..nx-2, j=1..ny-2
        curl = dHy_dx[:, 1:-1] - dHx_dy[1:-1, :]  # shape (nx-2, ny-2)
        self.Ez[1:-1, 1:-1] = self.Ceze[1:-1, 1:-1] * self.Ez[1:-1, 1:-1] + \
                              self.Cezh[1:-1, 1:-1] * curl

        # 4. Mur一阶吸收边界 (修正边界上的Ez)
        self._mur_abc()

        # 5. 硬源注入 (在源位置覆盖Ez, 模拟理想源)
        if source_func is not None:
            self.Ez[self.source_x, self.source_y] = amplitude * source_func(t)

    def run(self, n_steps: int, source_func='gaussian',
            save_interval: int = 10, progress: bool = True,
            custom_source: Optional[Callable] = None,
            probe_pos: Optional[int] = None) -> dict:
        """运行FDTD仿真

        Args:
            source_func: 'gaussian'/'modulated'/'sinusoidal' 或 None (使用custom_source)
            custom_source: 自定义源函数 callable(t_step) -> float
            probe_pos: 探针x位置（格点），None则设为source_x+20
        """
        n_snap = n_steps // save_interval + 1
        Ez_snapshots = np.zeros((n_snap, self.nx, self.ny))
        probe_Ez = np.zeros(n_steps)
        times = np.arange(n_steps) * self.dt

        probe_x = min(probe_pos, self.nx - 2) if probe_pos is not None else \
                  min(self.source_x + 20, self.nx - 2)
        probe_y = self.source_y

        if custom_source is not None:
            src = custom_source
        elif source_func == 'gaussian':
            src = lambda t: self.gaussian_pulse(t, tau=15, t0=40)
        elif source_func == 'modulated':
            src = lambda t: self.modulated_gaussian(t)
        elif source_func == 'sinusoidal':
            src = lambda t: self.sinusoidal(t)
        else:
            src = lambda t: self.gaussian_pulse(t, tau=15, t0=40)

        snap_idx = 0
        for n in range(n_steps):
            self.step_one(n, source_func=src)
            self.last_Ez = self.Ez.copy()
            probe_Ez[n] = self.Ez[probe_x, probe_y]
            if n % save_interval == 0:
                Ez_snapshots[snap_idx] = self.Ez.copy()
                snap_idx += 1
            if progress and n % (n_steps // 10 + 1) == 0:
                print(f"  step {n}/{n_steps}")

        Ez_snapshots = Ez_snapshots[:snap_idx]
        return {
            'Ez_snapshots': Ez_snapshots,
            'probe_Ez': probe_Ez,
            'times': times,
            'times_snap': np.arange(0, n_steps, save_interval) * self.dt
        }


# ============================================================
# 场景
# ============================================================

def run_free_space():
    """自由空间波传播"""
    print("\n[场景1] 自由空间")
    f = FDTD2DTM(150, 150, 1e-2)
    f.set_source(75, 75)
    return f, f.run(400, 'gaussian', 10)


def run_reflection():
    """PEC反射"""
    print("\n[场景2] PEC反射")
    f = FDTD2DTM(150, 150, 1e-2)
    f.set_source(40, 75)
    f.add_pec_block(110, 113, 40, 110)
    return f, f.run(500, 'gaussian', 10)


def run_dielectric():
    """介质平板"""
    print("\n[场景3] 介质板透射")
    f = FDTD2DTM(200, 150, 5e-3)
    f.set_source(40, 75)
    f.add_dielectric_slab(100, 130, 4.0)
    return f, f.run(600, 'modulated', 10)


def run_two_slit():
    """双缝干涉"""
    print("\n[场景4] 双缝干涉")
    f = FDTD2DTM(200, 300, 2e-3)
    f.set_source(30, 150)
    bx, cy, sw = 70, 150, 6
    for j in range(f.ny):
        if abs(j - (cy - 25)) > sw//2 and abs(j - (cy + 25)) > sw//2:
            f.sigma_e[bx-1:bx+1, j] = 1e10
    f._init_coeffs()
    return f, f.run(1000, 'sinusoidal', 10)


def run_waveguide():
    """平行板波导"""
    print("\n[场景5] 平行板波导")
    f = FDTD2DTM(300, 80, 2e-3)
    f.set_source(30, 40)
    f.sigma_e[:, 0:3] = 1e10
    f.sigma_e[:, 77:80] = 1e10
    f._init_coeffs()
    return f, f.run(800, 'modulated', 10)


def convergence_test():
    """收敛性测试 v2 — 固定物理问题 + 真实L₂范数 + 参考解验证
    
    策略: 中等域+近场探针，在边界反射到达前分析纯内部收敛性。
    使用窄高斯脉冲，固定物理参数的源和探针位置。
    """
    print("\n[收敛性测试 v2]")

    L = 2.0            # 域尺寸 m — 足够大且计算量可控
    tau_phys = 0.5e-9  # 高斯脉冲宽度 0.5 ns
    t0_phys = 1.5e-9   # 脉冲峰值延迟 1.5 ns  
    probe_dist = 0.2   # 探针距源 m (近场，4-40格)
    T_sim = 4e-9       # 4 ns (边界反射约>3.3ns到达探针)

    factors = [1, 2, 3, 4, 5, 6, 8, 10]
    nx_list = [40 * f for f in factors]
    dx_list = [L / nx for nx in nx_list]

    all_probes = []
    dt_list = []

    for nx, dx in zip(nx_list, dx_list):
        f = FDTD2DTM(nx, nx, dx)
        f.set_source(nx // 2, nx // 2)
        probe_cells = max(int(probe_dist / dx), 2)
        probe_x = f.source_x + probe_cells
        n_steps = int(T_sim / f.dt)
        def src(t_step):
            return np.exp(-((t_step * f.dt - t0_phys) / tau_phys) ** 2)
        print(f"  nx={nx:3d}, dx={dx:.5f}, dt={f.dt:.3e}, n={n_steps:3d}")
        r = f.run(n_steps, None, max(n_steps // 10, 3), progress=False,
                  custom_source=src, probe_pos=probe_x)
        all_probes.append(r['probe_Ez'])
        dt_list.append(f.dt)

    ref = all_probes[-1]
    ref_dt = dt_list[-1]
    ref_t = np.arange(len(ref)) * ref_dt

    # 时间窗口: 脉冲通过探针后, 反射到达前
    # 探针距源0.2m→飞行时间0.67ns→加t0=1.5ns→峰值~2.17ns
    # 边界(源距边1m)→反射到探针: 0.67+1.33=2.0ns→实际Mur透射而非全反, 窗口安全
    t_start, t_end = 1.5e-9, 3.5e-9

    errs = []
    for i, (nx, dx, probe, dt) in enumerate(zip(nx_list, dx_list, all_probes, dt_list)):
        this_t = np.arange(len(probe)) * dt
        t_mask = (this_t >= t_start) & (this_t <= t_end)
        ref_interp = np.interp(this_t[t_mask], ref_t, ref)
        probe_masked = probe[t_mask]

        l2_err = np.sqrt(np.mean((probe_masked - ref_interp)**2))
        ref_norm = np.sqrt(np.mean(ref_interp**2))
        rel_err = l2_err / ref_norm if ref_norm > 0 else 0.0
        errs.append(rel_err)
        print(f"  nx={nx:3d}, dx={dx:.5f}: L₂误差={rel_err:.6e}")

    dxs = np.array(dx_list)
    errs = np.array(errs)

    # Log-log 拟合 — 排除参考解(nx=400, err≈0)
    fit_mask = (errs > 1e-15)
    if np.sum(fit_mask) < 3:
        fit_mask = np.ones_like(errs, dtype=bool)

    coeffs = np.polyfit(np.log(dxs[fit_mask]), np.log(errs[fit_mask]), 1)
    p = coeffs[0]
    residuals = np.log(errs[fit_mask]) - (coeffs[0] * np.log(dxs[fit_mask]) + coeffs[1])
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((np.log(errs[fit_mask]) - np.mean(np.log(errs[fit_mask])))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    print(f"\n  收敛阶: p = {p:.4f}, R² = {r2:.4f}, 拟合点: {np.sum(fit_mask)}/{len(errs)}")

    return dxs, errs, p, r2


# ============================================================
# 可视化
# ============================================================

def plot_snapshot(fdtd, title="", save_path=None):
    """画Ez快照, 用99.5%分位数色标避免硬源奇异性压垮色彩"""
    fig, ax = plt.subplots(figsize=(7, 5))
    absEz = np.abs(fdtd.Ez)
    vmax = np.percentile(absEz, 99.5)
    vmin = -vmax
    im = ax.imshow(fdtd.Ez.T, origin='lower', cmap='RdBu_r',
                   vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Ez')
    ax.set_xlabel('x (grid)')
    ax.set_ylabel('y (grid)')
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  保存: {save_path}")
    plt.close()


def plot_probe(probe_Ez, times, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(times * 1e9, probe_Ez, 'b-', linewidth=1.5)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Ez')
    ax.set_title('Probe Time Signal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_convergence(dx, err, p, r_squared, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(dx, err, 'bo-', markersize=8, linewidth=2, label='FDTD L₂ error')

    # 拟合曲线
    C_fit = err[0] / dx[0]**p
    fit_line = C_fit * dx**p
    ax.loglog(dx, fit_line, 'r--', linewidth=2,
              label=rf'Fit: $p={p:.2f}$, $R^2={r_squared:.3f}$')

    # 理论二阶参考线
    ref2 = err[0] * (dx / dx[0])**2
    ax.loglog(dx, ref2, 'g:', linewidth=1.5, label=r'$O(\Delta x^2)$ reference')

    ax.set_xlabel(r'Spatial step $\Delta x$ (m)')
    ax.set_ylabel('Relative L₂ error')
    ax.set_title('FDTD Convergence Analysis')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  保存: {save_path}")
    plt.close()


if __name__ == '__main__':
    print("=" * 50)
    print("FDTD 2D TM Electromagnetic Wave Simulation")
    print("=" * 50)

    fdtd1, r1 = run_free_space()
    fdtd2, r2 = run_reflection()
    fdtd3, r3 = run_dielectric()
    fdtd4, r4 = run_two_slit()
    fdtd5, r5 = run_waveguide()
    dxs, errs, p, r2 = convergence_test()

    # 出图
    out = os.path.join(os.path.dirname(__file__), '..', '..', '03_论文输出', 'figures')
    os.makedirs(out, exist_ok=True)

    plot_snapshot(fdtd1, "Free space", os.path.join(out, "fig_free_space.png"))
    plot_snapshot(fdtd2, "PEC reflection", os.path.join(out, "fig_reflection.png"))
    plot_snapshot(fdtd3, "Dielectric slab", os.path.join(out, "fig_dielectric.png"))
    plot_snapshot(fdtd4, "Double-slit interference", os.path.join(out, "fig_interference.png"))
    # 双缝: 用10GHz代替2GHz, 使过缝周期数从1.6→10.7, 干涉条纹稳态形成
    # 10GHz: λ=15格, 周期=21.4步, 跑400步过缝后229步=10.7周期
    f_slit2 = FDTD2DTM(500, 300, 2e-3)
    f_slit2.set_source(80, 150)
    bx, cy, sw = 200, 150, 6       # 缝宽6格=12mm≈0.4λ
    slit_sep = 50                   # 缝间距50格=100mm≈3.3λ
    for j in range(f_slit2.ny):
        if abs(j - (cy - slit_sep//2)) > sw//2 and abs(j - (cy + slit_sep//2)) > sw//2:
            f_slit2.sigma_e[bx-2:bx+2, j] = 1e10
    f_slit2._init_coeffs()
    # 用10GHz正弦源
    f_slit2.run(400, None, 100, progress=False,
                custom_source=lambda t: np.sin(2*np.pi*10e9*t*f_slit2.dt))
    # 屏后区域归一化画图
    fig, ax = plt.subplots(figsize=(8, 4))
    x_max = 420
    Ez_crop = f_slit2.Ez[:x_max, :]
    behind = Ez_crop[bx:, :]
    vmax = np.percentile(np.abs(behind), 99.5)
    im = ax.imshow(Ez_crop.T, origin='lower', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, aspect='equal',
                   extent=[0, x_max, 0, f_slit2.ny])
    ax.axvline(bx, color='k', linewidth=1.5, linestyle='--', alpha=0.7, label='Screen')
    plt.colorbar(im, ax=ax, label='Ez', shrink=0.8)
    ax.set_xlabel('x (grid cells)')
    ax.set_ylabel('y (grid cells)')
    ax.set_title(f'Double-slit interference (10GHz, {10.7:.0f} periods past slits)')
    ax.set_xlim(0, x_max)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_interference.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存双缝图(屏后归一化)")
    # 波导: 跑400步(波传到~一半), 避免800步后波已出右边界
    f_wg2 = FDTD2DTM(300, 80, 2e-3)
    f_wg2.set_source(30, 40)
    f_wg2.sigma_e[:, 0:3] = 1e10
    f_wg2.sigma_e[:, 77:80] = 1e10
    f_wg2._init_coeffs()
    r_wg2 = f_wg2.run(400, 'modulated', 1000, progress=False)
    plot_snapshot(f_wg2, "Parallel-plate waveguide", os.path.join(out, "fig_waveguide.png"))
    plot_probe(r1['probe_Ez'], r1['times'], os.path.join(out, "fig_probe.png"))
    plot_convergence(dxs, errs, p, r2, os.path.join(out, "fig_convergence.png"))

    print(f"\nAll figures saved to: {out}")
