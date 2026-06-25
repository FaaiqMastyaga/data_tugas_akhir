#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import warnings

warnings.filterwarnings('ignore')

# Konfigurasi gaya plot akademis (Mendukung notasi Math/LaTeX)
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm', # Computer Modern (Standar LaTeX)
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 1.5,
    'figure.autolayout': True,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.7
})

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_BASE_DIR = PROJECT_ROOT / "experiment_results"

def calculate_dynamic_local_frame(df):
    ref_loc_x, ref_loc_y, ref_loc_z = [], [], []
    act_loc_x, act_loc_y, act_loc_z = [], [], []
    
    wb_pos = df[['wb_p_eskf_x', 'wb_p_eskf_y', 'wb_p_eskf_z']].values
    wb_quat = df[['wb_p_eskf_qx', 'wb_p_eskf_qy', 'wb_p_eskf_qz', 'wb_p_eskf_qw']].values
    
    ref_pos = df[['ref_p_x', 'ref_p_y', 'ref_p_z']].values
    act_pos = df[['act_p_x', 'act_p_y', 'act_p_z']].values
    
    for i in range(len(df)):
        rot_inv = R.from_quat(wb_quat[i]).inv()
        ref_local = rot_inv.apply(ref_pos[i] - wb_pos[i]) * 1000.0
        act_local = rot_inv.apply(act_pos[i] - wb_pos[i]) * 1000.0
        
        ref_loc_x.append(ref_local[0]); ref_loc_y.append(ref_local[1]); ref_loc_z.append(ref_local[2])
        act_loc_x.append(act_local[0]); act_loc_y.append(act_local[1]); act_loc_z.append(act_local[2])
        
    df['ref_loc_x'] = ref_loc_x; df['ref_loc_y'] = ref_loc_y; df['ref_loc_z'] = ref_loc_z
    df['act_loc_x'] = act_loc_x; df['act_loc_y'] = act_loc_y; df['act_loc_z'] = act_loc_z
    
    cx, cy = (np.max(ref_loc_x) + np.min(ref_loc_x)) / 2.0, (np.max(ref_loc_y) + np.min(ref_loc_y)) / 2.0
    df['ref_loc_x'] -= cx; df['act_loc_x'] -= cx
    df['ref_loc_y'] -= cy; df['act_loc_y'] -= cy
    return df

def get_drawing_phase_bounds(df, run_name):
    if "uji4" in run_name.lower() or "uji5" in run_name.lower():
        return None, None
        
    z_min = df['ref_loc_z'].min()
    z_max = df['ref_loc_z'].max()
    z_range = z_max - z_min
    
    if z_range < 2.0:
        return None, None
        
    threshold = z_min + (0.15 * z_range)
    drawing_mask = df['ref_loc_z'] <= threshold
    
    if drawing_mask.any():
        t_start = df.loc[drawing_mask, 'time_norm'].min()
        t_end = df.loc[drawing_mask, 'time_norm'].max()
        return t_start, t_end
    return None, None

def trim_timeseries(df, t_start, t_end, pre_margin=2.0, post_margin=2.0):
    if t_start is None or t_end is None:
        return df, t_start, t_end
    
    cut_start = max(0, t_start - pre_margin)
    cut_end = t_end + post_margin
    
    df_trimmed = df[(df['time_norm'] >= cut_start) & (df['time_norm'] <= cut_end)].copy()
    if df_trimmed.empty:
        return df, t_start, t_end
        
    time_offset = df_trimmed['time_norm'].iloc[0]
    df_trimmed['time_norm'] = df_trimmed['time_norm'] - time_offset
    
    new_t_start = t_start - time_offset
    new_t_end = t_end - time_offset
    return df_trimmed, new_t_start, new_t_end

def add_shading_and_fix_legend(ax, t_start, t_end):
    if t_start is not None and t_end is not None:
        ax.axvline(t_start, color='gray', linestyle=':', linewidth=1.5)
        ax.axvline(t_end, color='gray', linestyle=':', linewidth=1.5)
        ax.axvspan(t_start, t_end, color='#00FF00', alpha=0.08, label='Fase Eksekusi Geometri')
        
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

def plot_trajectories_2d(df, run_name, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].plot(df['ref_p_x'] * 1000, df['ref_p_y'] * 1000, 'k--', label=r'$p_{ref}$ (Target)', zorder=2)
    axes[0].plot(df['act_p_x'] * 1000, df['act_p_y'] * 1000, 'b-', label=r'$p_{act}$ (Aktual)', alpha=0.7, zorder=1)
    axes[0].set_title(rf"Trajektori Spasial Absolut (Kerangka Basis) - {run_name}"); axes[0].set_xlabel(r"Posisi Global $X$ (mm)"); axes[0].set_ylabel(r"Posisi Global $Y$ (mm)")
    axes[0].legend(); axes[0].axis('equal')
    
    axes[1].plot(df['ref_loc_x'], df['ref_loc_y'], 'k--', label=r'$p_{ref}$ (Target)', zorder=2)
    axes[1].plot(df['act_loc_x'], df['act_loc_y'], 'g-', label=r'$p_{act}$ (Aktual)', alpha=0.7, zorder=1)
    axes[1].set_title(rf"Trajektori Spasial Relatif (Kerangka Target) - {run_name}"); axes[1].set_xlabel(r"Posisi Lokal $X$ (mm)"); axes[1].set_ylabel(r"Posisi Lokal $Y$ (mm)")
    axes[1].legend(); axes[1].axis('equal')
    fig.savefig(out_dir / f"{run_name}_0_trajectory_2d.png", dpi=300); plt.close(fig)

def plot_pose_6dof(df, run_name, out_dir, t_start, t_end):
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    t = df['time_norm']
    
    axes[0, 0].plot(t, df['ref_p_x'] * 1000, 'k--', label=r'$x_{ref}$'); axes[0, 0].plot(t, df['act_p_x'] * 1000, 'b-', label=r'$x_{act}$', alpha=0.7); axes[0, 0].set_ylabel(r"Posisi $X$ (mm)")
    axes[1, 0].plot(t, df['ref_p_y'] * 1000, 'k--', label=r'$y_{ref}$'); axes[1, 0].plot(t, df['act_p_y'] * 1000, 'b-', label=r'$y_{act}$', alpha=0.7); axes[1, 0].set_ylabel(r"Posisi $Y$ (mm)")
    axes[2, 0].plot(t, df['ref_p_z'] * 1000, 'k--', label=r'$z_{ref}$'); axes[2, 0].plot(t, df['act_p_z'] * 1000, 'b-', label=r'$z_{act}$', alpha=0.7); axes[2, 0].set_ylabel(r"Posisi $Z$ (mm)"); axes[2, 0].set_xlabel(r"Waktu $t$ (s)")
    axes[0, 1].plot(t, np.degrees(df['ref_p_roll']), 'k--', label=r'$\phi_{ref}$'); axes[0, 1].plot(t, np.degrees(df['act_p_roll']), 'g-', label=r'$\phi_{act}$', alpha=0.7); axes[0, 1].set_ylabel(r"Sudut Roll $\phi$ (°)")
    axes[1, 1].plot(t, np.degrees(df['ref_p_pitch']), 'k--', label=r'$\theta_{ref}$'); axes[1, 1].plot(t, np.degrees(df['act_p_pitch']), 'g-', label=r'$\theta_{act}$', alpha=0.7); axes[1, 1].set_ylabel(r"Sudut Pitch $\theta$ (°)")
    axes[2, 1].plot(t, np.degrees(df['ref_p_yaw']), 'k--', label=r'$\psi_{ref}$'); axes[2, 1].plot(t, np.degrees(df['act_p_yaw']), 'g-', label=r'$\psi_{act}$', alpha=0.7); axes[2, 1].set_ylabel(r"Sudut Yaw $\psi$ (°)"); axes[2, 1].set_xlabel(r"Waktu $t$ (s)")
    
    for ax in axes.flat: add_shading_and_fix_legend(ax, t_start, t_end)
    
    fig.suptitle(rf"Analisis Pelacakan Pose 6-DOF (Translasi & Orientasi) - {run_name}", fontsize=16); fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_1_pose_tracking_6dof.png", dpi=300); plt.close(fig)

def plot_twist_control_6dof(df, run_name, out_dir, t_start, t_end):
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    t = df['time_norm']
    
    axes[0, 0].plot(t, df['ref_t_vx'] * 100, 'k--', label=r'$v_{x,ref}$'); axes[0, 0].plot(t, df['act_t_vx'] * 100, 'b-', label=r'$v_{x,act}$', alpha=0.7); axes[0, 0].set_ylabel(r"Kec. Linier $v_x$ (cm/s)")
    axes[1, 0].plot(t, df['ref_t_vy'] * 100, 'k--', label=r'$v_{y,ref}$'); axes[1, 0].plot(t, df['act_t_vy'] * 100, 'b-', label=r'$v_{y,act}$', alpha=0.7); axes[1, 0].set_ylabel(r"Kec. Linier $v_y$ (cm/s)")
    axes[2, 0].plot(t, df['ref_t_vz'] * 100, 'k--', label=r'$v_{z,ref}$'); axes[2, 0].plot(t, df['act_t_vz'] * 100, 'b-', label=r'$v_{z,act}$', alpha=0.7); axes[2, 0].set_ylabel(r"Kec. Linier $v_z$ (cm/s)"); axes[2, 0].set_xlabel(r"Waktu $t$ (s)")
    axes[0, 1].plot(t, np.degrees(df['ref_t_wx']), 'k--', label=r'$\omega_{x,ref}$'); axes[0, 1].plot(t, np.degrees(df['act_t_wx']), 'g-', label=r'$\omega_{x,act}$', alpha=0.7); axes[0, 1].set_ylabel(r"Kec. Sudut $\omega_x$ (°/s)")
    axes[1, 1].plot(t, np.degrees(df['ref_t_wy']), 'k--', label=r'$\omega_{y,ref}$'); axes[1, 1].plot(t, np.degrees(df['act_t_wy']), 'g-', label=r'$\omega_{y,act}$', alpha=0.7); axes[1, 1].set_ylabel(r"Kec. Sudut $\omega_y$ (°/s)")
    axes[2, 1].plot(t, np.degrees(df['ref_t_wz']), 'k--', label=r'$\omega_{z,ref}$'); axes[2, 1].plot(t, np.degrees(df['act_t_wz']), 'g-', label=r'$\omega_{z,act}$', alpha=0.7); axes[2, 1].set_ylabel(r"Kec. Sudut $\omega_z$ (°/s)"); axes[2, 1].set_xlabel(r"Waktu $t$ (s)")
    
    for ax in axes.flat: add_shading_and_fix_legend(ax, t_start, t_end)

    fig.suptitle(rf"Analisis Pelacakan Kinematik (Twist Target vs Aktual) - {run_name}", fontsize=16); fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_2a_twist_control_6dof.png", dpi=300); plt.close(fig)

def plot_twist_command_saturation_6dof(df, run_name, out_dir, t_start, t_end):
    if 'cmd_t_vx' not in df.columns: return
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    t = df['time_norm']
    
    axes[0, 0].plot(t, df['cmd_t_vx'] * 100, 'r-', label=r'$v_{cmd}$ (MPC)', alpha=0.4); axes[0, 0].plot(t, df['act_t_vx'] * 100, 'b-', label=r'$v_{act}$ (Robot)', alpha=0.9); axes[0, 0].set_ylabel(r"Kec. Linier $v_x$ (cm/s)")
    axes[1, 0].plot(t, df['cmd_t_vy'] * 100, 'r-', label=r'$v_{cmd}$ (MPC)', alpha=0.4); axes[1, 0].plot(t, df['act_t_vy'] * 100, 'b-', label=r'$v_{act}$ (Robot)', alpha=0.9); axes[1, 0].set_ylabel(r"Kec. Linier $v_y$ (cm/s)")
    axes[2, 0].plot(t, df['cmd_t_vz'] * 100, 'r-', label=r'$v_{cmd}$ (MPC)', alpha=0.4); axes[2, 0].plot(t, df['act_t_vz'] * 100, 'b-', label=r'$v_{act}$ (Robot)', alpha=0.9); axes[2, 0].set_ylabel(r"Kec. Linier $v_z$ (cm/s)"); axes[2, 0].set_xlabel(r"Waktu $t$ (s)")
    axes[0, 1].plot(t, np.degrees(df['cmd_t_wx']), 'r-', label=r'$\omega_{cmd}$ (MPC)', alpha=0.4); axes[0, 1].plot(t, np.degrees(df['act_t_wx']), 'g-', label=r'$\omega_{act}$ (Robot)', alpha=0.9); axes[0, 1].set_ylabel(r"Kec. Sudut $\omega_x$ (°/s)")
    axes[1, 1].plot(t, np.degrees(df['cmd_t_wy']), 'r-', label=r'$\omega_{cmd}$ (MPC)', alpha=0.4); axes[1, 1].plot(t, np.degrees(df['act_t_wy']), 'g-', label=r'$\omega_{act}$ (Robot)', alpha=0.9); axes[1, 1].set_ylabel(r"Kec. Sudut $\omega_y$ (°/s)")
    axes[2, 1].plot(t, np.degrees(df['cmd_t_wz']), 'r-', label=r'$\omega_{cmd}$ (MPC)', alpha=0.4); axes[2, 1].plot(t, np.degrees(df['act_t_wz']), 'g-', label=r'$\omega_{act}$ (Robot)', alpha=0.9); axes[2, 1].set_ylabel(r"Kec. Sudut $\omega_z$ (°/s)"); axes[2, 1].set_xlabel(r"Waktu $t$ (s)")
    
    for ax in axes.flat: add_shading_and_fix_legend(ax, t_start, t_end)

    fig.suptitle(rf"Evaluasi Limitasi Aktuator (Sinyal Kontrol MPC vs Respons Fisik) - {run_name}", fontsize=16); fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_2b_twist_saturation_6dof.png", dpi=300); plt.close(fig)

def plot_estimator_6dof(df, run_name, out_dir, t_start, t_end):
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    t = df['time_norm']
    
    axes[0, 0].plot(t, df['wb_t_raw_vx'] * 100, color='tab:orange', label=r'$v_{x,raw}$ (Derivatif)', alpha=0.5); axes[0, 0].plot(t, df['wb_t_eskf_vx'] * 100, color='tab:blue', label=r'$\hat{v}_x$ (LKF)'); axes[0, 0].set_ylabel(r"Kec. Linier $v_x$ (cm/s)")
    axes[1, 0].plot(t, df['wb_t_raw_vy'] * 100, color='tab:orange', label=r'$v_{y,raw}$ (Derivatif)', alpha=0.5); axes[1, 0].plot(t, df['wb_t_eskf_vy'] * 100, color='tab:blue', label=r'$\hat{v}_y$ (LKF)'); axes[1, 0].set_ylabel(r"Kec. Linier $v_y$ (cm/s)")
    axes[2, 0].plot(t, df['wb_t_raw_vz'] * 100, color='tab:orange', label=r'$v_{z,raw}$ (Derivatif)', alpha=0.5); axes[2, 0].plot(t, df['wb_t_eskf_vz'] * 100, color='tab:blue', label=r'$\hat{v}_z$ (LKF)'); axes[2, 0].set_ylabel(r"Kec. Linier $v_z$ (cm/s)"); axes[2, 0].set_xlabel(r"Waktu $t$ (s)")
    axes[0, 1].plot(t, np.degrees(df['wb_t_raw_wx']), color='tab:orange', label=r'$\omega_{x,raw}$ (Derivatif)', alpha=0.5); axes[0, 1].plot(t, np.degrees(df['wb_t_eskf_wx']), color='tab:green', label=r'$\hat{\omega}_x$ (ESKF)'); axes[0, 1].set_ylabel(r"Kec. Sudut $\omega_x$ (°/s)")
    axes[1, 1].plot(t, np.degrees(df['wb_t_raw_wy']), color='tab:orange', label=r'$\omega_{y,raw}$ (Derivatif)', alpha=0.5); axes[1, 1].plot(t, np.degrees(df['wb_t_eskf_wy']), color='tab:green', label=r'$\hat{\omega}_y$ (ESKF)'); axes[1, 1].set_ylabel(r"Kec. Sudut $\omega_y$ (°/s)")
    axes[2, 1].plot(t, np.degrees(df['wb_t_raw_wz']), color='tab:orange', label=r'$\omega_{z,raw}$ (Derivatif)', alpha=0.5); axes[2, 1].plot(t, np.degrees(df['wb_t_eskf_wz']), color='tab:green', label=r'$\hat{\omega}_z$ (ESKF)'); axes[2, 1].set_ylabel(r"Kec. Sudut $\omega_z$ (°/s)"); axes[2, 1].set_xlabel(r"Waktu $t$ (s)")
    
    for ax in axes.flat: add_shading_and_fix_legend(ax, t_start, t_end)

    fig.suptitle(rf"Kinerja Estimator Status (Derivatif Numerik vs Kalman Filter) - {run_name}", fontsize=16); fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_3_estimator_6dof.png", dpi=300); plt.close(fig)

def plot_overlay_correlation(df, run_name, out_dir, t_start=None, t_end=None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    t = df['time_norm']
    wb_vel_mag = np.sqrt(df['wb_t_eskf_vx']**2 + df['wb_t_eskf_vy']**2 + df['wb_t_eskf_vz']**2) * 100
    
    ax1 = axes[0]; color1 = 'tab:blue'
    ax1.set_ylabel(r'Kecepatan Linier Target $\|v_{ref}\|$ (cm/s)', color=color1)
    line1 = ax1.plot(t, wb_vel_mag, color=color1, label=r'$\|v_{ref}\|$ Target')
    ax1.tick_params(axis='y', labelcolor=color1); ax1.set_ylim(bottom=0)
    
    ax2 = ax1.twinx(); color2 = 'tab:red'
    ax2.set_ylabel(r'Galat Posisi 3D Euclidean $\|e_p\|$ (mm)', color=color2)
    line2 = ax2.plot(t, df['tracking_error_pos_mm'], color=color2, label=r'$\|e_p\|$ Posisi', alpha=0.8)
    
    if t_start is not None and t_end is not None:
        mask = (t >= t_start) & (t <= t_end)
        mean_err_pos = df.loc[mask, 'tracking_error_pos_mm'].mean()
        line2 += ax2.plot([t.min(), t.max()], [mean_err_pos, mean_err_pos], color=color2, linestyle='--', alpha=0.5, label=rf'Rata-rata $\|e_p\|$: {mean_err_pos:.2f} mm')

    ax2.tick_params(axis='y', labelcolor=color2); ax2.set_ylim(bottom=0)
    add_shading_and_fix_legend(ax1, t_start, t_end)
    ax1.legend(line1 + line2, [l.get_label() for l in line1 + line2], loc='upper right')
    ax1.set_title(r"Korelasi Translasi: Kecepatan Linier Target vs Galat Posisi 3D")
    
    wb_ang_mag = np.degrees(np.sqrt(df['wb_t_eskf_wx']**2 + df['wb_t_eskf_wy']**2 + df['wb_t_eskf_wz']**2))
    ax3 = axes[1]; color3 = 'tab:green'
    ax3.set_xlabel(r'Waktu $t$ (s)'); ax3.set_ylabel(r'Kecepatan Sudut Target $\|\omega_{ref}\|$ (°/s)', color=color3)
    line3 = ax3.plot(t, wb_ang_mag, color=color3, label=r'$\|\omega_{ref}\|$ Target')
    ax3.tick_params(axis='y', labelcolor=color3); ax3.set_ylim(bottom=0)
    
    ax4 = ax3.twinx(); color4 = 'tab:purple'
    ax4.set_ylabel(r'Galat Orientasi $\|\phi_e\|$ (°)', color=color4)
    line4 = ax4.plot(t, df['tracking_error_ori_deg'], color=color4, label=r'$\|\phi_e\|$ Orientasi', alpha=0.8)
    
    if t_start is not None and t_end is not None:
        mean_err_ori = df.loc[mask, 'tracking_error_ori_deg'].mean()
        line4 += ax4.plot([t.min(), t.max()], [mean_err_ori, mean_err_ori], color=color4, linestyle='--', alpha=0.5, label=rf'Rata-rata $\|\phi_e\|$: {mean_err_ori:.2f}°')

    ax4.tick_params(axis='y', labelcolor=color4); ax4.set_ylim(bottom=0)
    add_shading_and_fix_legend(ax3, t_start, t_end)
    ax3.legend(line3 + line4, [l.get_label() for l in line3 + line4], loc='upper right')
    ax3.set_title(r"Korelasi Rotasi: Kecepatan Sudut Target vs Galat Orientasi")
    
    fig.suptitle(rf"Analisis Kausalitas Cross-Coupling Dinamika Sistem - {run_name}", fontsize=16); fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_4_overlay_correlation.png", dpi=300); plt.close(fig)

def plot_joint_velocity_6dof(df, run_name, out_dir, t_start, t_end):
    if 'js_j1_v' not in df.columns: return
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    t = df['time_norm']
    
    axes[0, 0].plot(t, np.degrees(df['js_j1_v']), 'c-', label=r'$\dot{\theta}_1$'); axes[0, 0].set_ylabel(r"Kecepatan Sendi 1 (°/s)")
    axes[1, 0].plot(t, np.degrees(df['js_j2_v']), 'c-', label=r'$\dot{\theta}_2$'); axes[1, 0].set_ylabel(r"Kecepatan Sendi 2 (°/s)")
    axes[2, 0].plot(t, np.degrees(df['js_j3_v']), 'c-', label=r'$\dot{\theta}_3$'); axes[2, 0].set_ylabel(r"Kecepatan Sendi 3 (°/s)"); axes[2, 0].set_xlabel(r"Waktu $t$ (s)")
    axes[0, 1].plot(t, np.degrees(df['js_j4_v']), 'm-', label=r'$\dot{\theta}_4$'); axes[0, 1].set_ylabel(r"Kecepatan Sendi 4 (°/s)")
    axes[1, 1].plot(t, np.degrees(df['js_j5_v']), 'm-', label=r'$\dot{\theta}_5$'); axes[1, 1].set_ylabel(r"Kecepatan Sendi 5 (°/s)")
    axes[2, 1].plot(t, np.degrees(df['js_j6_v']), 'm-', label=r'$\dot{\theta}_6$'); axes[2, 1].set_ylabel(r"Kecepatan Sendi 6 (°/s)"); axes[2, 1].set_xlabel(r"Waktu $t$ (s)")
    
    for ax in axes.flat: add_shading_and_fix_legend(ax, t_start, t_end)

    fig.suptitle(rf"Evaluasi Ruang Konfigurasi (Joint Velocity) - {run_name}", fontsize=16); fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_5_joint_velocity_6dof.png", dpi=300); plt.close(fig)

def plot_dynamic_summary_trends():
    """ Menghasilkan grafik tren agregat (Galat vs Kecepatan) untuk Uji 5 dan Uji 7 """
    
    # 1. Tren Uji 5 (Hover Tracking)
    csv_uji5 = INPUT_BASE_DIR / "Summary_Uji_5_Point_Cart.csv"
    if csv_uji5.exists():
        df_u5 = pd.read_csv(csv_uji5)
        df_u5_mean = df_u5.groupby('Kecepatan_cm_s').mean(numeric_only=True).reset_index()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(df_u5_mean['Kecepatan_cm_s'], df_u5_mean['Pos_RMSE_mm'], 'b-o', label=r'Galat Posisi $RMSE_p$')
        ax.plot(df_u5_mean['Kecepatan_cm_s'], df_u5_mean['Pos_MaxErr_mm'], 'r--s', label=r'Galat Posisi Maksimal $\|e_p\|_{max}$')
        
        ax.set_title(r"Tren Galat Pelacakan Titik terhadap Laju Kecepatan Target (Uji 5)")
        ax.set_xlabel(r"Kecepatan Kereta Target $v_{feed}$ (cm/s)")
        ax.set_ylabel(r"Galat Spasial (mm)")
        ax.legend()
        fig.savefig(INPUT_BASE_DIR / "Plot_Trend_Uji5_Error_vs_Speed.png", dpi=300)
        plt.close(fig)
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.plot(df_u5_mean['Kecepatan_cm_s'], df_u5_mean['MPC_Phase_Lag_ms'], 'g-^', label=r'Keterlambatan Fasa ($\tau_{lag}$)')
        ax2.set_title(r"Tren Keterlambatan Fasa Sistem terhadap Laju Kecepatan Target (Uji 5)")
        ax2.set_xlabel(r"Kecepatan Kereta Target $v_{feed}$ (cm/s)")
        ax2.set_ylabel(r"Waktu Jeda $\tau_{lag}$ (ms)")
        ax2.legend()
        fig2.savefig(INPUT_BASE_DIR / "Plot_Trend_Uji5_PhaseLag_vs_Speed.png", dpi=300)
        plt.close(fig2)

    # 2. Tren Uji 7 (Trajectory Tracking)
    csv_uji7 = INPUT_BASE_DIR / "Summary_Uji_7_Trajectory_Cart.csv"
    if csv_uji7.exists():
        df_u7 = pd.read_csv(csv_uji7)
        
        # PERBAIKAN BUG KeyError: 'Shape'
        # Mengekstrak bentuk geometri langsung dari kolom Skenario
        df_u7['Shape'] = df_u7['Skenario'].apply(
            lambda x: 'circle' if 'circle' in x.lower() 
            else ('square' if 'square' in x.lower() else 'triangle')
        )
        
        df_u7_mean = df_u7.groupby(['Shape', 'Kecepatan_cm_s']).mean(numeric_only=True).reset_index()
        
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        for shape in df_u7_mean['Shape'].unique():
            subset = df_u7_mean[df_u7_mean['Shape'] == shape]
            shape_id = "Lingkaran" if shape == "circle" else "Persegi" if shape == "square" else "Segitiga"
            ax3.plot(subset['Kecepatan_cm_s'], subset['CV_Shape_RMSE_mm'], marker='o', label=rf'Geometri {shape_id}')
            
        ax3.set_title(r"Tren Degradasi Akurasi Geometri (CV) terhadap Kecepatan Target (Uji 7)")
        ax3.set_xlabel(r"Kecepatan Kereta Target $v_{feed}$ (cm/s)")
        ax3.set_ylabel(r"Galat Bentuk $RMSE_{shape}$ (mm)")
        ax3.legend()
        fig3.savefig(INPUT_BASE_DIR / "Plot_Trend_Uji7_CVShapeError_vs_Speed.png", dpi=300)
        plt.close(fig3)

def process_all_dynamic_timeseries():
    if not INPUT_BASE_DIR.exists(): return
    csv_files = []
    for pattern in ["Uji_4*/**/*.csv", "Uji_5*/**/*.csv", "Uji_6*/**/*.csv", "Uji_7*/**/*.csv"]:
        csv_files.extend(list(INPUT_BASE_DIR.glob(pattern)))
        
    total_files = len(csv_files)
    print(f"Menemukan {total_files} file CSV dinamis untuk divisualisasikan.\n")
    
    for idx, csv_path in enumerate(csv_files):
        run_name = csv_path.stem.replace("_timeseries", "")
        
        out_dir_full = csv_path.parent.parent / "plots" / run_name / "full_timeline"
        
        if (out_dir_full / f"{run_name}_4_overlay_correlation.png").exists():
            print(f"[{idx+1}/{total_files}] {run_name} -> SUDAH DIPROSES (Melewati)")
            continue
            
        print(f"[{idx+1}/{total_files}] Merender Plot: {run_name}")
        
        df = pd.read_csv(csv_path)
        df = calculate_dynamic_local_frame(df)
        
        t_start, t_end = get_drawing_phase_bounds(df, run_name)
        if t_start is not None and t_end is not None:
            df, t_start, t_end = trim_timeseries(df, t_start, t_end)
        
        out_dir_full.mkdir(parents=True, exist_ok=True)
        
        if "uji4" not in run_name.lower() and "uji5" not in run_name.lower():
            plot_trajectories_2d(df, run_name, out_dir_full)
            
        plot_pose_6dof(df, run_name, out_dir_full, t_start, t_end)
        plot_twist_control_6dof(df, run_name, out_dir_full, t_start, t_end)
        plot_twist_command_saturation_6dof(df, run_name, out_dir_full, t_start, t_end)
        plot_estimator_6dof(df, run_name, out_dir_full, t_start, t_end)
        plot_overlay_correlation(df, run_name, out_dir_full, t_start, t_end)
        plot_joint_velocity_6dof(df, run_name, out_dir_full, t_start, t_end)
        
        max_time = df['time_norm'].max()
        window_size = 20.0  
        
        if max_time > 30.0:
            out_dir_zoomed = csv_path.parent.parent / "plots" / run_name / "zoomed_slices"
            out_dir_zoomed.mkdir(parents=True, exist_ok=True)
            
            for slice_start in np.arange(0, max_time, window_size):
                slice_end = slice_start + window_size
                df_slice = df[(df['time_norm'] >= slice_start) & (df['time_norm'] < slice_end)]
                
                if len(df_slice) > 200: 
                    slice_name = f"{run_name}_zoom_{int(slice_start)}s_to_{int(slice_end)}s"
                    print(f"  -> Merender Irisan Zoom: {int(slice_start)}s - {int(slice_end)}s")
                    
                    if "uji4" not in run_name.lower() and "uji5" not in run_name.lower(): 
                        plot_trajectories_2d(df_slice, slice_name, out_dir_zoomed)
                        
                    plot_pose_6dof(df_slice, slice_name, out_dir_zoomed, None, None)
                    plot_twist_control_6dof(df_slice, slice_name, out_dir_zoomed, None, None)
                    plot_twist_command_saturation_6dof(df_slice, slice_name, out_dir_zoomed, None, None)
                    plot_estimator_6dof(df_slice, slice_name, out_dir_zoomed, None, None)
                    plot_overlay_correlation(df_slice, slice_name, out_dir_zoomed, None, None)
                    plot_joint_velocity_6dof(df_slice, slice_name, out_dir_zoomed, None, None)
    
    print("\nMelakukan agregasi grafik Summary Uji 5 & 7...")
    plot_dynamic_summary_trends()
    
    print("\nSeluruh Visualisasi selesai dan dilokalisasi dengan Notasi Akademik!")

if __name__ == "__main__":
    process_all_dynamic_timeseries()