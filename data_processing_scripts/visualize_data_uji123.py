#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from scipy.spatial.transform import Rotation as R
import warnings

warnings.filterwarnings('ignore')

# Konfigurasi gaya plot untuk format akademis (Mendukung notasi Math/LaTeX)
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm', # Computer Modern (Standar LaTeX)
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 2,
    'figure.autolayout': True,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.7
})

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_BASE_DIR = PROJECT_ROOT / "experiment_results"

def transform_to_local_frame(df):
    df_plot = df.copy()
    
    roll = df_plot['ref_pose_roll'].iloc[0]
    pitch = df_plot['ref_pose_pitch'].iloc[0]
    yaw = df_plot['ref_pose_yaw'].iloc[0]
    
    rot_matrix = R.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()
    rot_inv = rot_matrix.T
    
    ref_pts = np.vstack([df_plot['ref_pose_x'], df_plot['ref_pose_y'], df_plot['ref_pose_z']])
    act_pts = np.vstack([df_plot['actual_pose_x'], df_plot['actual_pose_y'], df_plot['actual_pose_z']])
    
    ref_rot = rot_inv @ ref_pts
    act_rot = rot_inv @ act_pts
    
    cx_local = (ref_rot[0, :].max() + ref_rot[0, :].min()) / 2.0
    cy_local = (ref_rot[1, :].max() + ref_rot[1, :].min()) / 2.0
    cz_local = ref_rot[2, :].mean() 
    
    df_plot['ref_plot_x'] = (ref_rot[0, :] - cx_local) * 1000
    df_plot['ref_plot_y'] = (ref_rot[1, :] - cy_local) * 1000
    df_plot['ref_plot_z'] = (ref_rot[2, :] - cz_local) * 1000
    
    df_plot['act_plot_x'] = (act_rot[0, :] - cx_local) * 1000
    df_plot['act_plot_y'] = (act_rot[1, :] - cy_local) * 1000
    df_plot['act_plot_z'] = (act_rot[2, :] - cz_local) * 1000
    
    return df_plot

def get_drawing_phase_bounds(df_local):
    z_min = df_local['ref_plot_z'].min()
    z_max = df_local['ref_plot_z'].max()
    z_range = z_max - z_min
    
    threshold = z_min + (0.15 * z_range)
    drawing_mask = df_local['ref_plot_z'] <= threshold
    
    if drawing_mask.any():
        t_start = df_local.loc[drawing_mask, 'time_norm'].min()
        t_end = df_local.loc[drawing_mask, 'time_norm'].max()
        return t_start, t_end
    return None, None

def trim_timeseries(df, df_local, t_start, t_end, pre_margin=2.0, post_margin=2.0):
    if t_start is None or t_end is None:
        return df, df_local, t_start, t_end
    
    cut_start = max(0, t_start - pre_margin)
    cut_end = t_end + post_margin
    
    df_trimmed = df[(df['time_norm'] >= cut_start) & (df['time_norm'] <= cut_end)].copy()
    df_local_trimmed = df_local[(df_local['time_norm'] >= cut_start) & (df_local['time_norm'] <= cut_end)].copy()
    
    if df_trimmed.empty:
        return df, df_local, t_start, t_end
        
    time_offset = df_trimmed['time_norm'].iloc[0]
    df_trimmed['time_norm'] = df_trimmed['time_norm'] - time_offset
    df_local_trimmed['time_norm'] = df_local_trimmed['time_norm'] - time_offset
    
    new_t_start = t_start - time_offset
    new_t_end = t_end - time_offset
    
    return df_trimmed, df_local_trimmed, new_t_start, new_t_end

def plot_2d_trajectory(df_plot, run_name, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(df_plot['ref_plot_x'], df_plot['ref_plot_y'], 'k--', label=r'$p_{ref}$ (Target)', zorder=2)
    ax.plot(df_plot['act_plot_x'], df_plot['act_plot_y'], 'b-', label=r'$p_{act}$ (Aktual)', alpha=0.7, zorder=1)
    
    ax.set_title(rf"Trajektori Spasial Relatif (Kerangka Target) - {run_name}")
    ax.set_xlabel(r"Posisi Lokal $X$ (mm)")
    ax.set_ylabel(r"Posisi Lokal $Y$ (mm)")
    ax.legend()
    ax.axis('equal') 
    
    fig.savefig(out_dir / f"{run_name}_1_trajectory_2d.png", dpi=300)
    plt.close(fig)

def plot_transient_pose(df_plot, run_name, out_dir, t_start, t_end):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t = df_plot['time_norm']
    
    axes[0].plot(t, df_plot['ref_plot_x'], 'k--', label=r'$x_{ref}$')
    axes[0].plot(t, df_plot['act_plot_x'], 'b-', label=r'$x_{act}$', alpha=0.7)
    axes[0].set_ylabel(r"Posisi $X$ (mm)")
    axes[0].legend(loc='upper right')
    
    axes[1].plot(t, df_plot['ref_plot_y'], 'k--', label=r'$y_{ref}$')
    axes[1].plot(t, df_plot['act_plot_y'], 'g-', label=r'$y_{act}$', alpha=0.7)
    axes[1].set_ylabel(r"Posisi $Y$ (mm)")
    axes[1].legend(loc='upper right')
    
    axes[2].plot(t, df_plot['ref_plot_z'], 'k--', label=r'$z_{ref}$')
    axes[2].plot(t, df_plot['act_plot_z'], 'r-', label=r'$z_{act}$', alpha=0.7)
    axes[2].set_ylabel(r"Posisi $Z$ (mm)")
    axes[2].set_xlabel(r"Waktu $t$ (s)")
    axes[2].legend(loc='upper right')
    
    if t_start is not None and t_end is not None:
        for ax in axes:
            ax.axvline(t_start, color='gray', linestyle=':', linewidth=1.5)
            ax.axvline(t_end, color='gray', linestyle=':', linewidth=1.5)
            ax.axvspan(t_start, t_end, color='#00FF00', alpha=0.08, label='Fase Eksekusi Geometri')
            
        handles, labels = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[0].legend(by_label.values(), by_label.keys(), loc='upper right')

    axes[0].set_title(rf"Respons Transien Posisi Lokal - {run_name}")
    
    fig.savefig(out_dir / f"{run_name}_2_transient_pose.png", dpi=300)
    plt.close(fig)

def plot_transient_twist(df, run_name, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t = df['time_norm']
    
    axes[0].plot(t, df['ref_twist_vx'] * 100, 'k--', label=r'$v_{x,ref}$')
    axes[0].plot(t, df['actual_twist_vx'] * 100, 'b-', label=r'$v_{x,act}$', alpha=0.7)
    axes[0].set_ylabel(r"Kec. Linier $v_x$ (cm/s)")
    axes[0].legend(loc='upper right')
    
    axes[1].plot(t, df['ref_twist_vy'] * 100, 'k--', label=r'$v_{y,ref}$')
    axes[1].plot(t, df['actual_twist_vy'] * 100, 'g-', label=r'$v_{y,act}$', alpha=0.7)
    axes[1].set_ylabel(r"Kec. Linier $v_y$ (cm/s)")
    axes[1].set_xlabel(r"Waktu $t$ (s)")
    axes[1].legend(loc='upper right')
    
    axes[0].set_title(rf"Profil Kecepatan Kinematik (Kerangka Basis) - {run_name}")
    
    fig.savefig(out_dir / f"{run_name}_3_transient_twist.png", dpi=300)
    plt.close(fig)

def plot_error_over_time(df, run_name, out_dir, t_start, t_end):
    fig, ax = plt.subplots(figsize=(10, 4))
    t = df['time_norm']
    
    ax.plot(t, df['error_3d_mm'], 'r-', label=r'Galat Posisi 3D Euclidean $\|e_p\|$')
    
    if t_start is not None and t_end is not None:
        ax.axvline(t_start, color='gray', linestyle=':', linewidth=1.5)
        ax.axvline(t_end, color='gray', linestyle=':', linewidth=1.5)
        ax.axvspan(t_start, t_end, color='#00FF00', alpha=0.08, label='Fase Eksekusi Geometri')
        
        drawing_mask = (t >= t_start) & (t <= t_end)
        mean_error = df.loc[drawing_mask, 'error_3d_mm'].mean()
        ax.axhline(mean_error, color='k', linestyle='--', label=rf"Rata-rata (Fase Eksekusi): {mean_error:.2f} mm")
    else:
        ax.axhline(df['error_3d_mm'].mean(), color='k', linestyle='--', label=rf"Rata-rata Keseluruhan: {df['error_3d_mm'].mean():.2f} mm")
    
    ax.set_title(rf"Perkembangan Galat Pelacakan Spasial - {run_name}")
    ax.set_xlabel(r"Waktu $t$ (s)")
    ax.set_ylabel(r"Galat Spasial $\|e_p\|$ (mm)")
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=0)
    
    fig.savefig(out_dir / f"{run_name}_4_error_vs_time.png", dpi=300)
    plt.close(fig)

def process_all_timeseries():
    if not INPUT_BASE_DIR.exists():
        print(f"Folder '{INPUT_BASE_DIR}' tidak ditemukan. Pastikan path-nya benar.")
        return

    csv_files = list(INPUT_BASE_DIR.rglob("timeseries/*.csv"))
    # Filter hanya untuk Uji 1, 2, 3
    csv_files = [f for f in csv_files if any(x in f.parent.parent.name for x in ["Uji_1", "Uji_2", "Uji_3"])]
    
    print(f"Ditemukan {len(csv_files)} file timeseries Uji 1/2/3 untuk diplot.")
    
    for idx, csv_path in enumerate(csv_files):
        run_name = csv_path.stem.replace("_timeseries", "")
        print(f"[{idx+1}/{len(csv_files)}] Membuat plot untuk: {run_name}")
        
        df = pd.read_csv(csv_path)
        df['time_norm'] = df['time'] - df['time'].iloc[0]
        
        df_local = transform_to_local_frame(df)
        t_start, t_end = get_drawing_phase_bounds(df_local)
        
        if t_start is not None and t_end is not None:
            df, df_local, t_start, t_end = trim_timeseries(df, df_local, t_start, t_end, pre_margin=2.0, post_margin=2.0)
        
        out_dir = csv_path.parent.parent / "plots" / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        plot_2d_trajectory(df_local, run_name, out_dir)
        plot_transient_pose(df_local, run_name, out_dir, t_start, t_end)
        plot_transient_twist(df, run_name, out_dir)
        plot_error_over_time(df, run_name, out_dir, t_start, t_end)

def plot_summary_trends():
    speed_csv = INPUT_BASE_DIR / "Summary_Uji_2_Speed.csv"
    if speed_csv.exists():
        df_speed = pd.read_csv(speed_csv)
        df_mean = df_speed.groupby(['Shape', 'Speed_cm_s']).mean(numeric_only=True).reset_index()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        for shape in df_mean['Shape'].unique():
            subset = df_mean[df_mean['Shape'] == shape]
            shape_id = "Lingkaran" if shape == "circle" else "Persegi" if shape == "square" else "Segitiga"
            ax.plot(subset['Speed_cm_s'], subset['CV_Shape_RMSE_mm'], marker='o', label=rf'Geometri {shape_id}')
            
        ax.set_title(r"Tren Degradasi Akurasi Geometri (CV) terhadap Kecepatan Pemakanan (Uji 2)")
        ax.set_xlabel(r"Kecepatan Eksekusi $v_{feed}$ (cm/s)")
        ax.set_ylabel(r"Galat Bentuk $RMSE_{shape}$ (mm)")
        ax.legend()
        fig.savefig(INPUT_BASE_DIR / "Plot_Trend_Uji2_Speed.png", dpi=300)
        plt.close(fig)

    angle_csv = INPUT_BASE_DIR / "Summary_Uji_3_Angle.csv"
    if angle_csv.exists():
        df_angle = pd.read_csv(angle_csv)
        df_angle_mean = df_angle.groupby('Pitch').mean(numeric_only=True).reset_index()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(df_angle_mean['Pitch'], df_angle_mean['CV_Offset_mm'], 'r-o', label=r'Defleksi Posisi Absolut $\|e_{offset}\|$')
        ax.plot(df_angle_mean['Pitch'], df_angle_mean['CV_Shape_RMSE_mm'], 'b-s', label=r'Galat Bentuk $RMSE_{shape}$')
        
        ax.set_title(r"Tren Defleksi dan Galat Bentuk terhadap Sudut Kemiringan Papan (Uji 3)")
        ax.set_xlabel(r"Sudut Kemiringan Papan $\theta$ (°)")
        ax.set_ylabel(r"Galat Geometri (mm)")
        ax.legend()
        fig.savefig(INPUT_BASE_DIR / "Plot_Trend_Uji3_Angle.png", dpi=300)
        plt.close(fig)

if __name__ == "__main__":
    process_all_timeseries()
    plot_summary_trends()
    print("\nSeluruh plot Uji 1-3 berhasil digenerate dengan Notasi Akademik!")