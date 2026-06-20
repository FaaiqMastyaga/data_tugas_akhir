#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from scipy.spatial.transform import Rotation as R

# Konfigurasi gaya plot untuk format akademis
plt.rcParams.update({
    'font.family': 'serif',
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

INPUT_BASE_DIR = Path(__file__).parent.parent / "experiment_results"

def transform_to_local_frame(df):
    """
    Mentransformasi koordinat dari Base Frame ke Local Frame (Whiteboard).
    Menerapkan Matriks Rotasi dan Translasi Bounding Box Geometris murni.
    """
    df_plot = df.copy()
    
    # 1. Ambil orientasi bidang papan dari target referensi
    roll = df_plot['ref_pose_roll'].iloc[0]
    pitch = df_plot['ref_pose_pitch'].iloc[0]
    yaw = df_plot['ref_pose_yaw'].iloc[0]
    
    # Matriks Rotasi (Base -> Board) dan Invers-nya
    rot_matrix = R.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()
    rot_inv = rot_matrix.T
    
    # 2. Susun titik sebagai matriks vektor murni tanpa translasi
    ref_pts = np.vstack([df_plot['ref_pose_x'], df_plot['ref_pose_y'], df_plot['ref_pose_z']])
    act_pts = np.vstack([df_plot['actual_pose_x'], df_plot['actual_pose_y'], df_plot['actual_pose_z']])
    
    # 3. Terapkan Matriks Rotasi terlebih dahulu agar sumbu sejajar dengan bidang papan
    ref_rot = rot_inv @ ref_pts
    act_rot = rot_inv @ act_pts
    
    # 4. Cari Titik Pusat Geometris (Translasi Vector t), BUKAN Temporal Mean
    cx_local = (ref_rot[0, :].max() + ref_rot[0, :].min()) / 2.0
    cy_local = (ref_rot[1, :].max() + ref_rot[1, :].min()) / 2.0
    cz_local = ref_rot[2, :].mean() # Z tidak terlalu kritis untuk plot 2D
    
    # 5. Terapkan Matriks Translasi dan konversi ke milimeter
    df_plot['ref_plot_x'] = (ref_rot[0, :] - cx_local) * 1000
    df_plot['ref_plot_y'] = (ref_rot[1, :] - cy_local) * 1000
    df_plot['ref_plot_z'] = (ref_rot[2, :] - cz_local) * 1000
    
    df_plot['act_plot_x'] = (act_rot[0, :] - cx_local) * 1000
    df_plot['act_plot_y'] = (act_rot[1, :] - cy_local) * 1000
    df_plot['act_plot_z'] = (act_rot[2, :] - cz_local) * 1000
    
    return df_plot

def get_drawing_phase_bounds(df):
    """
    Mendeteksi waktu mulai dan selesai fase menggambar murni berdasarkan
    kapan sumbu Z referensi menekan papan tulis (berada di titik terendahnya).
    """
    z_min = df['ref_pose_z'].min()
    # Masking: Fase aktif adalah ketika Z berada dalam toleransi 2 mm dari titik terendah (nempel papan)
    drawing_mask = df['ref_pose_z'] <= (z_min + 0.002)
    
    if drawing_mask.any():
        t_start = df.loc[drawing_mask, 'time_norm'].min()
        t_end = df.loc[drawing_mask, 'time_norm'].max()
        return t_start, t_end
    return None, None

def plot_2d_trajectory(df_plot, run_name, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(df_plot['ref_plot_x'], df_plot['ref_plot_y'], 'k--', label='Referensi (Target)', zorder=2)
    ax.plot(df_plot['act_plot_x'], df_plot['act_plot_y'], 'b-', label='Aktual (Robot)', alpha=0.7, zorder=1)
    
    ax.set_title(f"Trajektori Spasial 2D (Whiteboard Frame) - {run_name}")
    ax.set_xlabel("Local X (mm)")
    ax.set_ylabel("Local Y (mm)")
    ax.legend()
    ax.axis('equal') 
    
    fig.savefig(out_dir / f"{run_name}_1_trajectory_2d.png", dpi=300)
    plt.close(fig)

def plot_transient_pose(df_plot, run_name, out_dir):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t = df_plot['time_norm']
    
    # Dapatkan batas fase menggambar
    t_start, t_end = get_drawing_phase_bounds(df_plot)
    
    # Sumbu X
    axes[0].plot(t, df_plot['ref_plot_x'], 'k--', label='Ref Local X')
    axes[0].plot(t, df_plot['act_plot_x'], 'b-', label='Actual Local X', alpha=0.7)
    axes[0].set_ylabel("Posisi X (mm)")
    axes[0].legend(loc='upper right')
    
    # Sumbu Y
    axes[1].plot(t, df_plot['ref_plot_y'], 'k--', label='Ref Local Y')
    axes[1].plot(t, df_plot['act_plot_y'], 'g-', label='Actual Local Y', alpha=0.7)
    axes[1].set_ylabel("Posisi Y (mm)")
    axes[1].legend(loc='upper right')
    
    # Sumbu Z
    axes[2].plot(t, df_plot['ref_plot_z'], 'k--', label='Ref Local Z')
    axes[2].plot(t, df_plot['act_plot_z'], 'r-', label='Actual Local Z', alpha=0.7)
    axes[2].set_ylabel("Posisi Z (mm)")
    axes[2].set_xlabel("Waktu (s)")
    axes[2].legend(loc='upper right')
    
    # Suntikkan anotasi Fase Menggambar jika terdeteksi
    if t_start is not None and t_end is not None:
        for ax in axes:
            ax.axvline(t_start, color='gray', linestyle=':', linewidth=1.5)
            ax.axvline(t_end, color='gray', linestyle=':', linewidth=1.5)
            ax.axvspan(t_start, t_end, color='#00FF00', alpha=0.08, label='Fase Gambar')
            
        # Hindari duplikasi label legend untuk axvspan
        handles, labels = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[0].legend(by_label.values(), by_label.keys(), loc='upper right')

    axes[0].set_title(f"Respons Transien Posisi (Whiteboard Frame) - {run_name}")
    
    fig.savefig(out_dir / f"{run_name}_2_transient_pose.png", dpi=300)
    plt.close(fig)

def plot_transient_twist(df, run_name, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t = df['time_norm']
    
    axes[0].plot(t, df['ref_twist_vx'] * 100, 'k--', label='Ref Vx')
    axes[0].plot(t, df['actual_twist_vx'] * 100, 'b-', label='Actual Vx', alpha=0.7)
    axes[0].set_ylabel("Kecepatan X (cm/s)")
    axes[0].legend(loc='upper right')
    
    axes[1].plot(t, df['ref_twist_vy'] * 100, 'k--', label='Ref Vy')
    axes[1].plot(t, df['actual_twist_vy'] * 100, 'g-', label='Actual Vy', alpha=0.7)
    axes[1].set_ylabel("Kecepatan Y (cm/s)")
    axes[1].set_xlabel("Waktu (s)")
    axes[1].legend(loc='upper right')
    
    axes[0].set_title(f"Profil Kecepatan Linear (Base Frame) - {run_name}")
    
    fig.savefig(out_dir / f"{run_name}_3_transient_twist.png", dpi=300)
    plt.close(fig)

def plot_error_over_time(df, run_name, out_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    t = df['time_norm']
    
    t_start, t_end = get_drawing_phase_bounds(df)
    
    ax.plot(t, df['error_3d_mm'], 'r-', label='Galat Euclidean 3D')
    
    if t_start is not None and t_end is not None:
        ax.axvline(t_start, color='gray', linestyle=':', linewidth=1.5)
        ax.axvline(t_end, color='gray', linestyle=':', linewidth=1.5)
        ax.axvspan(t_start, t_end, color='#00FF00', alpha=0.08, label='Fase Gambar')
        
        # Hitung rata-rata galat HANYA pada fase menggambar
        drawing_mask = (t >= t_start) & (t <= t_end)
        mean_error = df.loc[drawing_mask, 'error_3d_mm'].mean()
        ax.axhline(mean_error, color='k', linestyle='--', label=f"Mean (Drawing Phase): {mean_error:.2f} mm")
    else:
        ax.axhline(df['error_3d_mm'].mean(), color='k', linestyle='--', label=f"Mean (All): {df['error_3d_mm'].mean():.2f} mm")
    
    ax.set_title(f"Perkembangan Galat Pelacakan (Tracking Error) - {run_name}")
    ax.set_xlabel("Waktu (s)")
    ax.set_ylabel("Galat (mm)")
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=0)
    
    fig.savefig(out_dir / f"{run_name}_4_error_vs_time.png", dpi=300)
    plt.close(fig)

def process_all_timeseries():
    if not INPUT_BASE_DIR.exists():
        print(f"Folder '{INPUT_BASE_DIR}' tidak ditemukan. Pastikan path-nya benar.")
        return

    csv_files = list(INPUT_BASE_DIR.rglob("timeseries/*.csv"))
    print(f"Ditemukan {len(csv_files)} file timeseries untuk diplot.")
    
    for idx, csv_path in enumerate(csv_files):
        run_name = csv_path.stem.replace("_timeseries", "")
        print(f"[{idx+1}/{len(csv_files)}] Membuat plot untuk: {run_name}")
        
        df = pd.read_csv(csv_path)
        df['time_norm'] = df['time'] - df['time'].iloc[0]
        
        # Eksekusi Transformasi ke Local Frame (Orientasi + Translasi)
        df_local = transform_to_local_frame(df)
        
        out_dir = csv_path.parent.parent / "plots" / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot lintasan ruang menggunakan dataframe frame lokal
        plot_2d_trajectory(df_local, run_name, out_dir)
        plot_transient_pose(df_local, run_name, out_dir)
        
        # Plot kontrol/metrik Euclidean menggunakan dataframe asli (karena invarian/absolut)
        plot_transient_twist(df, run_name, out_dir)
        plot_error_over_time(df, run_name, out_dir)

def plot_summary_trends():
    speed_csv = INPUT_BASE_DIR / "Summary_Uji_2_Speed.csv"
    if speed_csv.exists():
        df_speed = pd.read_csv(speed_csv)
        df_mean = df_speed.groupby(['Shape', 'Speed_cm_s']).mean(numeric_only=True).reset_index()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        for shape in df_mean['Shape'].unique():
            subset = df_mean[df_mean['Shape'] == shape]
            ax.plot(subset['Speed_cm_s'], subset['CV_Shape_RMSE_mm'], marker='o', label=shape.capitalize())
            
        ax.set_title("Pengaruh Kecepatan Trajektori terhadap Kualitas Bentuk (Shape RMSE)")
        ax.set_xlabel("Kecepatan (cm/s)")
        ax.set_ylabel("CV Shape RMSE (mm)")
        ax.legend()
        fig.savefig(INPUT_BASE_DIR / "Plot_Trend_Uji2_Speed.png", dpi=300)
        plt.close(fig)

    angle_csv = INPUT_BASE_DIR / "Summary_Uji_3_Angle.csv"
    if angle_csv.exists():
        df_angle = pd.read_csv(angle_csv)
        df_angle_mean = df_angle.groupby('Pitch').mean(numeric_only=True).reset_index()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(df_angle_mean['Pitch'], df_angle_mean['CV_Offset_mm'], 'r-o', label='Defleksi Trajektori (Offset)')
        ax.plot(df_angle_mean['Pitch'], df_angle_mean['CV_Shape_RMSE_mm'], 'b-s', label='Galat Bentuk (Shape RMSE)')
        
        ax.set_title("Pengaruh Kemiringan Papan terhadap Defleksi dan Galat Bentuk")
        ax.set_xlabel("Sudut Pitch (Derajat)")
        ax.set_ylabel("Galat CV (mm)")
        ax.legend()
        fig.savefig(INPUT_BASE_DIR / "Plot_Trend_Uji3_Angle.png", dpi=300)
        plt.close(fig)

if __name__ == "__main__":
    process_all_timeseries()
    plot_summary_trends()
    print("\nSeluruh plot telah berhasil di-generate dengan Local Frame Transformation!")