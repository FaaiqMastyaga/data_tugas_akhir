#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import warnings

warnings.filterwarnings('ignore')

# Konfigurasi gaya plot akademis
plt.rcParams.update({
    'font.family': 'serif',
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
        
        ref_rel = ref_pos[i] - wb_pos[i]
        act_rel = act_pos[i] - wb_pos[i]
        
        ref_local = rot_inv.apply(ref_rel) * 1000.0
        act_local = rot_inv.apply(act_rel) * 1000.0
        
        ref_loc_x.append(ref_local[0]); ref_loc_y.append(ref_local[1]); ref_loc_z.append(ref_local[2])
        act_loc_x.append(act_local[0]); act_loc_y.append(act_local[1]); act_loc_z.append(act_local[2])
        
    df['ref_loc_x'] = ref_loc_x; df['ref_loc_y'] = ref_loc_y; df['ref_loc_z'] = ref_loc_z
    df['act_loc_x'] = act_loc_x; df['act_loc_y'] = act_loc_y; df['act_loc_z'] = act_loc_z
    
    cx = (np.max(ref_loc_x) + np.min(ref_loc_x)) / 2.0
    cy = (np.max(ref_loc_y) + np.min(ref_loc_y)) / 2.0
    df['ref_loc_x'] -= cx; df['act_loc_x'] -= cx
    df['ref_loc_y'] -= cy; df['act_loc_y'] -= cy
    
    return df

def plot_trajectories_2d(df, run_name, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].plot(df['ref_p_x'] * 1000, df['ref_p_y'] * 1000, 'k--', label='Referensi (Global)', zorder=2)
    axes[0].plot(df['act_p_x'] * 1000, df['act_p_y'] * 1000, 'b-', label='Aktual Robot (Global)', alpha=0.7, zorder=1)
    axes[0].set_title(f"Trajektori Absolut (Base Frame) - {run_name}")
    axes[0].set_xlabel("Global X (mm)")
    axes[0].set_ylabel("Global Y (mm)")
    axes[0].legend()
    axes[0].axis('equal')
    
    axes[1].plot(df['ref_loc_x'], df['ref_loc_y'], 'k--', label='Referensi (Di Papan)', zorder=2)
    axes[1].plot(df['act_loc_x'], df['act_loc_y'], 'g-', label='Aktual Tinta (Di Papan)', alpha=0.7, zorder=1)
    axes[1].set_title(f"Trajektori Relatif (Whiteboard Frame) - {run_name}")
    axes[1].set_xlabel("Local X (mm)")
    axes[1].set_ylabel("Local Y (mm)")
    axes[1].legend()
    axes[1].axis('equal')
    
    fig.savefig(out_dir / f"{run_name}_0_trajectory_2d.png", dpi=300)
    plt.close(fig)

def plot_pose_6dof(df, run_name, out_dir):
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    t = df['time_norm']
    
    axes[0, 0].plot(t, df['ref_p_x'] * 1000, 'k--', label='Ref X'); axes[0, 0].plot(t, df['act_p_x'] * 1000, 'b-', label='Act X', alpha=0.7)
    axes[0, 0].set_ylabel("Posisi X (mm)"); axes[0, 0].legend(loc='upper right')
    
    axes[1, 0].plot(t, df['ref_p_y'] * 1000, 'k--', label='Ref Y'); axes[1, 0].plot(t, df['act_p_y'] * 1000, 'b-', label='Act Y', alpha=0.7)
    axes[1, 0].set_ylabel("Posisi Y (mm)"); axes[1, 0].legend(loc='upper right')
    
    axes[2, 0].plot(t, df['ref_p_z'] * 1000, 'k--', label='Ref Z'); axes[2, 0].plot(t, df['act_p_z'] * 1000, 'b-', label='Act Z', alpha=0.7)
    axes[2, 0].set_ylabel("Posisi Z (mm)"); axes[2, 0].set_xlabel("Waktu (s)"); axes[2, 0].legend(loc='upper right')
    
    axes[0, 1].plot(t, np.degrees(df['ref_p_roll']), 'k--', label='Ref Roll'); axes[0, 1].plot(t, np.degrees(df['act_p_roll']), 'g-', label='Act Roll', alpha=0.7)
    axes[0, 1].set_ylabel("Sudut Roll (deg)"); axes[0, 1].legend(loc='upper right')
    
    axes[1, 1].plot(t, np.degrees(df['ref_p_pitch']), 'k--', label='Ref Pitch'); axes[1, 1].plot(t, np.degrees(df['act_p_pitch']), 'g-', label='Act Pitch', alpha=0.7)
    axes[1, 1].set_ylabel("Sudut Pitch (deg)"); axes[1, 1].legend(loc='upper right')
    
    axes[2, 1].plot(t, np.degrees(df['ref_p_yaw']), 'k--', label='Ref Yaw'); axes[2, 1].plot(t, np.degrees(df['act_p_yaw']), 'g-', label='Act Yaw', alpha=0.7)
    axes[2, 1].set_ylabel("Sudut Yaw (deg)"); axes[2, 1].set_xlabel("Waktu (s)"); axes[2, 1].legend(loc='upper right')

    fig.suptitle(f"Analisis Pelacakan Pose 6-DOF (Base Frame) - {run_name}", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_1_pose_tracking_6dof.png", dpi=300)
    plt.close(fig)

def plot_twist_control_6dof(df, run_name, out_dir):
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    t = df['time_norm']
    
    axes[0, 0].plot(t, df['ref_t_vx'] * 100, 'k--', label='Ref Vx'); axes[0, 0].plot(t, df['act_t_vx'] * 100, 'b-', label='Act Vx', alpha=0.7)
    axes[0, 0].set_ylabel("Vx (cm/s)"); axes[0, 0].legend(loc='upper right')
    
    axes[1, 0].plot(t, df['ref_t_vy'] * 100, 'k--', label='Ref Vy'); axes[1, 0].plot(t, df['act_t_vy'] * 100, 'b-', label='Act Vy', alpha=0.7)
    axes[1, 0].set_ylabel("Vy (cm/s)"); axes[1, 0].legend(loc='upper right')
    
    axes[2, 0].plot(t, df['ref_t_vz'] * 100, 'k--', label='Ref Vz'); axes[2, 0].plot(t, df['act_t_vz'] * 100, 'b-', label='Act Vz', alpha=0.7)
    axes[2, 0].set_ylabel("Vz (cm/s)"); axes[2, 0].set_xlabel("Waktu (s)"); axes[2, 0].legend(loc='upper right')
    
    axes[0, 1].plot(t, np.degrees(df['ref_t_wx']), 'k--', label='Ref Wx'); axes[0, 1].plot(t, np.degrees(df['act_t_wx']), 'g-', label='Act Wx', alpha=0.7)
    axes[0, 1].set_ylabel("Wx (deg/s)"); axes[0, 1].legend(loc='upper right')
    
    axes[1, 1].plot(t, np.degrees(df['ref_t_wy']), 'k--', label='Ref Wy'); axes[1, 1].plot(t, np.degrees(df['act_t_wy']), 'g-', label='Act Wy', alpha=0.7)
    axes[1, 1].set_ylabel("Wy (deg/s)"); axes[1, 1].legend(loc='upper right')
    
    axes[2, 1].plot(t, np.degrees(df['ref_t_wz']), 'k--', label='Ref Wz'); axes[2, 1].plot(t, np.degrees(df['act_t_wz']), 'g-', label='Act Wz', alpha=0.7)
    axes[2, 1].set_ylabel("Wz (deg/s)"); axes[2, 1].set_xlabel("Waktu (s)"); axes[2, 1].legend(loc='upper right')

    fig.suptitle(f"Analisis Kontrol Kecepatan 6-DOF (Twist) - {run_name}", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_2_twist_control_6dof.png", dpi=300)
    plt.close(fig)

def plot_estimator_6dof(df, run_name, out_dir):
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    t = df['time_norm']
    
    axes[0, 0].plot(t, df['wb_t_raw_vx'] * 100, color='tab:orange', label='Raw Vx', alpha=0.5); axes[0, 0].plot(t, df['wb_t_eskf_vx'] * 100, color='tab:blue', label='Filtered Vx')
    axes[0, 0].set_ylabel("Vx (cm/s)"); axes[0, 0].legend(loc='upper right')
    
    axes[1, 0].plot(t, df['wb_t_raw_vy'] * 100, color='tab:orange', label='Raw Vy', alpha=0.5); axes[1, 0].plot(t, df['wb_t_eskf_vy'] * 100, color='tab:blue', label='Filtered Vy')
    axes[1, 0].set_ylabel("Vy (cm/s)"); axes[1, 0].legend(loc='upper right')
    
    axes[2, 0].plot(t, df['wb_t_raw_vz'] * 100, color='tab:orange', label='Raw Vz', alpha=0.5); axes[2, 0].plot(t, df['wb_t_eskf_vz'] * 100, color='tab:blue', label='Filtered Vz')
    axes[2, 0].set_ylabel("Vz (cm/s)"); axes[2, 0].set_xlabel("Waktu (s)"); axes[2, 0].legend(loc='upper right')
    
    axes[0, 1].plot(t, np.degrees(df['wb_t_raw_wx']), color='tab:orange', label='Raw Wx', alpha=0.5); axes[0, 1].plot(t, np.degrees(df['wb_t_eskf_wx']), color='tab:green', label='Filtered Wx')
    axes[0, 1].set_ylabel("Wx (deg/s)"); axes[0, 1].legend(loc='upper right')
    
    axes[1, 1].plot(t, np.degrees(df['wb_t_raw_wy']), color='tab:orange', label='Raw Wy', alpha=0.5); axes[1, 1].plot(t, np.degrees(df['wb_t_eskf_wy']), color='tab:green', label='Filtered Wy')
    axes[1, 1].set_ylabel("Wy (deg/s)"); axes[1, 1].legend(loc='upper right')
    
    axes[2, 1].plot(t, np.degrees(df['wb_t_raw_wz']), color='tab:orange', label='Raw Wz', alpha=0.5); axes[2, 1].plot(t, np.degrees(df['wb_t_eskf_wz']), color='tab:green', label='Filtered Wz')
    axes[2, 1].set_ylabel("Wz (deg/s)"); axes[2, 1].set_xlabel("Waktu (s)"); axes[2, 1].legend(loc='upper right')

    fig.suptitle(f"Kinerja Estimator State 6-DOF (Raw Derivatif vs Kalman Filter) - {run_name}", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_3_estimator_6dof.png", dpi=300)
    plt.close(fig)

def plot_overlay_correlation(df, run_name, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    t = df['time_norm']
    
    wb_vel_mag = np.sqrt(df['wb_t_eskf_vx']**2 + df['wb_t_eskf_vy']**2 + df['wb_t_eskf_vz']**2) * 100
    
    ax1 = axes[0]
    color1 = 'tab:blue'
    ax1.set_ylabel('Kecepatan Linier Papan (cm/s)', color=color1)
    line1 = ax1.plot(t, wb_vel_mag, color=color1, label='Kecepatan Linier Target')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(bottom=0)
    
    ax2 = ax1.twinx()  
    color2 = 'tab:red'
    ax2.set_ylabel('Euclidean Error 3D (mm)', color=color2)
    line2 = ax2.plot(t, df['tracking_error_pos_mm'], color=color2, label='Error Posisi', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(bottom=0)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    ax1.set_title("Korelasi Translasi: Kecepatan Linier vs Galat Posisi 3D")
    
    wb_ang_mag = np.degrees(np.sqrt(df['wb_t_eskf_wx']**2 + df['wb_t_eskf_wy']**2 + df['wb_t_eskf_wz']**2))
    
    ax3 = axes[1]
    color3 = 'tab:green'
    ax3.set_xlabel('Waktu (s)')
    ax3.set_ylabel('Kecepatan Angular Papan (deg/s)', color=color3)
    line3 = ax3.plot(t, wb_ang_mag, color=color3, label='Kecepatan Angular Target')
    ax3.tick_params(axis='y', labelcolor=color3)
    ax3.set_ylim(bottom=0)
    
    ax4 = ax3.twinx()
    color4 = 'tab:purple'
    ax4.set_ylabel('Angular Error (deg)', color=color4)
    line4 = ax4.plot(t, df['tracking_error_ori_deg'], color=color4, label='Error Orientasi', alpha=0.8)
    ax4.tick_params(axis='y', labelcolor=color4)
    ax4.set_ylim(bottom=0)
    
    lines_ang = line3 + line4
    labels_ang = [l.get_label() for l in lines_ang]
    ax3.legend(lines_ang, labels_ang, loc='upper right')
    ax3.set_title("Korelasi Rotasi: Kecepatan Angular vs Galat Orientasi")
    
    fig.suptitle(f"Bukti Fisika Cross-Coupling Inersia - {run_name}", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_4_overlay_correlation.png", dpi=300)
    plt.close(fig)

def plot_joint_velocity_6dof(df, run_name, out_dir):
    """ Plot 6 Subplot memantau saturasi kecepatan (clamping) pada sendi aktuator """
    if 'js_j1_v' not in df.columns:
        return
        
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    t = df['time_norm']
    
    axes[0, 0].plot(t, np.degrees(df['js_j1_v']), 'c-', label='Joint 1 Vel'); axes[0, 0].set_ylabel("J1 (deg/s)"); axes[0, 0].legend(loc='upper right')
    axes[1, 0].plot(t, np.degrees(df['js_j2_v']), 'c-', label='Joint 2 Vel'); axes[1, 0].set_ylabel("J2 (deg/s)"); axes[1, 0].legend(loc='upper right')
    axes[2, 0].plot(t, np.degrees(df['js_j3_v']), 'c-', label='Joint 3 Vel'); axes[2, 0].set_ylabel("J3 (deg/s)"); axes[2, 0].set_xlabel("Waktu (s)"); axes[2, 0].legend(loc='upper right')
    
    axes[0, 1].plot(t, np.degrees(df['js_j4_v']), 'm-', label='Joint 4 Vel'); axes[0, 1].set_ylabel("J4 (deg/s)"); axes[0, 1].legend(loc='upper right')
    axes[1, 1].plot(t, np.degrees(df['js_j5_v']), 'm-', label='Joint 5 Vel'); axes[1, 1].set_ylabel("J5 (deg/s)"); axes[1, 1].legend(loc='upper right')
    axes[2, 1].plot(t, np.degrees(df['js_j6_v']), 'm-', label='Joint 6 Vel'); axes[2, 1].set_ylabel("J6 (deg/s)"); axes[2, 1].set_xlabel("Waktu (s)"); axes[2, 1].legend(loc='upper right')

    fig.suptitle(f"Evaluasi Saturasi Kecepatan Aktuator (Joint Velocity) - {run_name}", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_5_joint_velocity_6dof.png", dpi=300)
    plt.close(fig)

def process_all_dynamic_timeseries():
    if not INPUT_BASE_DIR.exists():
        print(f"Folder '{INPUT_BASE_DIR}' tidak ditemukan.")
        return

    csv_files = []
    for pattern in ["Uji_4*/**/*.csv", "Uji_5*/**/*.csv", "Uji_6*/**/*.csv"]:
        csv_files.extend(list(INPUT_BASE_DIR.glob(pattern)))
        
    print(f"Ditemukan {len(csv_files)} file timeseries dinamis untuk divisualisasi.")
    
    for idx, csv_path in enumerate(csv_files):
        run_name = csv_path.stem.replace("_timeseries", "")
        print(f"\n[{idx+1}/{len(csv_files)}] Generating plots for: {run_name}")
        
        df = pd.read_csv(csv_path)
        df = calculate_dynamic_local_frame(df)
        
        out_dir_full = csv_path.parent.parent / "plots" / run_name / "full_timeline"
        out_dir_full.mkdir(parents=True, exist_ok=True)
        
        if "uji4" not in run_name.lower():
            plot_trajectories_2d(df, run_name, out_dir_full)
        plot_pose_6dof(df, run_name, out_dir_full)
        plot_twist_control_6dof(df, run_name, out_dir_full)
        plot_estimator_6dof(df, run_name, out_dir_full)
        plot_overlay_correlation(df, run_name, out_dir_full)
        plot_joint_velocity_6dof(df, run_name, out_dir_full)
        
        out_dir_zoomed = csv_path.parent.parent / "plots" / run_name / "zoomed_slices"
        out_dir_zoomed.mkdir(parents=True, exist_ok=True)
        
        max_time = df['time_norm'].max()
        window_size = 20.0  
        
        for start_t in np.arange(0, max_time, window_size):
            end_t = start_t + window_size
            df_slice = df[(df['time_norm'] >= start_t) & (df['time_norm'] < end_t)]
            
            if len(df_slice) > 200:
                slice_name = f"{run_name}_zoom_{int(start_t)}s_to_{int(end_t)}s"
                print(f"  -> Slicing timeline: {int(start_t)}s - {int(end_t)}s")
                
                if "uji4" not in run_name.lower():
                    plot_trajectories_2d(df_slice, slice_name, out_dir_zoomed)
                plot_pose_6dof(df_slice, slice_name, out_dir_zoomed)
                plot_twist_control_6dof(df_slice, slice_name, out_dir_zoomed)
                plot_estimator_6dof(df_slice, slice_name, out_dir_zoomed)
                plot_overlay_correlation(df_slice, slice_name, out_dir_zoomed)
                plot_joint_velocity_6dof(df_slice, slice_name, out_dir_zoomed)

if __name__ == "__main__":
    process_all_dynamic_timeseries()
    print("\nVisualisasi 6-DOF dengan fitur Temporal Slicing dan Joint Velocity selesai! Periksa folder 'zoomed_slices'.")