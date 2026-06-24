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
        ax.axvspan(t_start, t_end, color='#00FF00', alpha=0.08, label='Fase Gambar')
        
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

def plot_trajectories_2d(df, run_name, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].plot(df['ref_p_x'] * 1000, df['ref_p_y'] * 1000, 'k--', label='Referensi (Global)', zorder=2)
    axes[0].plot(df['act_p_x'] * 1000, df['act_p_y'] * 1000, 'b-', label='Aktual Robot (Global)', alpha=0.7, zorder=1)
    axes[0].set_title(f"Trajektori Absolut (Base Frame) - {run_name}"); axes[0].set_xlabel("Global X (mm)"); axes[0].set_ylabel("Global Y (mm)")
    axes[0].legend(); axes[0].axis('equal')
    
    axes[1].plot(df['ref_loc_x'], df['ref_loc_y'], 'k--', label='Referensi (Di Papan)', zorder=2)
    axes[1].plot(df['act_loc_x'], df['act_loc_y'], 'g-', label='Aktual Tinta (Di Papan)', alpha=0.7, zorder=1)
    axes[1].set_title(f"Trajektori Relatif (Whiteboard Frame) - {run_name}"); axes[1].set_xlabel("Local X (mm)"); axes[1].set_ylabel("Local Y (mm)")
    axes[1].legend(); axes[1].axis('equal')
    fig.savefig(out_dir / f"{run_name}_0_trajectory_2d.png", dpi=300); plt.close(fig)

def plot_pose_6dof(df, run_name, out_dir, t_start, t_end):
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    t = df['time_norm']
    
    axes[0, 0].plot(t, df['ref_p_x'] * 1000, 'k--', label='Ref X'); axes[0, 0].plot(t, df['act_p_x'] * 1000, 'b-', label='Act X', alpha=0.7); axes[0, 0].set_ylabel("Posisi X (mm)")
    axes[1, 0].plot(t, df['ref_p_y'] * 1000, 'k--', label='Ref Y'); axes[1, 0].plot(t, df['act_p_y'] * 1000, 'b-', label='Act Y', alpha=0.7); axes[1, 0].set_ylabel("Posisi Y (mm)")
    axes[2, 0].plot(t, df['ref_p_z'] * 1000, 'k--', label='Ref Z'); axes[2, 0].plot(t, df['act_p_z'] * 1000, 'b-', label='Act Z', alpha=0.7); axes[2, 0].set_ylabel("Posisi Z (mm)"); axes[2, 0].set_xlabel("Waktu (s)")
    axes[0, 1].plot(t, np.degrees(df['ref_p_roll']), 'k--', label='Ref Roll'); axes[0, 1].plot(t, np.degrees(df['act_p_roll']), 'g-', label='Act Roll', alpha=0.7); axes[0, 1].set_ylabel("Sudut Roll (deg)")
    axes[1, 1].plot(t, np.degrees(df['ref_p_pitch']), 'k--', label='Ref Pitch'); axes[1, 1].plot(t, np.degrees(df['act_p_pitch']), 'g-', label='Act Pitch', alpha=0.7); axes[1, 1].set_ylabel("Sudut Pitch (deg)")
    axes[2, 1].plot(t, np.degrees(df['ref_p_yaw']), 'k--', label='Ref Yaw'); axes[2, 1].plot(t, np.degrees(df['act_p_yaw']), 'g-', label='Act Yaw', alpha=0.7); axes[2, 1].set_ylabel("Sudut Yaw (deg)"); axes[2, 1].set_xlabel("Waktu (s)")
    
    for ax in axes.flat: add_shading_and_fix_legend(ax, t_start, t_end)
    
    fig.suptitle(f"Analisis Pelacakan Pose 6-DOF (Base Frame) - {run_name}", fontsize=16); fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_1_pose_tracking_6dof.png", dpi=300); plt.close(fig)

def plot_twist_control_6dof(df, run_name, out_dir, t_start, t_end):
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    t = df['time_norm']
    
    axes[0, 0].plot(t, df['ref_t_vx'] * 100, 'k--', label='Ref Vx'); axes[0, 0].plot(t, df['act_t_vx'] * 100, 'b-', label='Act Vx', alpha=0.7); axes[0, 0].set_ylabel("Vx (cm/s)")
    axes[1, 0].plot(t, df['ref_t_vy'] * 100, 'k--', label='Ref Vy'); axes[1, 0].plot(t, df['act_t_vy'] * 100, 'b-', label='Act Vy', alpha=0.7); axes[1, 0].set_ylabel("Vy (cm/s)")
    axes[2, 0].plot(t, df['ref_t_vz'] * 100, 'k--', label='Ref Vz'); axes[2, 0].plot(t, df['act_t_vz'] * 100, 'b-', label='Act Vz', alpha=0.7); axes[2, 0].set_ylabel("Vz (cm/s)"); axes[2, 0].set_xlabel("Waktu (s)")
    axes[0, 1].plot(t, np.degrees(df['ref_t_wx']), 'k--', label='Ref Wx'); axes[0, 1].plot(t, np.degrees(df['act_t_wx']), 'g-', label='Act Wx', alpha=0.7); axes[0, 1].set_ylabel("Wx (deg/s)")
    axes[1, 1].plot(t, np.degrees(df['ref_t_wy']), 'k--', label='Ref Wy'); axes[1, 1].plot(t, np.degrees(df['act_t_wy']), 'g-', label='Act Wy', alpha=0.7); axes[1, 1].set_ylabel("Wy (deg/s)")
    axes[2, 1].plot(t, np.degrees(df['ref_t_wz']), 'k--', label='Ref Wz'); axes[2, 1].plot(t, np.degrees(df['act_t_wz']), 'g-', label='Act Wz', alpha=0.7); axes[2, 1].set_ylabel("Wz (deg/s)"); axes[2, 1].set_xlabel("Waktu (s)")
    
    for ax in axes.flat: add_shading_and_fix_legend(ax, t_start, t_end)

    fig.suptitle(f"Analisis Pelacakan Kecepatan 6-DOF (Twist Ref vs Actual) - {run_name}", fontsize=16); fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_2a_twist_control_6dof.png", dpi=300); plt.close(fig)

def plot_twist_command_saturation_6dof(df, run_name, out_dir, t_start, t_end):
    if 'cmd_t_vx' not in df.columns: return
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    t = df['time_norm']
    
    axes[0, 0].plot(t, df['cmd_t_vx'] * 100, 'r-', label='MPC Cmd', alpha=0.4); axes[0, 0].plot(t, df['act_t_vx'] * 100, 'b-', label='Actual', alpha=0.9); axes[0, 0].set_ylabel("Vx (cm/s)")
    axes[1, 0].plot(t, df['cmd_t_vy'] * 100, 'r-', label='MPC Cmd', alpha=0.4); axes[1, 0].plot(t, df['act_t_vy'] * 100, 'b-', label='Actual', alpha=0.9); axes[1, 0].set_ylabel("Vy (cm/s)")
    axes[2, 0].plot(t, df['cmd_t_vz'] * 100, 'r-', label='MPC Cmd', alpha=0.4); axes[2, 0].plot(t, df['act_t_vz'] * 100, 'b-', label='Actual', alpha=0.9); axes[2, 0].set_ylabel("Vz (cm/s)"); axes[2, 0].set_xlabel("Waktu (s)")
    axes[0, 1].plot(t, np.degrees(df['cmd_t_wx']), 'r-', label='MPC Cmd', alpha=0.4); axes[0, 1].plot(t, np.degrees(df['act_t_wx']), 'g-', label='Actual', alpha=0.9); axes[0, 1].set_ylabel("Wx (deg/s)")
    axes[1, 1].plot(t, np.degrees(df['cmd_t_wy']), 'r-', label='MPC Cmd', alpha=0.4); axes[1, 1].plot(t, np.degrees(df['act_t_wy']), 'g-', label='Actual', alpha=0.9); axes[1, 1].set_ylabel("Wy (deg/s)")
    axes[2, 1].plot(t, np.degrees(df['cmd_t_wz']), 'r-', label='MPC Cmd', alpha=0.4); axes[2, 1].plot(t, np.degrees(df['act_t_wz']), 'g-', label='Actual', alpha=0.9); axes[2, 1].set_ylabel("Wz (deg/s)"); axes[2, 1].set_xlabel("Waktu (s)")
    
    for ax in axes.flat: add_shading_and_fix_legend(ax, t_start, t_end)

    fig.suptitle(f"Saturasi Perangkat Keras (MPC Command vs Actual) - {run_name}", fontsize=16); fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_2b_twist_saturation_6dof.png", dpi=300); plt.close(fig)

def plot_estimator_6dof(df, run_name, out_dir, t_start, t_end):
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    t = df['time_norm']
    
    axes[0, 0].plot(t, df['wb_t_raw_vx'] * 100, color='tab:orange', label='Raw Vx', alpha=0.5); axes[0, 0].plot(t, df['wb_t_eskf_vx'] * 100, color='tab:blue', label='Filtered Vx'); axes[0, 0].set_ylabel("Vx (cm/s)")
    axes[1, 0].plot(t, df['wb_t_raw_vy'] * 100, color='tab:orange', label='Raw Vy', alpha=0.5); axes[1, 0].plot(t, df['wb_t_eskf_vy'] * 100, color='tab:blue', label='Filtered Vy'); axes[1, 0].set_ylabel("Vy (cm/s)")
    axes[2, 0].plot(t, df['wb_t_raw_vz'] * 100, color='tab:orange', label='Raw Vz', alpha=0.5); axes[2, 0].plot(t, df['wb_t_eskf_vz'] * 100, color='tab:blue', label='Filtered Vz'); axes[2, 0].set_ylabel("Vz (cm/s)"); axes[2, 0].set_xlabel("Waktu (s)")
    axes[0, 1].plot(t, np.degrees(df['wb_t_raw_wx']), color='tab:orange', label='Raw Wx', alpha=0.5); axes[0, 1].plot(t, np.degrees(df['wb_t_eskf_wx']), color='tab:green', label='Filtered Wx'); axes[0, 1].set_ylabel("Wx (deg/s)")
    axes[1, 1].plot(t, np.degrees(df['wb_t_raw_wy']), color='tab:orange', label='Raw Wy', alpha=0.5); axes[1, 1].plot(t, np.degrees(df['wb_t_eskf_wy']), color='tab:green', label='Filtered Wy'); axes[1, 1].set_ylabel("Wy (deg/s)")
    axes[2, 1].plot(t, np.degrees(df['wb_t_raw_wz']), color='tab:orange', label='Raw Wz', alpha=0.5); axes[2, 1].plot(t, np.degrees(df['wb_t_eskf_wz']), color='tab:green', label='Filtered Wz'); axes[2, 1].set_ylabel("Wz (deg/s)"); axes[2, 1].set_xlabel("Waktu (s)")
    
    for ax in axes.flat: add_shading_and_fix_legend(ax, t_start, t_end)

    fig.suptitle(f"Kinerja Estimator State 6-DOF - {run_name}", fontsize=16); fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_3_estimator_6dof.png", dpi=300); plt.close(fig)

def plot_overlay_correlation(df, run_name, out_dir, t_start=None, t_end=None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    t = df['time_norm']
    wb_vel_mag = np.sqrt(df['wb_t_eskf_vx']**2 + df['wb_t_eskf_vy']**2 + df['wb_t_eskf_vz']**2) * 100
    
    ax1 = axes[0]; color1 = 'tab:blue'
    ax1.set_ylabel('Kecepatan Linier Papan (cm/s)', color=color1)
    line1 = ax1.plot(t, wb_vel_mag, color=color1, label='Kec. Target')
    ax1.tick_params(axis='y', labelcolor=color1); ax1.set_ylim(bottom=0)
    
    ax2 = ax1.twinx(); color2 = 'tab:red'
    ax2.set_ylabel('Euclidean Error 3D (mm)', color=color2)
    line2 = ax2.plot(t, df['tracking_error_pos_mm'], color=color2, label='Error Posisi', alpha=0.8)
    
    if t_start is not None and t_end is not None:
        mask = (t >= t_start) & (t <= t_end)
        mean_err_pos = df.loc[mask, 'tracking_error_pos_mm'].mean()
        line2 += ax2.plot([t.min(), t.max()], [mean_err_pos, mean_err_pos], color=color2, linestyle='--', alpha=0.5, label=f'Mean: {mean_err_pos:.2f} mm')

    ax2.tick_params(axis='y', labelcolor=color2); ax2.set_ylim(bottom=0)
    add_shading_and_fix_legend(ax1, t_start, t_end)
    ax1.legend(line1 + line2, [l.get_label() for l in line1 + line2], loc='upper right')
    ax1.set_title("Korelasi Translasi: Kecepatan Linier vs Galat Posisi 3D")
    
    wb_ang_mag = np.degrees(np.sqrt(df['wb_t_eskf_wx']**2 + df['wb_t_eskf_wy']**2 + df['wb_t_eskf_wz']**2))
    ax3 = axes[1]; color3 = 'tab:green'
    ax3.set_xlabel('Waktu (s)'); ax3.set_ylabel('Kecepatan Angular Papan (deg/s)', color=color3)
    line3 = ax3.plot(t, wb_ang_mag, color=color3, label='Kec. Target')
    ax3.tick_params(axis='y', labelcolor=color3); ax3.set_ylim(bottom=0)
    
    ax4 = ax3.twinx(); color4 = 'tab:purple'
    ax4.set_ylabel('Angular Error (deg)', color=color4)
    line4 = ax4.plot(t, df['tracking_error_ori_deg'], color=color4, label='Error Orientasi', alpha=0.8)
    
    if t_start is not None and t_end is not None:
        mean_err_ori = df.loc[mask, 'tracking_error_ori_deg'].mean()
        line4 += ax4.plot([t.min(), t.max()], [mean_err_ori, mean_err_ori], color=color4, linestyle='--', alpha=0.5, label=f'Mean: {mean_err_ori:.2f} deg')

    ax4.tick_params(axis='y', labelcolor=color4); ax4.set_ylim(bottom=0)
    add_shading_and_fix_legend(ax3, t_start, t_end)
    ax3.legend(line3 + line4, [l.get_label() for l in line3 + line4], loc='upper right')
    ax3.set_title("Korelasi Rotasi: Kecepatan Angular vs Galat Orientasi")
    
    fig.suptitle(f"Bukti Fisika Cross-Coupling Inersia - {run_name}", fontsize=16); fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_4_overlay_correlation.png", dpi=300); plt.close(fig)

def plot_joint_velocity_6dof(df, run_name, out_dir, t_start, t_end):
    if 'js_j1_v' not in df.columns: return
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    t = df['time_norm']
    
    axes[0, 0].plot(t, np.degrees(df['js_j1_v']), 'c-', label='J1 Vel'); axes[0, 0].set_ylabel("J1 (deg/s)")
    axes[1, 0].plot(t, np.degrees(df['js_j2_v']), 'c-', label='J2 Vel'); axes[1, 0].set_ylabel("J2 (deg/s)")
    axes[2, 0].plot(t, np.degrees(df['js_j3_v']), 'c-', label='J3 Vel'); axes[2, 0].set_ylabel("J3 (deg/s)"); axes[2, 0].set_xlabel("Waktu (s)")
    axes[0, 1].plot(t, np.degrees(df['js_j4_v']), 'm-', label='J4 Vel'); axes[0, 1].set_ylabel("J4 (deg/s)")
    axes[1, 1].plot(t, np.degrees(df['js_j5_v']), 'm-', label='J5 Vel'); axes[1, 1].set_ylabel("J5 (deg/s)")
    axes[2, 1].plot(t, np.degrees(df['js_j6_v']), 'm-', label='J6 Vel'); axes[2, 1].set_ylabel("J6 (deg/s)"); axes[2, 1].set_xlabel("Waktu (s)")
    
    for ax in axes.flat: add_shading_and_fix_legend(ax, t_start, t_end)

    fig.suptitle(f"Evaluasi Saturasi Kecepatan Aktuator (Joint Velocity) - {run_name}", fontsize=16); fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_5_joint_velocity_6dof.png", dpi=300); plt.close(fig)

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
        
        # --- PENGEMBALIAN FITUR ZOOMED SLICES ---
        # Render plot irisan per 20 detik HANYA jika durasi akhir melebihi 30 detik
        max_time = df['time_norm'].max()
        window_size = 20.0  
        
        if max_time > 30.0:
            out_dir_zoomed = csv_path.parent.parent / "plots" / run_name / "zoomed_slices"
            out_dir_zoomed.mkdir(parents=True, exist_ok=True)
            
            for slice_start in np.arange(0, max_time, window_size):
                slice_end = slice_start + window_size
                df_slice = df[(df['time_norm'] >= slice_start) & (df['time_norm'] < slice_end)]
                
                if len(df_slice) > 200: # Syarat minimum jumlah baris data agar plot tidak kosong
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
                
    print("\nSeluruh Visualisasi selesai!")

if __name__ == "__main__":
    process_all_dynamic_timeseries()