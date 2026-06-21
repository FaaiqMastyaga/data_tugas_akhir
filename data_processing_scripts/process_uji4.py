#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy import signal
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import warnings

# Abaikan warning interpolasi pandas yang tidak relevan
warnings.filterwarnings('ignore', category=FutureWarning)

PROJECT_ROOT = Path(__file__).parent.parent
BASE_DIR = PROJECT_ROOT / "experiment_data"
OUTPUT_DIR = PROJECT_ROOT / "experiment_results"

def calculate_phase_lag(ref_sig, act_sig, dt):
    if len(ref_sig) < 10 or np.std(ref_sig) < 1e-6 or np.std(act_sig) < 1e-6:
        return 0.0
        
    ref = ref_sig - np.mean(ref_sig)
    act = act_sig - np.mean(act_sig)
    
    corr = signal.correlate(act, ref, mode='full')
    lags = signal.correlation_lags(len(act), len(ref), mode='full')
    
    lag_idx = np.argmax(corr)
    delay_seconds = lags[lag_idx] * dt
    
    return delay_seconds * 1000.0

def extract_dynamic_bag(bag_path, typestore):
    data_dict = {
        'ref_pose': [], 'act_pose': [], 'ref_twist': [], 'act_twist': [],
        'wb_pose_eskf': [], 'wb_twist_raw': [], 'wb_twist_eskf': [],
        'joint_states': []
    }
    
    try:
        with Reader(bag_path) as reader:
            topic_types = {name: info.msgtype for name, info in reader.topics.items()}
            for connection, timestamp, rawdata in reader.messages():
                topic = connection.topic
                
                # --- POSE TOPICS ---
                if topic in ['/tracking/state/ref_pose', '/tracking/state/actual_pose', '/estimated_target_pose/eskf']:
                    msg = typestore.deserialize_cdr(rawdata, topic_types[topic])
                    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    
                    qx, qy, qz, qw = msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
                    roll, pitch, yaw = R.from_quat([qx, qy, qz, qw]).as_euler('xyz', degrees=False)
                    
                    row = [t, msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, 
                           roll, pitch, yaw, qx, qy, qz, qw]
                    
                    if topic == '/tracking/state/ref_pose': data_dict['ref_pose'].append(row)
                    elif topic == '/tracking/state/actual_pose': data_dict['act_pose'].append(row)
                    elif topic == '/estimated_target_pose/eskf': data_dict['wb_pose_eskf'].append(row)

                # --- TWIST TOPICS ---
                elif topic in ['/tracking/state/ref_twist', '/tracking/state/actual_twist', 
                               '/estimated_target_twist/raw', '/estimated_target_twist/eskf']:
                    msg = typestore.deserialize_cdr(rawdata, topic_types[topic])
                    twist_obj = msg.twist if hasattr(msg, 'twist') else msg
                    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 if hasattr(msg, 'header') else timestamp * 1e-9
                    row = [t, twist_obj.linear.x, twist_obj.linear.y, twist_obj.linear.z, 
                           twist_obj.angular.x, twist_obj.angular.y, twist_obj.angular.z]
                           
                    if topic == '/tracking/state/ref_twist': data_dict['ref_twist'].append(row)
                    elif topic == '/tracking/state/actual_twist': data_dict['act_twist'].append(row)
                    elif topic == '/estimated_target_twist/raw': data_dict['wb_twist_raw'].append(row)
                    elif topic == '/estimated_target_twist/eskf': data_dict['wb_twist_eskf'].append(row)
                
                # --- JOINT STATES TOPIC ---
                elif topic == '/joint_states':
                    msg = typestore.deserialize_cdr(rawdata, topic_types[topic])
                    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 if hasattr(msg, 'header') else timestamp * 1e-9
                    if len(msg.velocity) >= 6:
                        # Map kecepatan berdasarkan nama sendi (jaga-jaga jika urutan berubah)
                        vel_dict = dict(zip(msg.name, msg.velocity))
                        try:
                            row = [t, vel_dict['elfin_joint1'], vel_dict['elfin_joint2'], 
                                   vel_dict['elfin_joint3'], vel_dict['elfin_joint4'], 
                                   vel_dict['elfin_joint5'], vel_dict['elfin_joint6']]
                            data_dict['joint_states'].append(row)
                        except KeyError:
                            pass
                        
    except Exception as e:
        print(f"  [ERROR] Gagal membaca ROSBAG: {e}")
        return None, None

    cols_pose = ['time', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'qx', 'qy', 'qz', 'qw']
    cols_twist = ['time', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz']
    cols_joints = ['time', 'j1_v', 'j2_v', 'j3_v', 'j4_v', 'j5_v', 'j6_v']
    
    dfs = {
        'ref_p_': pd.DataFrame(data_dict['ref_pose'], columns=cols_pose),
        'act_p_': pd.DataFrame(data_dict['act_pose'], columns=cols_pose),
        'ref_t_': pd.DataFrame(data_dict['ref_twist'], columns=cols_twist),
        'act_t_': pd.DataFrame(data_dict['act_twist'], columns=cols_twist),
        'wb_p_eskf_': pd.DataFrame(data_dict['wb_pose_eskf'], columns=cols_pose),
        'wb_t_raw_': pd.DataFrame(data_dict['wb_twist_raw'], columns=cols_twist),
        'wb_t_eskf_': pd.DataFrame(data_dict['wb_twist_eskf'], columns=cols_twist),
        'js_': pd.DataFrame(data_dict['joint_states'], columns=cols_joints)
    }
    
    if dfs['ref_p_'].empty or dfs['act_p_'].empty:
        return None, None

    base_times = dfs['ref_p_']['time'].values
    
    def interpolate_df(df, prefix):
        if df.empty: 
            cols = [c for c in df.columns if c != 'time']
            return pd.DataFrame(0.0, index=base_times, columns=cols).add_prefix(prefix)
            
        df_idx = df.set_index('time')
        union_idx = df_idx.index.union(base_times).sort_values()
        df_interp = df_idx.reindex(union_idx).interpolate(method='index').bfill().ffill()
        return df_interp.reindex(base_times).add_prefix(prefix)

    list_interp = [interpolate_df(df, prefix) for prefix, df in dfs.items()]
    df_merged = pd.concat(list_interp, axis=1).reset_index().rename(columns={'index': 'time'})
    df_merged['time_norm'] = df_merged['time'] - df_merged['time'].iloc[0]
    
    dt_mean = df_merged['time_norm'].diff().mean()
    
    # --- METRIK MPC ---
    dx = df_merged['act_p_x'] - df_merged['ref_p_x']
    dy = df_merged['act_p_y'] - df_merged['ref_p_y']
    dz = df_merged['act_p_z'] - df_merged['ref_p_z']
    df_merged['tracking_error_pos_mm'] = np.sqrt(dx**2 + dy**2 + dz**2) * 1000.0
    
    dot_q = (df_merged['ref_p_qx'] * df_merged['act_p_qx'] +
             df_merged['ref_p_qy'] * df_merged['act_p_qy'] +
             df_merged['ref_p_qz'] * df_merged['act_p_qz'] +
             df_merged['ref_p_qw'] * df_merged['act_p_qw'])
    dot_q = np.clip(np.abs(dot_q), -1.0, 1.0)
    df_merged['tracking_error_ori_deg'] = np.degrees(2 * np.arccos(dot_q))

    ref_vel_mag = np.sqrt(df_merged['ref_t_vx']**2 + df_merged['ref_t_vy']**2 + df_merged['ref_t_vz']**2)
    dynamic_mask = ref_vel_mag > 0.005 
    
    if dynamic_mask.any():
        dynamic_indices = np.where(dynamic_mask)[0]
        start_idx = max(0, dynamic_indices[0] - int(0.5 / dt_mean))
        end_idx = min(len(df_merged), dynamic_indices[-1] + int(0.5 / dt_mean))
        dynamic_df = df_merged.iloc[start_idx:end_idx].copy()
    else:
        dynamic_df = df_merged.copy()

    dyn_ref_vel = np.sqrt(dynamic_df['ref_t_vx']**2 + dynamic_df['ref_t_vy']**2 + dynamic_df['ref_t_vz']**2)
    dyn_act_vel = np.sqrt(dynamic_df['act_t_vx']**2 + dynamic_df['act_t_vy']**2 + dynamic_df['act_t_vz']**2)
    mpc_phase_lag_ms = calculate_phase_lag(dyn_ref_vel.values, dyn_act_vel.values, dt_mean)

    # --- METRIK ESTIMATOR ---
    noise_raw = np.std(df_merged['wb_t_raw_vx']) + np.std(df_merged['wb_t_raw_vy']) + np.std(df_merged['wb_t_raw_wx'])
    noise_eskf = np.std(df_merged['wb_t_eskf_vx']) + np.std(df_merged['wb_t_eskf_vy']) + np.std(df_merged['wb_t_eskf_wx'])
    noise_reduction_ratio = noise_raw / noise_eskf if noise_eskf > 1e-6 else 0
    
    dyn_wb_raw = np.sqrt(dynamic_df['wb_t_raw_vx']**2 + dynamic_df['wb_t_raw_vy']**2 + dynamic_df['wb_t_raw_wx']**2)
    dyn_wb_eskf = np.sqrt(dynamic_df['wb_t_eskf_vx']**2 + dynamic_df['wb_t_eskf_vy']**2 + dynamic_df['wb_t_eskf_wx']**2)
    eskf_phase_lag_ms = calculate_phase_lag(dyn_wb_raw.values, dyn_wb_eskf.values, dt_mean)

    metrics = {
        'Pos_RMSE_mm': np.sqrt((df_merged['tracking_error_pos_mm'] ** 2).mean()), 
        'Pos_MaxErr_mm': df_merged['tracking_error_pos_mm'].max(),
        'Ori_RMSE_deg': np.sqrt((df_merged['tracking_error_ori_deg'] ** 2).mean()),
        'Ori_MaxErr_deg': df_merged['tracking_error_ori_deg'].max(),
        'MPC_Phase_Lag_ms': mpc_phase_lag_ms,
        'ESKF_Phase_Lag_ms': eskf_phase_lag_ms,
        'Noise_Reduction_Ratio': noise_reduction_ratio
    }
    
    return metrics, df_merged

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    
    db3_files = []
    for pattern in ["*uji4*.db3", "*uji5*.db3", "*uji6*.db3"]:
        db3_files.extend(list(BASE_DIR.rglob(pattern)))
        
    if not db3_files:
        print("Tidak ada file ROSBAG untuk pengujian dinamis (uji4, uji5, uji6) yang ditemukan.")
        return
        
    print(f"Ditemukan {len(db3_files)} data pengujian dinamis.\n")
    
    all_results = []
    
    for idx, db3_path in enumerate(db3_files):
        run_name = db3_path.parent.name
        
        if "uji4" in run_name.lower(): test_cat = "Uji_4_Point_Tracking"
        elif "uji5" in run_name.lower(): test_cat = "Uji_5_Linear_Tracking"
        elif "uji6" in run_name.lower(): test_cat = "Uji_6_Dynamic_Drawing"
        else: test_cat = "Uji_Dinamis_Lainnya"
        
        print(f"[{idx+1}/{len(db3_files)}] Memproses: {run_name} ...")
        
        metrics, df_timeseries = extract_dynamic_bag(db3_path.parent, typestore)
        if metrics is None: 
            continue
            
        ts_dir = OUTPUT_DIR / test_cat / "timeseries"
        ts_dir.mkdir(parents=True, exist_ok=True)
        df_timeseries.to_csv(ts_dir / f"{run_name}_timeseries.csv", index=False)
        
        all_results.append({
            'Kategori': test_cat,
            'Skenario': run_name,
            **metrics
        })

    if all_results:
        print("\n===================================================")
        df_all = pd.DataFrame(all_results)
        for cat_name, df_group in df_all.groupby('Kategori'):
            csv_path = OUTPUT_DIR / f"Summary_{cat_name}.csv"
            
            float_cols = ['Pos_RMSE_mm', 'Pos_MaxErr_mm', 'Ori_RMSE_deg', 'Ori_MaxErr_deg', 
                          'MPC_Phase_Lag_ms', 'ESKF_Phase_Lag_ms', 'Noise_Reduction_Ratio']
            df_group[float_cols] = df_group[float_cols].round(3)
            
            df_group.to_csv(csv_path, index=False)
            print(f"Tabel Disimpan: {csv_path.name}")
            
if __name__ == "__main__":
    main()