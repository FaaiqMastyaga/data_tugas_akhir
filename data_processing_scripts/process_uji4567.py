#!/usr/bin/env python3
import pandas as pd
import numpy as np
import cv2
import re
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy import signal
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

PROJECT_ROOT = Path(__file__).parent.parent
BASE_DIR = PROJECT_ROOT / "experiment_data"
OUTPUT_DIR = PROJECT_ROOT / "experiment_results"

# --- KONSTANTA COMPUTER VISION ---
SCALE_PX_PER_MM = 10.0
MARGIN_MM = 50.0  
MARKER_DIST_X_MM = 229.0  
MARKER_DIST_Y_MM = 129.0  

# ==========================================
# FUNGSI EKSTRAKSI METADATA DARI NAMA FOLDER
# ==========================================
def parse_metadata(db3_path, run_name):
    speed = 0
    iteration = 0
    direction = 'manual'
    
    # Ekstrak Kecepatan (contoh: speed_2_cm_s -> 2)
    speed_match = re.search(r'speed_(\d+)_cm_s', run_name.lower())
    if speed_match:
        speed = int(speed_match.group(1))
        
    # Ekstrak Iterasi (angka di ujung string, contoh: _1 -> 1)
    iter_match = re.search(r'_(\d+)$', run_name)
    if iter_match:
        iteration = int(iter_match.group(1))
        
    # Ekstrak Arah dari path folder
    path_str = str(db3_path).lower()
    if 'start_left' in path_str:
        direction = 'start_left'
    elif 'start_right' in path_str:
        direction = 'start_right'
        
    return speed, iteration, direction

# ==========================================
# MODUL 1: EKSTRAKSI ROSBAG & METRIK KENDALI
# ==========================================
def calculate_phase_lag(ref_sig, act_sig, dt):
    if len(ref_sig) < 10 or np.std(ref_sig) < 1e-6 or np.std(act_sig) < 1e-6:
        return 0.0
    ref = ref_sig - np.mean(ref_sig)
    act = act_sig - np.mean(act_sig)
    corr = signal.correlate(act, ref, mode='full')
    lags = signal.correlation_lags(len(act), len(ref), mode='full')
    lag_idx = np.argmax(corr)
    return lags[lag_idx] * dt * 1000.0

def extract_dynamic_bag(bag_path, typestore):
    data_dict = {
        'ref_pose': [], 'act_pose': [], 'ref_twist': [], 'act_twist': [], 'cmd_twist': [],
        'wb_pose_eskf': [], 'wb_twist_raw': [], 'wb_twist_eskf': [], 'joint_states': []
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
                    row = [t, msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, roll, pitch, yaw, qx, qy, qz, qw]
                    if topic == '/tracking/state/ref_pose': data_dict['ref_pose'].append(row)
                    elif topic == '/tracking/state/actual_pose': data_dict['act_pose'].append(row)
                    elif topic == '/estimated_target_pose/eskf': data_dict['wb_pose_eskf'].append(row)

                # --- TWIST TOPICS ---
                elif topic in ['/tracking/state/ref_twist', '/tracking/state/actual_twist', '/tracking/state/command_twist', '/estimated_target_twist/raw', '/estimated_target_twist/eskf']:
                    msg = typestore.deserialize_cdr(rawdata, topic_types[topic])
                    twist_obj = msg.twist if hasattr(msg, 'twist') else msg
                    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 if hasattr(msg, 'header') else timestamp * 1e-9
                    row = [t, twist_obj.linear.x, twist_obj.linear.y, twist_obj.linear.z, twist_obj.angular.x, twist_obj.angular.y, twist_obj.angular.z]
                    if topic == '/tracking/state/ref_twist': data_dict['ref_twist'].append(row)
                    elif topic == '/tracking/state/actual_twist': data_dict['act_twist'].append(row)
                    elif topic == '/tracking/state/command_twist': data_dict['cmd_twist'].append(row)
                    elif topic == '/estimated_target_twist/raw': data_dict['wb_twist_raw'].append(row)
                    elif topic == '/estimated_target_twist/eskf': data_dict['wb_twist_eskf'].append(row)
                
                # --- JOINT STATES TOPIC ---
                elif topic == '/joint_states':
                    msg = typestore.deserialize_cdr(rawdata, topic_types[topic])
                    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 if hasattr(msg, 'header') else timestamp * 1e-9
                    if len(msg.velocity) >= 6:
                        vel_dict = dict(zip(msg.name, msg.velocity))
                        try:
                            row = [t, vel_dict['elfin_joint1'], vel_dict['elfin_joint2'], vel_dict['elfin_joint3'], vel_dict['elfin_joint4'], vel_dict['elfin_joint5'], vel_dict['elfin_joint6']]
                            data_dict['joint_states'].append(row)
                        except KeyError: pass
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
        'cmd_t_': pd.DataFrame(data_dict['cmd_twist'], columns=cols_twist),
        'wb_p_eskf_': pd.DataFrame(data_dict['wb_pose_eskf'], columns=cols_pose),
        'wb_t_raw_': pd.DataFrame(data_dict['wb_twist_raw'], columns=cols_twist),
        'wb_t_eskf_': pd.DataFrame(data_dict['wb_twist_eskf'], columns=cols_twist),
        'js_': pd.DataFrame(data_dict['joint_states'], columns=cols_joints)
    }
    
    if dfs['ref_p_'].empty or dfs['act_p_'].empty: return None, None
    base_times = dfs['ref_p_']['time'].values
    
    def interpolate_df(df, prefix):
        if df.empty: return pd.DataFrame(0.0, index=base_times, columns=[c for c in df.columns if c != 'time']).add_prefix(prefix)
        df_idx = df.set_index('time')
        union_idx = df_idx.index.union(base_times).sort_values()
        return df_idx.reindex(union_idx).interpolate(method='index').bfill().ffill().reindex(base_times).add_prefix(prefix)

    df_merged = pd.concat([interpolate_df(df, prefix) for prefix, df in dfs.items()], axis=1).reset_index().rename(columns={'index': 'time'})
    df_merged['time_norm'] = df_merged['time'] - df_merged['time'].iloc[0]
    dt_mean = df_merged['time_norm'].diff().mean()
    
    # --- METRIK MPC ---
    df_merged['tracking_error_pos_mm'] = np.sqrt((df_merged['act_p_x'] - df_merged['ref_p_x'])**2 + (df_merged['act_p_y'] - df_merged['ref_p_y'])**2 + (df_merged['act_p_z'] - df_merged['ref_p_z'])**2) * 1000.0
    dot_q = np.clip(np.abs(df_merged['ref_p_qx'] * df_merged['act_p_qx'] + df_merged['ref_p_qy'] * df_merged['act_p_qy'] + df_merged['ref_p_qz'] * df_merged['act_p_qz'] + df_merged['ref_p_qw'] * df_merged['act_p_qw']), -1.0, 1.0)
    df_merged['tracking_error_ori_deg'] = np.degrees(2 * np.arccos(dot_q))

    ref_vel_mag = np.sqrt(df_merged['ref_t_vx']**2 + df_merged['ref_t_vy']**2 + df_merged['ref_t_vz']**2)
    dynamic_mask = ref_vel_mag > 0.005 
    dynamic_df = df_merged.iloc[max(0, np.where(dynamic_mask)[0][0] - int(0.5 / dt_mean)) : min(len(df_merged), np.where(dynamic_mask)[0][-1] + int(0.5 / dt_mean))].copy() if dynamic_mask.any() else df_merged.copy()

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

# ==========================================
# MODUL 2: COMPUTER VISION 
# ==========================================
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1); rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect

def fit_circle_least_squares(x, y):
    x_m, y_m = np.mean(x), np.mean(y)
    def objective_func(c): return np.sqrt((x - c[0])**2 + (y - c[1])**2) - c[2]
    res = least_squares(objective_func, [x_m, y_m, 100.0])
    return res.x[0], res.x[1], res.x[2]

def get_reference_polygon(shape_type, cx, cy, scale):
    R = 60.0 * scale
    if shape_type == 'square': return np.array([[cx-R, cy-R], [cx+R, cy-R], [cx+R, cy+R], [cx-R, cy+R]], np.int32)
    elif shape_type == 'triangle': h_off = R * 0.866; return np.array([[cx, cy-R], [cx+R, cy+h_off], [cx-R, cy+h_off]], np.int32)
    return None

def fit_polygon_translation(x_ink, y_ink, shape_type, cx, cy, scale, img_shape):
    ink_mask = np.zeros(img_shape, dtype=np.uint8)
    ink_mask[y_ink, x_ink] = 255
    ink_dist = cv2.distanceTransform(cv2.bitwise_not(ink_mask), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    poly_pts = get_reference_polygon(shape_type, cx, cy, scale)
    sampled_pts = []
    for i in range(len(poly_pts)):
        p1, p2 = poly_pts[i], poly_pts[(i+1)%len(poly_pts)]
        dist = np.linalg.norm(p2 - p1)
        for t in np.linspace(0, 1, max(2, int(dist / 2.0)), endpoint=False): sampled_pts.append(p1 * (1-t) + p2 * t)
    sampled_pts = np.array(sampled_pts)
    
    def objective(c): return map_coordinates(ink_dist, [(sampled_pts + c)[:, 1], (sampled_pts + c)[:, 0]], order=1, mode='nearest')
    res = least_squares(objective, [np.mean(x_ink) - cx, np.mean(y_ink) - cy])
    return cx + res.x[0], cy + res.x[1], get_reference_polygon(shape_type, cx + res.x[0], cy + res.x[1], scale)

def process_vision(img_path, shape_type, out_vis_path):
    img = cv2.imread(str(img_path))
    if img is None: return None
    
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(img)
    if ids is None or len(ids) < 4: return None
        
    src_pts = order_points(np.array([np.mean(c[0], axis=0) for c in corners]))
    margin_px, w_px, h_px = int(MARGIN_MM * SCALE_PX_PER_MM), int(MARKER_DIST_X_MM * SCALE_PX_PER_MM), int(MARKER_DIST_Y_MM * SCALE_PX_PER_MM)
    dst_pts = np.array([[margin_px, margin_px], [margin_px + w_px, margin_px], [margin_px + w_px, margin_px + h_px], [margin_px, margin_px + h_px]], dtype="float32")
    canvas_w, canvas_h = w_px + 2*margin_px, h_px + 2*margin_px
    warped = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src_pts, dst_pts), (canvas_w, canvas_h))
    
    mask = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    cv2.rectangle(mask, (0, 0), (canvas_w, int(35.0 * SCALE_PX_PER_MM)), 0, -1)
    cv2.rectangle(mask, (0, canvas_h - int(15.0 * SCALE_PX_PER_MM)), (canvas_w, canvas_h), 0, -1)
    cv2.rectangle(mask, (0, 0), (int(15.0 * SCALE_PX_PER_MM), canvas_h), 0, -1)
    cv2.rectangle(mask, (canvas_w - int(15.0 * SCALE_PX_PER_MM), 0), (canvas_w, canvas_h), 0, -1)
    
    block = int(35.0 * SCALE_PX_PER_MM)
    for pt in dst_pts:
        cv2.rectangle(mask, (int(pt[0])-block, int(pt[1])-block), (int(pt[0])+block, int(pt[1])+block), 0, -1)
    
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15)
    binary_masked = cv2.bitwise_and(binary, binary, mask=mask)
    
    # TAHAP 1: DYNAMIC CAD CORRIDOR MASKING (POTONG DULU)
    center_x_abs = margin_px + (w_px / 2.0)
    center_y_abs = margin_px + (h_px / 2.0)
    ref_mask_cad = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    
    if shape_type == 'circle':
        cv2.circle(ref_mask_cad, (int(center_x_abs), int(center_y_abs)), int(60.0 * SCALE_PX_PER_MM), 255, 1, cv2.LINE_8)
    else:
        poly_pts_cad = get_reference_polygon(shape_type, center_x_abs, center_y_abs, SCALE_PX_PER_MM)
        if poly_pts_cad is not None:
            cv2.polylines(ref_mask_cad, [poly_pts_cad], True, 255, 1, cv2.LINE_8)
            
    ref_mask_inv_cad = cv2.bitwise_not(ref_mask_cad)
    dist_map_cad = cv2.distanceTransform(ref_mask_inv_cad, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    corridor_mask = (dist_map_cad <= 250.0).astype(np.uint8) * 255
    binary_corridor = cv2.bitwise_and(binary_masked, binary_masked, mask=corridor_mask)

    # TAHAP 2: CONNECTED COMPONENTS (TIMBANG SISANYA)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_corridor, connectivity=8)
    binary_filtered = np.zeros_like(binary_corridor)
    if num_labels > 1:
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > max(400, max(stats[1:, cv2.CC_STAT_AREA]) * 0.05): binary_filtered[labels == i] = 255
                
    y_coords, x_coords = np.nonzero(cv2.ximgproc.thinning(cv2.morphologyEx(binary_filtered, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8)), thinningType=cv2.ximgproc.THINNING_ZHANGSUEN))
    cv_metrics = {k: np.nan for k in ['CV_Abs_MAE_mm', 'CV_Abs_RMSE_mm', 'CV_Abs_MaxErr_mm', 'CV_Shape_MAE_mm', 'CV_Shape_RMSE_mm', 'CV_Shape_MaxErr_mm', 'CV_Offset_mm']}
    
    vis_img = warped.copy()
    dark_overlay = warped.copy(); dark_overlay[mask == 0] = dark_overlay[mask == 0] // 3
    vis_img = cv2.addWeighted(vis_img, 0.4, dark_overlay, 0.6, 0)
    vis_img[cv2.dilate(cv2.ximgproc.thinning(cv2.morphologyEx(binary_filtered, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8)), thinningType=cv2.ximgproc.THINNING_ZHANGSUEN), np.ones((5,5), np.uint8), iterations=1) == 255] = [255, 0, 0] 
    
    if len(x_coords) > 50:
        ref_mask_fit = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        if shape_type == 'circle':
            cv2.circle(vis_img, (int(center_x_abs), int(center_y_abs)), int(60.0 * SCALE_PX_PER_MM), (0, 255, 0), 2, cv2.LINE_AA)
            xc_fit, yc_fit, r_fit = fit_circle_least_squares(x_coords, y_coords)
            cv2.circle(vis_img, (int(xc_fit), int(yc_fit)), int(r_fit), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(vis_img, (int(xc_fit), int(yc_fit)), 5, (0, 0, 255), -1)
            cv2.circle(ref_mask_fit, (int(xc_fit), int(yc_fit)), int(r_fit), 255, 1, cv2.LINE_8)
            cv_metrics['CV_Offset_mm'] = np.sqrt((xc_fit - center_x_abs)**2 + (yc_fit - center_y_abs)**2) / SCALE_PX_PER_MM
        else:
            poly_pts_cad = get_reference_polygon(shape_type, center_x_abs, center_y_abs, SCALE_PX_PER_MM)
            cv2.polylines(vis_img, [poly_pts_cad], True, (0, 255, 0), 2, cv2.LINE_AA)
            fit_cx, fit_cy, fit_poly = fit_polygon_translation(x_coords, y_coords, shape_type, center_x_abs, center_y_abs, SCALE_PX_PER_MM, (canvas_h, canvas_w))
            cv2.polylines(vis_img, [fit_poly], True, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(vis_img, (int(fit_cx), int(fit_cy)), 5, (0, 0, 255), -1)
            cv2.polylines(ref_mask_fit, [fit_poly], True, 255, 1, cv2.LINE_8)
            cv_metrics['CV_Offset_mm'] = np.sqrt((fit_cx - center_x_abs)**2 + (fit_cy - center_y_abs)**2) / SCALE_PX_PER_MM
                
        errors_abs_mm = cv2.distanceTransform(cv2.bitwise_not(ref_mask_cad), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[y_coords, x_coords] / SCALE_PX_PER_MM
        cv_metrics.update({'CV_Abs_MAE_mm': np.mean(errors_abs_mm), 'CV_Abs_RMSE_mm': np.sqrt(np.mean(errors_abs_mm**2)), 'CV_Abs_MaxErr_mm': np.max(errors_abs_mm)})
        
        errors_shape_mm = cv2.distanceTransform(cv2.bitwise_not(ref_mask_fit), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[y_coords, x_coords] / SCALE_PX_PER_MM
        cv_metrics.update({'CV_Shape_MAE_mm': np.mean(errors_shape_mm), 'CV_Shape_RMSE_mm': np.sqrt(np.mean(errors_shape_mm**2)), 'CV_Shape_MaxErr_mm': np.max(errors_shape_mm)})
        cv2.circle(vis_img, (int(center_x_abs), int(center_y_abs)), 8, (0, 255, 0), -1)

    if not np.isnan(cv_metrics['CV_Abs_RMSE_mm']):
        for i, txt in enumerate([f"Shape: {shape_type.upper()}", f"Abs RMSE: {cv_metrics['CV_Abs_RMSE_mm']:.2f} mm", f"Shape RMSE: {cv_metrics['CV_Shape_RMSE_mm']:.2f} mm", f"Offset: {cv_metrics['CV_Offset_mm']:.2f} mm", f"Shape MaxErr: {cv_metrics['CV_Shape_MaxErr_mm']:.2f} mm"]):
            cv2.putText(vis_img, txt, (40, 40 + i*35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 5)
            cv2.putText(vis_img, txt, (40, 40 + i*35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
            
    cv2.imwrite(str(out_vis_path), vis_img)
    return cv_metrics

# ==========================================
# MODUL 3: MAIN EXECUTION
# ==========================================
def get_shape_type_from_name(run_name):
    name_lower = run_name.lower()
    if 'persegi' in name_lower or 'square' in name_lower: return 'square'
    if 'segitiga' in name_lower or 'triangle' in name_lower: return 'triangle'
    return 'circle'

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    
    db3_files = []
    # TAMBAHAN: Mencari file untuk Uji 4, 5, 6, dan 7
    for pattern in ["*uji4*.db3", "*uji5*.db3", "*uji6*.db3", "*uji7*.db3"]:
        db3_files.extend(list(BASE_DIR.rglob(pattern)))
        
    if not db3_files:
        print("Tidak ada file ROSBAG untuk pengujian dinamis ditemukan.")
        return
        
    print(f"Ditemukan {len(db3_files)} data pengujian dinamis.\n")
    all_results = []
    
    for idx, db3_path in enumerate(db3_files):
        run_name = db3_path.parent.name
        
        # Kategorisasi cerdas
        if "uji4" in run_name.lower(): test_cat = "Uji_4_Point_Manual"
        elif "uji5" in run_name.lower(): test_cat = "Uji_5_Point_Cart"
        elif "uji6" in run_name.lower(): test_cat = "Uji_6_Trajectory_Manual"
        elif "uji7" in run_name.lower(): test_cat = "Uji_7_Trajectory_Cart"
        else: test_cat = "Uji_Dinamis_Lainnya"
        
        # Ekstrak Metadata (Kecepatan, Iterasi, Arah)
        speed, iteration, direction = parse_metadata(db3_path, run_name)
        
        # MENCEGAH OVERWRITE (NAME COLLISION)
        # Jika bukan uji manual, gabungkan arah dengan nama asli run_name menjadi entitas nama unik
        unique_run_name = f"{direction}_{run_name}" if direction != 'manual' else run_name
        
        print(f"\n[{idx+1}/{len(db3_files)}] Memproses: {unique_run_name} (Speed: {speed}, Dir: {direction}, Iter: {iteration})")
        
        # 1. Ekstrak data dari ROSBAG
        metrics, df_timeseries = extract_dynamic_bag(db3_path.parent, typestore)
        if metrics is None: continue
            
        ts_dir = OUTPUT_DIR / test_cat / "timeseries"
        ts_dir.mkdir(parents=True, exist_ok=True)
        # Gunakan nama unik untuk penyimpanan timeseries CSV
        df_timeseries.to_csv(ts_dir / f"{unique_run_name}_timeseries.csv", index=False)

        # 2. Ekstrak data dari Foto JPG (Hanya Uji 6 & Uji 7)
        cv_metrics = {k: np.nan for k in ['CV_Abs_MAE_mm', 'CV_Abs_RMSE_mm', 'CV_Abs_MaxErr_mm', 'CV_Shape_MAE_mm', 'CV_Shape_RMSE_mm', 'CV_Shape_MaxErr_mm', 'CV_Offset_mm']}
        
        # Pengecekan: Uji 4 dan Uji 5 di-bypass dari proses Visi Komputer
        if "uji4" not in run_name.lower() and "uji5" not in run_name.lower():
            # Tetap gunakan run_name asli untuk mencari JPG di sistem file
            image_path = db3_path.parent.parent / f"{run_name}.jpg"
            if image_path.exists():
                print(f"  -> Memproses CV dari foto: {image_path.name}")
                vis_dir = OUTPUT_DIR / test_cat / "vision_images"
                vis_dir.mkdir(parents=True, exist_ok=True)
                # Gunakan nama unik untuk menyimpan foto analisis hasil output CV
                res_cv = process_vision(image_path, get_shape_type_from_name(run_name), vis_dir / f"{unique_run_name}_analyzed.jpg")
                if res_cv: cv_metrics = res_cv
            else:
                print(f"  -> [WARNING] Foto {image_path.name} tidak ditemukan! Mengabaikan analisis CV.")
        else:
            print("  -> (Point Tracking) Melewati proses Computer Vision.")
        
        # Gabungkan metrik beserta metadata uji
        all_results.append({
            'Kategori': test_cat,
            'Skenario': unique_run_name, # Terhubung ke nama unik agar tidak melebur di rekapan akhir
            'Arah_Mulai': direction,
            'Kecepatan_cm_s': speed,
            'Iterasi': iteration,
            **metrics,
            **cv_metrics
        })

    if all_results:
        print("\n===================================================")
        df_all = pd.DataFrame(all_results)
        for cat_name, df_group in df_all.groupby('Kategori'):
            csv_path = OUTPUT_DIR / f"Summary_{cat_name}.csv"
            
            # Sortir khusus untuk estetika laporan akhir
            if 'Kecepatan_cm_s' in df_group.columns:
                df_group = df_group.sort_values(by=['Skenario', 'Arah_Mulai', 'Kecepatan_cm_s', 'Iterasi'])
                
            float_cols = [c for c in df_group.columns if 'mm' in c or 'deg' in c or 'ms' in c or 'Ratio' in c]
            df_group[float_cols] = df_group[float_cols].round(3)
            df_group.to_csv(csv_path, index=False)
            print(f"Tabel Disimpan: {csv_path.name}")
            
if __name__ == "__main__":
    main()