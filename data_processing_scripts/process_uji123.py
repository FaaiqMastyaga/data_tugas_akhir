#!/usr/bin/env python3
import pandas as pd
import numpy as np
import cv2
import re
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from collections import defaultdict

# ==========================================
# KONFIGURASI EKSEKUSI
# ==========================================
TEST_MODE = False        # False = Eksekusi Seluruh File, True = Sampel Uji
SAMPLES_PER_GROUP = 2   

SCALE_PX_PER_MM = 10.0
MARGIN_MM = 50.0  
MARKER_DIST_X_MM = 229.0  
MARKER_DIST_Y_MM = 129.0  

# Penyesuaian Path Berbasis Struktur Repositori
PROJECT_ROOT = Path(__file__).parent.parent
BASE_DIR = PROJECT_ROOT / "experiment_data"
OUTPUT_DIR = PROJECT_ROOT / "experiment_results"

# ==========================================
# MODUL 1: EKSTRAKSI ROSBAG & METRIK KENDALI
# ==========================================
def extract_bag_to_metrics(bag_path, typestore):
    ref_pose_data, act_pose_data, ref_twist_data, act_twist_data = [], [], [], []
    
    try:
        with Reader(bag_path) as reader:
            topic_types = {name: info.msgtype for name, info in reader.topics.items()}
            for connection, timestamp, rawdata in reader.messages():
                topic = connection.topic
                
                if topic in ['/tracking/state/ref_pose', '/tracking/state/actual_pose']:
                    msg = typestore.deserialize_cdr(rawdata, topic_types[topic])
                    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    q = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
                    roll, pitch, yaw = R.from_quat(q).as_euler('xyz', degrees=False)
                    row = [t, msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, roll, pitch, yaw]
                    if 'ref' in topic: ref_pose_data.append(row)
                    else: act_pose_data.append(row)
                        
                elif topic in ['/tracking/state/ref_twist', '/tracking/state/actual_twist']:
                    msg = typestore.deserialize_cdr(rawdata, topic_types[topic])
                    twist_obj = msg.twist if hasattr(msg, 'twist') else msg
                    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 if hasattr(msg, 'header') else timestamp * 1e-9
                    row = [t, twist_obj.linear.x, twist_obj.linear.y, twist_obj.linear.z, twist_obj.angular.x, twist_obj.angular.y, twist_obj.angular.z]
                    if 'ref' in topic: ref_twist_data.append(row)
                    else: act_twist_data.append(row)
                        
    except Exception as e:
        print(f"  [ERROR] Gagal membaca ROSBAG: {e}")
        return None

    pose_cols = ['time', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']
    twist_cols = ['time', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz']
    
    df_ref_pose = pd.DataFrame(ref_pose_data, columns=pose_cols)
    df_act_pose = pd.DataFrame(act_pose_data, columns=pose_cols)
    df_ref_twist = pd.DataFrame(ref_twist_data, columns=twist_cols)
    df_act_twist = pd.DataFrame(act_twist_data, columns=twist_cols)
    
    if df_ref_pose.empty or df_act_pose.empty: return None

    all_times = np.unique(np.concatenate([df_ref_pose['time'].values, df_act_pose['time'].values]))
    
    def interpolate_df(df, prefix):
        if df.empty: 
            cols = [c for c in df.columns if c != 'time']
            return pd.DataFrame(0.0, index=all_times, columns=cols).add_prefix(prefix)
            
        df_idx = df.set_index('time')
        union_idx = df_idx.index.union(all_times).sort_values()
        df_interp = df_idx.reindex(union_idx).interpolate(method='index').bfill().ffill()
        return df_interp.reindex(all_times).add_prefix(prefix)

    df_merged = pd.concat([
        interpolate_df(df_ref_pose, 'ref_pose_'),
        interpolate_df(df_act_pose, 'actual_pose_'),
        interpolate_df(df_ref_twist, 'ref_twist_'),
        interpolate_df(df_act_twist, 'actual_twist_')
    ], axis=1).reset_index().rename(columns={'index': 'time'}).fillna(0.0)
    
    dx = df_merged['actual_pose_x'] - df_merged['ref_pose_x']
    dy = df_merged['actual_pose_y'] - df_merged['ref_pose_y']
    dz = df_merged['actual_pose_z'] - df_merged['ref_pose_z']
    df_merged['error_3d_mm'] = np.sqrt(dx**2 + dy**2 + dz**2) * 1000.0
    
    return {
        'ROS_MAE_mm': df_merged['error_3d_mm'].mean(), 
        'ROS_RMSE_mm': np.sqrt((df_merged['error_3d_mm'] ** 2).mean()), 
        'ROS_MaxErr_mm': df_merged['error_3d_mm'].max()
    }, df_merged

# ==========================================
# MODUL 2: COMPUTER VISION
# ==========================================
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect

def fit_circle_least_squares(x, y):
    x_m, y_m = np.mean(x), np.mean(y)
    def objective_func(c):
        xc, yc, R = c
        return np.sqrt((x - xc)**2 + (y - yc)**2) - R
    res = least_squares(objective_func, [x_m, y_m, 100.0])
    return res.x[0], res.x[1], res.x[2]

def get_reference_polygon(shape_type, cx, cy, scale):
    R = 60.0 * scale
    if shape_type == 'square':
        return np.array([[cx-R, cy-R], [cx+R, cy-R], [cx+R, cy+R], [cx-R, cy+R]], np.int32)
    elif shape_type == 'triangle':
        h_off = R * 0.866 
        return np.array([[cx, cy-R], [cx+R, cy+h_off], [cx-R, cy+h_off]], np.int32)
    return None

def fit_polygon_translation(x_ink, y_ink, shape_type, cx, cy, scale, img_shape):
    ink_mask = np.zeros(img_shape, dtype=np.uint8)
    ink_mask[y_ink, x_ink] = 255
    ink_dist = cv2.distanceTransform(cv2.bitwise_not(ink_mask), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    poly_pts = get_reference_polygon(shape_type, cx, cy, scale)
    sampled_pts = []
    
    for i in range(len(poly_pts)):
        p1 = poly_pts[i]
        p2 = poly_pts[(i+1)%len(poly_pts)]
        dist = np.linalg.norm(p2 - p1)
        num_samples = max(2, int(dist / 2.0))
        for t in np.linspace(0, 1, num_samples, endpoint=False):
            sampled_pts.append(p1 * (1-t) + p2 * t)
    sampled_pts = np.array(sampled_pts)
    
    def objective(c):
        dx, dy = c
        pts = sampled_pts + np.array([dx, dy])
        return map_coordinates(ink_dist, [pts[:, 1], pts[:, 0]], order=1, mode='nearest')
        
    dx_guess, dy_guess = np.mean(x_ink) - cx, np.mean(y_ink) - cy
    res = least_squares(objective, [dx_guess, dy_guess])
    
    best_cx = cx + res.x[0]
    best_cy = cy + res.x[1]
    return best_cx, best_cy, get_reference_polygon(shape_type, best_cx, best_cy, scale)

def process_vision(img_path, shape_type, out_vis_path):
    img = cv2.imread(str(img_path))
    if img is None: return None
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(img)
    
    if ids is None or len(ids) < 4: return None
        
    pts = np.array([np.mean(c[0], axis=0) for c in corners])
    src_pts = order_points(pts)
    
    margin_px = int(MARGIN_MM * SCALE_PX_PER_MM)
    w_px = int(MARKER_DIST_X_MM * SCALE_PX_PER_MM)
    h_px = int(MARKER_DIST_Y_MM * SCALE_PX_PER_MM)
    
    dst_pts = np.array([
        [margin_px, margin_px], 
        [margin_px + w_px, margin_px], 
        [margin_px + w_px, margin_px + h_px], 
        [margin_px, margin_px + h_px]
    ], dtype="float32")
    
    canvas_w, canvas_h = int(w_px + 2*margin_px), int(h_px + 2*margin_px)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (canvas_w, canvas_h))
    
    mask = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    top_cut = margin_px + int(2.0 * SCALE_PX_PER_MM) 
    cv2.rectangle(mask, (0, 0), (canvas_w, top_cut), 0, -1)
    cv2.rectangle(mask, (0, 0), (margin_px - 5, canvas_h), 0, -1)
    cv2.rectangle(mask, (canvas_w - margin_px + 5, 0), (canvas_w, canvas_h), 0, -1)
    cv2.rectangle(mask, (0, canvas_h - margin_px + 5), (canvas_w, canvas_h), 0, -1)

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
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_corridor, connectivity=8)
    binary_filtered = np.zeros_like(binary_corridor)
    
    if num_labels > 1:
        max_area = max(stats[1:, cv2.CC_STAT_AREA])
        area_threshold = max(400, max_area * 0.05)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > area_threshold:  
                binary_filtered[labels == i] = 255
                
    binary_clean = cv2.morphologyEx(binary_filtered, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

    thinned = cv2.ximgproc.thinning(binary_clean, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    y_coords, x_coords = np.nonzero(thinned)
    
    cv_metrics = {
        'CV_Abs_MAE_mm': np.nan, 'CV_Abs_RMSE_mm': np.nan, 'CV_Abs_MaxErr_mm': np.nan,
        'CV_Shape_MAE_mm': np.nan, 'CV_Shape_RMSE_mm': np.nan, 'CV_Shape_MaxErr_mm': np.nan,
        'CV_Offset_mm': np.nan
    }
    
    vis_img = warped.copy()
    dark_overlay = warped.copy()
    dark_overlay[mask == 0] = dark_overlay[mask == 0] // 3
    vis_img = cv2.addWeighted(vis_img, 0.4, dark_overlay, 0.6, 0)
    
    thick_thinned = cv2.dilate(thinned, np.ones((5,5), np.uint8), iterations=1)
    vis_img[thick_thinned == 255] = [255, 0, 0] 
    
    if len(x_coords) > 50:
        ref_mask_fit = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        
        if shape_type == 'circle':
            radius_px = int(60.0 * SCALE_PX_PER_MM)
            cv2.circle(vis_img, (int(center_x_abs), int(center_y_abs)), radius_px, (0, 255, 0), 2, cv2.LINE_AA)
            
            xc_fit, yc_fit, r_fit = fit_circle_least_squares(x_coords, y_coords)
            cv2.circle(vis_img, (int(xc_fit), int(yc_fit)), int(r_fit), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(vis_img, (int(xc_fit), int(yc_fit)), 5, (0, 0, 255), -1)
            
            cv2.circle(ref_mask_fit, (int(xc_fit), int(yc_fit)), int(r_fit), 255, 1, cv2.LINE_8)
            cv_metrics['CV_Offset_mm'] = np.sqrt((xc_fit - center_x_abs)**2 + (yc_fit - center_y_abs)**2) / SCALE_PX_PER_MM
        else:
            poly_pts_cad = get_reference_polygon(shape_type, center_x_abs, center_y_abs, SCALE_PX_PER_MM)
            if poly_pts_cad is not None:
                cv2.polylines(vis_img, [poly_pts_cad], True, (0, 255, 0), 2, cv2.LINE_AA)
                
                fit_cx, fit_cy, fit_poly = fit_polygon_translation(x_coords, y_coords, shape_type, center_x_abs, center_y_abs, SCALE_PX_PER_MM, (canvas_h, canvas_w))
                cv2.polylines(vis_img, [fit_poly], True, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.circle(vis_img, (int(fit_cx), int(fit_cy)), 5, (0, 0, 255), -1)
                
                cv2.polylines(ref_mask_fit, [fit_poly], True, 255, 1, cv2.LINE_8)
                cv_metrics['CV_Offset_mm'] = np.sqrt((fit_cx - center_x_abs)**2 + (fit_cy - center_y_abs)**2) / SCALE_PX_PER_MM
                
        errors_abs_px = dist_map_cad[y_coords, x_coords]
        errors_abs_mm = errors_abs_px / SCALE_PX_PER_MM
        cv_metrics['CV_Abs_MAE_mm'] = np.mean(errors_abs_mm)
        cv_metrics['CV_Abs_RMSE_mm'] = np.sqrt(np.mean(errors_abs_mm**2))
        cv_metrics['CV_Abs_MaxErr_mm'] = np.max(errors_abs_mm)
        
        ref_mask_inv_fit = cv2.bitwise_not(ref_mask_fit)
        dist_map_fit = cv2.distanceTransform(ref_mask_inv_fit, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        errors_shape_px = dist_map_fit[y_coords, x_coords]
        errors_shape_mm = errors_shape_px / SCALE_PX_PER_MM
        cv_metrics['CV_Shape_MAE_mm'] = np.mean(errors_shape_mm)
        cv_metrics['CV_Shape_RMSE_mm'] = np.sqrt(np.mean(errors_shape_mm**2))
        cv_metrics['CV_Shape_MaxErr_mm'] = np.max(errors_shape_mm)
        
        cv2.circle(vis_img, (int(center_x_abs), int(center_y_abs)), 8, (0, 255, 0), -1)

    y0, dy = 40, 35
    if not np.isnan(cv_metrics['CV_Abs_RMSE_mm']):
        texts = [
            f"Shape: {shape_type.upper()}",
            f"Abs RMSE: {cv_metrics['CV_Abs_RMSE_mm']:.2f} mm",
            f"Shape RMSE: {cv_metrics['CV_Shape_RMSE_mm']:.2f} mm",
            f"Offset: {cv_metrics['CV_Offset_mm']:.2f} mm",
            f"Shape MaxErr: {cv_metrics['CV_Shape_MaxErr_mm']:.2f} mm"
        ]
        for i, txt in enumerate(texts):
            cv2.putText(vis_img, txt, (40, y0 + i*dy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 5)
            cv2.putText(vis_img, txt, (40, y0 + i*dy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
            
    cv2.imwrite(str(out_vis_path), vis_img)
    return cv_metrics

# ==========================================
# MODUL 3: MAIN EXECUTOR & PARSER
# ==========================================
def parse_experiment_info(folder_name):
    info = {'Uji': '', 'Shape': '', 'Speed_cm_s': '5', 'Pitch': '0', 'Roll': '0', 'Iter': ''}
    if "uji1" in folder_name:
        info['Uji'] = 'Uji_1_Repeatability'
        m = re.search(r'uji1_([a-z]+)_(\d+)', folder_name)
        if m: info['Shape'], info['Iter'] = m.groups()
    elif "uji2" in folder_name:
        info['Uji'] = 'Uji_2_Speed'
        m = re.search(r'uji2_([a-z]+)_speed_(\d+)_cm_s_(\d+)', folder_name)
        if m: info['Shape'], info['Speed_cm_s'], info['Iter'] = m.groups()
    elif "uji3" in folder_name:
        info['Uji'] = 'Uji_3_Angle'
        info['Shape'] = 'circle'
        if m_pitch := re.search(r'pitch_([+-]?\d+)', folder_name): info['Pitch'] = m_pitch.group(1)
        if m_roll := re.search(r'roll_([+-]?\d+)', folder_name): info['Roll'] = m_roll.group(1)
        if m_iter := re.search(r'_(\d+)$', folder_name): info['Iter'] = m_iter.group(1)
    return info

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    
    db3_files = list(BASE_DIR.rglob("*.db3"))
    file_inventory = []
    
    for f in db3_files:
        meta = parse_experiment_info(f.parent.name)
        if meta['Uji']: file_inventory.append({'path': f, 'meta': meta})
            
    if TEST_MODE:
        print(f"*** TEST MODE AKTIF: Mengambil {SAMPLES_PER_GROUP} sampel dari setiap grup ***\n")
        groups = defaultdict(list)
        for item in file_inventory:
            groups[(item['meta']['Uji'], item['meta']['Shape'])].append(item)
            
        sampled_files = []
        for key, items in groups.items():
            sampled_files.extend(items[:SAMPLES_PER_GROUP])
        files_to_process = sampled_files
    else:
        print(f"*** PRODUCTION MODE: Memproses seluruh {len(file_inventory)} data ***\n")
        files_to_process = file_inventory

    all_results = []
    
    for idx, item in enumerate(files_to_process):
        db3_path = item['path']
        meta = item['meta']
        run_name = db3_path.parent.name
        
        print(f"[{idx+1}/{len(files_to_process)}] Memproses: {run_name} ...")
        
        ros_res = extract_bag_to_metrics(db3_path.parent, typestore)
        if not ros_res: continue
        ros_metrics, df_timeseries = ros_res
        
        ts_dir = OUTPUT_DIR / meta['Uji'] / "timeseries"
        ts_dir.mkdir(parents=True, exist_ok=True)
        df_timeseries.to_csv(ts_dir / f"{run_name}_timeseries.csv", index=False)
        
        cv_metrics = {
            'CV_Abs_MAE_mm': np.nan, 'CV_Abs_RMSE_mm': np.nan, 'CV_Abs_MaxErr_mm': np.nan,
            'CV_Shape_MAE_mm': np.nan, 'CV_Shape_RMSE_mm': np.nan, 'CV_Shape_MaxErr_mm': np.nan,
            'CV_Offset_mm': np.nan
        }
        jpg_candidates = list(db3_path.parent.parent.glob(f"{run_name}.jpg"))
        
        if jpg_candidates:
            vis_dir = OUTPUT_DIR / meta['Uji'] / "vision_images"
            vis_dir.mkdir(parents=True, exist_ok=True)
            res_cv = process_vision(jpg_candidates[0], meta['Shape'], vis_dir / f"{run_name}_analyzed.jpg")
            if res_cv: cv_metrics = res_cv

        all_results.append({**meta, **ros_metrics, **cv_metrics})

    if not all_results: return
    df_all = pd.DataFrame(all_results)
    
    print("\n===================================================")
    for uji_name, df_group in df_all.groupby('Uji'):
        csv_path = OUTPUT_DIR / f"Summary_{uji_name}.csv"
        df_group['Iter'] = pd.to_numeric(df_group['Iter'])
        df_group['Speed_cm_s'] = pd.to_numeric(df_group['Speed_cm_s'])
        df_group['Pitch'] = pd.to_numeric(df_group['Pitch'])
        df_group['Roll'] = pd.to_numeric(df_group['Roll'])
        
        df_group = df_group.sort_values(by=['Shape', 'Speed_cm_s', 'Pitch', 'Roll', 'Iter'])
        
        float_cols = [c for c in df_group.columns if 'mm' in c]
        df_group[float_cols] = df_group[float_cols].round(3)
        df_group.to_csv(csv_path, index=False)
        print(f"Tabel Disimpan: {csv_path.name}")

if __name__ == "__main__":
    main()