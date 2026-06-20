#!/usr/bin/env python3

import os
import sys
import math
import numpy as np

# Native ROS 2 Bag libraries
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

def euler_from_quaternion(x, y, z, w):
    """Convert a quaternion into euler angles (roll, pitch, yaw) in degrees"""
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.degrees(math.atan2(t3, t4))

    return roll_x, pitch_y, yaw_z

def process_bag(bag_path, telemetry_topic):
    aimooe_topic = "/aimooe/tool_info"

    # 1. Setup the Bag Reader
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr')

    reader = rosbag2_py.SequentialReader()
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening bag: {e}")
        return

    # 2. Get Message Types for BOTH topics
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_meta.name: topic_meta.type for topic_meta in topic_types}
            
    if telemetry_topic not in type_map:
        print(f"ERROR: Telemetry topic '{telemetry_topic}' not found in bag.")
        return
    if aimooe_topic not in type_map:
        print(f"WARNING: Aimooe topic '{aimooe_topic}' not found. Did you update the bash script to record it?")

    telemetry_msg_type = get_message(type_map[telemetry_topic])
    
    aimooe_msg_type = None
    if aimooe_topic in type_map:
        aimooe_msg_type = get_message(type_map[aimooe_topic])

    # 3. Data Storage Arrays
    x_data, y_data, z_data = [], [], []
    roll_data, pitch_data, yaw_data = [], [], []
    fre_data, mae_data, telemetry_rmse_data = [], [], []

    # 4. Iterate through messages
    telemetry_count = 0
    aimooe_count = 0
    
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        
        # --- PROCESS CUSTOM TELEMETRY ---
        if topic == telemetry_topic:
            msg = deserialize_message(data, telemetry_msg_type)
            
            x_data.append(msg.pose.position.x * 1000.0)
            y_data.append(msg.pose.position.y * 1000.0)
            z_data.append(msg.pose.position.z * 1000.0)
            
            q = msg.pose.orientation
            r, p, y = euler_from_quaternion(q.x, q.y, q.z, q.w)
            roll_data.append(r)
            pitch_data.append(p)
            yaw_data.append(y)
            
            mae_data.append(msg.mean_abs_error * 1000.0)
            telemetry_rmse_data.append(msg.rms_error * 1000.0)
            telemetry_count += 1
            
        # --- PROCESS AIMOOE TRACKER ---
        elif topic == aimooe_topic and aimooe_msg_type:
            msg = deserialize_message(data, aimooe_msg_type)
            
            # Loop through the array to find the valid tool
            for tool in msg.tools:
                if tool.is_valid:
                    # Note: If Aimooe naturally publishes in mm instead of meters, 
                    # remove the '* 1000.0' below so your data isn't massively scaled!
                    fre_data.append(tool.rms_error * 1000.0) 
                    aimooe_count += 1
                    break # Stop looping once we find the tracked tool

    if telemetry_count == 0:
        print("No valid telemetry messages found to process.")
        return

    # 5. Calculate Final Statistics for the Thesis Table
    print("=========================================================")
    print(f" EXPERIMENT RESULTS: {os.path.basename(bag_path)}")
    print(f" Telemetry Samples: {telemetry_count} | Aimooe Samples: {aimooe_count}")
    print("=========================================================\n")

    print("[1] SENSOR ACCURACY METRICS (Means)")
    
    if aimooe_count > 0:
        print(f"    Fiducial Registration Error (FRE)\t\t: {np.mean(fre_data):.4f} mm")
    else:
        print("    Fiducial Registration Error (FRE)\t\t: [MISSING - NO AIMOOE DATA]")
        
    print(f"    Absolute Euclidean Error (MAE)\t\t: {np.mean(mae_data):.4f} mm")
    print(f"    Telemetry RMSE (Custom C++ SVD Math)\t: {np.mean(telemetry_rmse_data):.4f} mm\n")

    print("[2] POSE JITTER METRICS (Standard Deviations)")
    std_x, std_y, std_z = np.std(x_data), np.std(y_data), np.std(z_data)
    max_translational_jitter = max(std_x, std_y, std_z)
    
    print(f"    Std Dev X : {std_x:.4f} mm")
    print(f"    Std Dev Y : {std_y:.4f} mm")
    print(f"    Std Dev Z : {std_z:.4f} mm")
    print("    ---------------------------------")
    print(f"    >> MAX POSE JITTER (For Table) : {max_translational_jitter:.4f} mm <<")
    print("    ---------------------------------")
    print(f"    Std Dev Roll  : {np.std(roll_data):.4f} deg")
    print(f"    Std Dev Pitch : {np.std(pitch_data):.4f} deg")
    print(f"    Std Dev Yaw   : {np.std(yaw_data):.4f} deg\n")
    print("=========================================================")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: ./process_exp1.py <path_to_bag_folder> <telemetry_topic>")
        sys.exit(1)

    bag_folder = sys.argv[1]
    telemetry_topic = sys.argv[2]
    
    if not os.path.isdir(bag_folder):
        print(f"ERROR: The folder '{bag_folder}' does not exist.")
        sys.exit(1)
        
    process_bag(bag_folder, telemetry_topic)