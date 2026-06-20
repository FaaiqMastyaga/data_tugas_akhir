#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
import numpy as np
import math
import time
import sys

class SpatialValidator(Node):
    def __init__(self, target_frame, source_frame, gt_x, gt_y, gt_z, duration=3.0):
        super().__init__('spatial_validator')
        
        # Initialize TF2 Buffer and Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.target_frame = target_frame
        self.source_frame = source_frame
        self.gt_x = gt_x
        self.gt_y = gt_y
        self.gt_z = gt_z
        self.duration = duration
        
        # Arrays to store the live readings
        self.x_data = []
        self.y_data = []
        self.z_data = []
        
        self.get_logger().info(f"Mengambil data {target_frame} -> {source_frame} selama {duration} detik. Mohon tahan instrumen...")
        
        # Timer to poll TF at 10 Hz
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.start_time = time.time()

    def timer_callback(self):
        # Stop recording after duration is reached
        if time.time() - self.start_time > self.duration:
            self.timer.cancel()
            self.calculate_and_print()
            rclpy.shutdown()
            return

        try:
            # Look up the transform at the current time
            t = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                rclpy.time.Time())
            
            # Convert meters to millimeters and store
            self.x_data.append(t.transform.translation.x * 1000.0)
            self.y_data.append(t.transform.translation.y * 1000.0)
            self.z_data.append(t.transform.translation.z * 1000.0)

        except Exception as e:
            # It's normal for TF to occasionally drop a frame or not be ready
            pass

    def calculate_and_print(self):
        if not self.x_data:
            print(f"\n[ERROR] Tidak ada data TF yang diterima. Pastikan frame '{self.target_frame}' dan '{self.source_frame}' aktif.")
            return
            
        # 1. Average the readings
        mean_x = np.mean(self.x_data)
        mean_y = np.mean(self.y_data)
        mean_z = np.mean(self.z_data)
        
        # 2. Calculate the Errors
        err_x = mean_x - self.gt_x
        err_y = mean_y - self.gt_y
        err_z = mean_z - self.gt_z

        # 3. Calculate Relative Positional Error (Euclidean Distance)
        rpe = math.sqrt(err_x**2 + err_y**2 + err_z**2)
        
        # 4. Print Table-Ready Output
        print("\n" + "="*55)
        print(f" HASIL EKSPERIMEN 2: Titik ({self.gt_x}, {self.gt_y}, {self.gt_z}) mm")
        print("="*55)
        print(f" Jumlah Sampel Diambil : {len(self.x_data)}")
        print(f" Titik Sampel (GT)     : X = {self.gt_x:8.3f} mm, Y = {self.gt_y:8.3f} mm, Z = {self.gt_z:8.3f} mm")
        print(f" Pembacaan Kamera      : X = {mean_x:8.3f} mm, Y = {mean_y:8.3f} mm, Z = {mean_z:8.3f} mm")
        print("-" * 55)
        print(f" Error X               : {err_x:8.3f} mm")
        print(f" Error Y               : {err_y:8.3f} mm")
        print(f" Error Z               : {err_z:8.3f} mm")
        print(f" >> RPE                : {rpe:8.3f} mm <<")
        print("="*55 + "\n")

def main(args=None):
    # Get user input before initializing ROS
    print("\n--- Karakterisasi Akurasi Instrumen (Otomatis) ---")
    try:
        gt_x = float(input("Masukkan Titik Uji X (mm) [contoh: 100]  : "))
        gt_y = float(input("Masukkan Titik Uji Y (mm) [contoh: 150]  : "))
    except ValueError:
        print("Input tidak valid. Harap masukkan angka.")
        sys.exit(1)

    rclpy.init(args=args)
    
    # Target and source match your tf2_echo command
    node = SpatialValidator(
        target_frame='whiteboard', 
        source_frame='robot_base_aligned', 
        gt_x=gt_x, 
        gt_y=gt_y,
        gt_z=0.0,
        duration=3.0 # Averages data for 3 seconds
    )
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()