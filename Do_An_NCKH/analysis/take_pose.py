import os
import cv2
import math
import json
import sys
import torch
import numpy as np
import insightface
import torch.nn as nn
from tqdm import tqdm
from insightface.app import FaceAnalysis

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(os.path.join(base_dir, "model"), "wf4m_mbf_rgb.onnx")

model_pack_name = 'buffalo_l'
provider = ['CUDAExecutionProvider']

# Detect face model
detector = FaceAnalysis(name=model_pack_name, provider=provider)
detector.prepare(ctx_id=0, det_size=(640, 640))

# Extract embedding model
handler = insightface.model_zoo.get_model(model_path, provider=provider)
handler.prepare(ctx_id=0)

# Hàm để trích xuất pose
def detect_extract(frame):
    pose = None
    landmark = None
    faces = detector.get(frame)
    for i in faces:
        landmark = i.landmark_2d_106
        pose = i.pose
    if len(faces) == 0:
        return None, None
    return landmark, pose

# def extract_frames(video_path=None):
#     # Path to video
#     video_path = video_path

#     # Array to save frame and embedding
#     frames = []

#     # Read video if path exists, else open camera
#     if video_path is not None:
#         cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         print("Cannot load video.")
#     else:
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         while True:
#             ret, frame = cap.read()
            
#             if not ret:
#                 break

#             frames.append(frame)
            
#             if cv2.waitKey(15) & 0xFF == ord('q'):
#                 break
        
#         cap.release()
#         cv2.destroyAllWindows()

#     return frames

# Hàm trích xuất frames từ video
def extract_frames(video_path=None):
    # Path to video
    video_path = video_path

    # Array to save frame and embedding
    frames = []

    # Read video if path exists, else open camera
    if video_path is not None:
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Cannot load video.")
    else:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_to_skip = fps * 1
        frame_count = 0
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            if frame_count % frame_to_skip == 0:
                frames.append(frame)
            frame_count += 1
            
            if cv2.waitKey(15) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    return frames

# Hàm lấy pose từ frames và thêm vào file JSON
def get_best_embeddings(video_path=None):
    poses = []
    landmarks = []

    # Kiểm tra nếu file đã tồn tại và đọc dữ liệu cũ
    if os.path.exists('landmark.json'):
        with open('pose.json', 'r') as f:
            poses = json.load(f)
        with open('landmark.json', 'r') as f:
            landmarks = json.load(f)

    frames = extract_frames(video_path)
    new_landmark = []
    new_pose = []
    for frame in tqdm(frames):
        landmark, pose = detect_extract(frame)
        if pose is not None:
            # Chuyển đổi pose thành list để lưu vào JSON
            new_landmark.append(landmark.tolist())
            new_pose.append(pose.tolist())

    # Thêm các pose mới vào danh sách poses hiện có
    poses.extend(new_pose)
    landmarks.extend(new_landmark)

    # Lưu tất cả poses (bao gồm dữ liệu cũ và mới) vào file JSON
    with open('landmark.json', 'w') as f:
        json.dump(landmarks, f)

    with open('pose.json', 'w') as f:
        json.dump(poses, f)
    
    print(f"Đã thêm {len(new_landmark)} landmarks vào file landmark.json")
    print(f"Đã thêm {len(new_pose)} landmarks vào file pose.json")

# Sử dụng hàm với video đầu vào
get_best_embeddings("/home/quocnc1/Documents/arcface_test/20241030_153359.mp4")