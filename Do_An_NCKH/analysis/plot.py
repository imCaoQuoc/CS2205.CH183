import json
import numpy as np
import matplotlib.pyplot as plt

# Hàm tính EAR
def calculate_ear(eye_landmarks):
    P1, P2, P3, P4, P5, P6 = eye_landmarks
    A = np.linalg.norm(np.array(P2) - np.array(P6))
    B = np.linalg.norm(np.array(P3) - np.array(P5))
    C = np.linalg.norm(np.array(P1) - np.array(P4))
    ear = (A + B) / (2.0 * C)
    return ear

# Load landmark và pose
with open("landmark.json", "r") as f:
    landmarks_data = json.load(f)

with open("pose.json", "r") as f:
    poses_data = json.load(f)

# Landmark indices cho mắt
LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

# Danh sách chứa EAR và Pose
left_eye_ears = []
right_eye_ears = []
poses = []

# Duyệt qua từng frame
for frame_landmarks, pose in zip(landmarks_data, poses_data):
    frame_landmarks = np.array(frame_landmarks)

    # Lấy landmark của mắt
    left_eye_landmarks = frame_landmarks[LEFT_EYE_INDICES]
    right_eye_landmarks = frame_landmarks[RIGHT_EYE_INDICES]

    # Tính EAR cho từng mắt
    left_ear = calculate_ear(left_eye_landmarks)
    right_ear = calculate_ear(right_eye_landmarks)

    # Lưu kết quả
    left_eye_ears.append(left_ear)
    right_eye_ears.append(right_ear)
    poses.append(pose)  # Pose là [yaw, pitch, roll]

# Chuyển poses thành numpy array để dễ xử lý
poses = np.array(poses)
yaw, pitch, roll = poses[:, 0], poses[:, 1], poses[:, 2]

# Plot kết quả
plt.figure(figsize=(12, 8))

# EAR cho mắt trái
plt.subplot(2, 1, 1)
plt.plot(left_eye_ears, label="Left Eye EAR", color="blue")
plt.plot(right_eye_ears, label="Right Eye EAR", color="orange")
plt.title("Eye Aspect Ratio (EAR) Over Frames")
plt.xlabel("Frame Index")
plt.ylabel("EAR")
plt.legend()
plt.grid(True)

# Pose (yaw, pitch, roll)
plt.subplot(2, 1, 2)
plt.plot(yaw, label="Yaw", color="green")
plt.plot(pitch, label="Pitch", color="red")
plt.plot(roll, label="Roll", color="purple")
plt.title("Pose (Yaw, Pitch, Roll) Over Frames")
plt.xlabel("Frame Index")
plt.ylabel("Pose")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



# import json
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# # Đọc dữ liệu từ file JSON
# with open('poses.json', 'r') as f:
#     data = json.load(f)

# # Tách riêng các giá trị pitch, yaw, roll
# pitch_values = [pose[0] for pose in data]
# yaw_values = [pose[1] for pose in data]
# roll_values = [pose[2] for pose in data]

# # Hàm vẽ barplot với khoảng giá trị là 1 đơn vị
# def plot_distribution(values, title):
#     plt.figure(figsize=(10, 6))
#     # Tạo các bin với khoảng cách 1 đơn vị, từ giá trị nhỏ nhất đến lớn nhất
#     bins = np.arange(int(min(values)), int(max(values)) + 2)  # +2 để bao gồm giá trị lớn nhất
#     sns.histplot(values, bins=bins, kde=False)
#     plt.title(f"Distribution of {title}")
#     plt.xlabel(title)
#     plt.ylabel("Frequency")
#     plt.show()

# # Vẽ barplot cho từng giá trị
# plot_distribution(pitch_values, "Pitch")
# plot_distribution(yaw_values, "Yaw")
# plot_distribution(roll_values, "Roll")
