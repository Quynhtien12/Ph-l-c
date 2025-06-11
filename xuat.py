import cv2
import mediapipe as mp
import math
import os
import pandas as pd

# Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# Kích thước ảnh chuẩn
RESIZE_WIDTH = 480
RESIZE_HEIGHT = 320

# Hàm lấy tọa độ landmark theo kích thước ảnh
def get_landmark_coords(landmarks, idx):
    if landmarks and idx < len(landmarks):
        x = int(landmarks[idx].x * RESIZE_WIDTH)
        y = int(landmarks[idx].y * RESIZE_HEIGHT)
        return [x, y]
    else:
        return None

# Hàm tính góc ở cổ (neck)
def calculate_neck_angle(nose, neck):
    # Tạo vector tham chiếu nằm ngang
    reference_vector = [1, 0]  # Vector ngang (1, 0)
    
    # Vector từ cổ đến mũi
    neck_to_nose = [nose[0] - neck[0], nose[1] - neck[1]]

    # Tính tích vô hướng (dot product)
    dot = neck_to_nose[0] * reference_vector[0] + neck_to_nose[1] * reference_vector[1]
    
    # Tính độ dài vector
    norm_n2n = math.sqrt(neck_to_nose[0] ** 2 + neck_to_nose[1] ** 2)
    norm_ref = math.sqrt(reference_vector[0] ** 2 + reference_vector[1] ** 2)
    
    # Tính cos góc
    cos_angle = dot / (norm_n2n * norm_ref + 1e-6)
    
    # Chuyển đổi sang độ
    angle = math.degrees(math.acos(max(min(cos_angle, 1.0), -1.0)))

    # Nếu mũi thấp hơn cổ (cúi đầu), góc âm
    if nose[1] > neck[1]:
        angle = -angle
    
    return angle

# Dữ liệu lưu trữ
data = []

# Thư mục chứa dữ liệu hình ảnh
for label in ['Normal', 'Sleep']:
    folder = f'Data/{label}'
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        image = cv2.imread(path)

        if image is None:
            print(f"⚠️ Không thể đọc ảnh: {filename}")
            continue

        original_height, original_width = image.shape[:2]
        
        # Resize ảnh nếu chưa đúng kích thước
        if original_width != RESIZE_WIDTH or original_height != RESIZE_HEIGHT:
            print(f"🔄 Đang resize ảnh: {filename}")
            image = cv2.resize(image, (RESIZE_WIDTH, RESIZE_HEIGHT))
        else:
            print(f"✅ Ảnh đã đúng kích thước: {filename}")

        # Xử lý ảnh với MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            
            try:
                # Lấy tọa độ các landmark
                shoulder = get_landmark_coords(lm, mp_pose.PoseLandmark.LEFT_SHOULDER)
                right_shoulder = get_landmark_coords(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER)
                nose = get_landmark_coords(lm, mp_pose.PoseLandmark.NOSE)

                # Tính trung điểm của hai vai để ra cổ
                if shoulder and right_shoulder:
                    neck = [(shoulder[0] + right_shoulder[0]) // 2,
                            (shoulder[1] + right_shoulder[1]) // 2]
                else:
                    print(f"⚠️ Thiếu dữ liệu cổ trong ảnh: {filename}")
                    continue

                # Tính góc tại cổ (giống như đặt thước đo độ lên cổ)
                neck_angle = calculate_neck_angle(nose, neck)

                # Lưu vào danh sách
                data.append([filename, neck_angle, label]) 

            except Exception as e:
                print(f"⚠️ Lỗi xử lý ảnh: {filename} | Chi tiết: {str(e)}")
                continue

# Xuất dữ liệu ra file CSV
df = pd.DataFrame(data, columns=['filename', 'Neck Angle', 'label'])
df.to_csv('neck_angle_data.csv', index=False)
print("✅ Đã xuất xong dữ liệu góc tại cổ kèm tên hình.")
