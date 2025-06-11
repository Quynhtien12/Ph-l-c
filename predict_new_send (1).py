import cv2
import mediapipe as mp
import math
import joblib
import base64
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import time

# Khởi tạo Firebase Admin SDK
cred = credentials.Certificate("sdk_admin.json")
firebase_admin.initialize_app(cred)

# Kết nối tới Firestore
db = firestore.client()
last_date_time = None 

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils


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

# Hàm tính góc tại cổ
def calculate_neck_angle(nose, neck):
    reference_vector = [1, 0]
    neck_to_nose = [nose[0] - neck[0], nose[1] - neck[1]]
    dot = neck_to_nose[0] * reference_vector[0] + neck_to_nose[1] * reference_vector[1]
    norm_n2n = math.sqrt(neck_to_nose[0] ** 2 + neck_to_nose[1] ** 2)
    norm_ref = math.sqrt(reference_vector[0] ** 2 + reference_vector[1] ** 2)
    cos_angle = dot / (norm_n2n * norm_ref + 1e-6)
    angle = math.degrees(math.acos(max(min(cos_angle, 1.0), -1.0)))
    if nose[1] > neck[1]:
        angle = -angle
    return angle

def predict_from_base64(base64_string):
    try:
        # Load mô hình đã train
        model = joblib.load('model.pkl')
        print("✅ Mô hình đã được load thành công.")

        # Giải mã base64 thành ảnh
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Không thể giải mã ảnh từ chuỗi base64.")

        # Resize ảnh về kích thước chuẩn
        image = cv2.resize(image, (RESIZE_WIDTH, RESIZE_HEIGHT))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Xử lý ảnh với MediaPipe
        result = pose.process(image_rgb)

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark

            shoulder = get_landmark_coords(lm, mp_pose.PoseLandmark.LEFT_SHOULDER)
            right_shoulder = get_landmark_coords(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER)
            nose = get_landmark_coords(lm, mp_pose.PoseLandmark.NOSE)

            if shoulder and right_shoulder:
                neck = [(shoulder[0] + right_shoulder[0]) // 2,
                        (shoulder[1] + right_shoulder[1]) // 2]

                # Tính góc ở cổ
                neck_angle = calculate_neck_angle(nose, neck)
                print(f"🧮 Góc đo được: {neck_angle:.2f}°")

                # Dự đoán
                prediction = model.predict([[neck_angle]])
                print(f"🔍 Tư thế dự đoán: {prediction[0]}")

                # Hiển thị ảnh
                # cv2.imshow("image predicted", image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                return prediction[0]
            else:
                raise ValueError("⚠️ Không thể tìm thấy cổ trong hình ảnh.")
        else:
            raise ValueError("⚠️ Không phát hiện được pose landmarks.")

    except Exception as e:
        print(f"⚠️ Lỗi trong quá trình dự đoán: {e}")
        # if 'image' in locals() and image is not None:
        #     cv2.imshow("Image eror", image)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        return None



def get_latest_image_info():

    # Tham chiếu tới collection 'image_data'
    collection_ref = db.collection('image_data')
    
    # Lấy document mới nhất theo 'date_time'
    docs = collection_ref.order_by('date_time', direction=firestore.Query.DESCENDING).limit(1).stream()
    
    for doc in docs:
        data = doc.to_dict()
        # print(f'Document ID: {doc.id}')
        
        date_time = data["date_time"]
        img_base64 = data["image"]  # Giữ nguyên dữ liệu base64 (không giải mã)
        
        # Trả về hai giá trị
        return date_time, img_base64
    
    print("Không tìm thấy ảnh.")
    return None, None

def upload_predict_to_firestore(date_time, predict):

    try:
        doc_ref = db.collection('Predict').document()  # Tạo document ID tự động
        doc_ref.set({
            'date_time': date_time,
            'predict': predict
        })
        print(f"✅ Dự đoán đã được lưu: {predict} tại {date_time}")
    except Exception as e:
        print(f"❌ Lỗi khi lưu dữ liệu dự đoán: {e}")


# ===== VÒNG LẶP NHẬN BASE64 =====
while True:
    date_time, img_base_pre = get_latest_image_info()
        # Kiểm tra nếu thời gian lấy ảnh khác lần trước
    if date_time != last_date_time:
        print(f"📸 Ảnh mới cập nhật:")
        print(f"Thời gian: {date_time}")
        print(f"Kích thước ảnh: {len(img_base_pre)} bytes")
        result = predict_from_base64(img_base_pre)
        print(f"Dự đoán: {result}")
        # Gửi dự đoán lên Firestore
        upload_predict_to_firestore(date_time, result)
        
        # Cập nhật lại thời gian đã xử lý
        last_date_time = date_time
    # else:
    #     print("⏳ Không có ảnh mới, đang chờ cập nhật...")

    time.sleep(5)  # Thời gian chờ trước khi kiểm tra lại
