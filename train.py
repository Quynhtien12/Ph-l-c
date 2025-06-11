# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Đọc dữ liệu đã lưu
df = pd.read_csv('neck_angle_data.csv')


# Tách dữ liệu đầu vào (X) và nhãn (y)
X = df[['Neck Angle']].values
y = df['label'].values

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình SVM
model = SVC(kernel='linear')

# Train mô hình
model.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = model.predict(X_test)

# Đánh giá mô hình
print("=== Đánh giá mô hình ===")
print(classification_report(y_test, y_pred))
print(f"Độ chính xác: {accuracy_score(y_test, y_pred):.2f}")

# Lưu mô hình sau khi train
joblib.dump(model, 'model.pkl')
print("✅ Mô hình đã được lưu vào model.pkl")
