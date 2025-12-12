# file: app.py
# Nguyễn Việt Quang B22DCCN650
import os

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import joblib  # Dùng để tải StandardScaler

# ---------------- Cấu hình ----------------
MODEL_PATH = "../best_B_Deep_BN_DO.h5"
# Đường dẫn tạm thời lưu scaler. Trong code train của bạn chưa lưu scaler, 
# nên tôi sẽ tạo và tải lại scaler bằng dữ liệu đã chuẩn hóa.
# THAY THẾ bằng đường dẫn SCALER.SAVE nếu bạn đã lưu nó trong tập lệnh train.
SCALER_DUMP_PATH = "../scaler_diabetes.joblib"

app = Flask(__name__)

# ---------------- Khởi tạo Mô hình và Scaler ----------------
# Tải mô hình
try:
    model = keras.models.load_model(MODEL_PATH)
    model.summary()
except Exception as e:
    print(f"Lỗi khi tải mô hình {MODEL_PATH}: {e}")
    model = None

# Tải lại StandardScaler (vì bạn không lưu trong code train, tôi giả định bạn có thể
# tái tạo lại scaler từ dữ liệu gốc đã được làm sạch)
try:
    # 1. Tải dữ liệu gốc đã làm sạch (như trong tập lệnh train của bạn)
    df = pd.read_csv("../diabetes.csv")
    cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for c in cols_with_zero:
        df[c] = df[c].replace(0, df[c].median())

    X = df.drop("Outcome", axis=1)

    # 2. Fit và lưu Scaler
    scaler = StandardScaler()
    scaler.fit(X)
    joblib.dump(scaler, SCALER_DUMP_PATH)
    print(f"Đã tạo lại và lưu StandardScaler vào {SCALER_DUMP_PATH}")

except Exception as e:
    print(f"Lỗi khi tái tạo StandardScaler: {e}")
    scaler = None


# ---------------- Định tuyến API ----------------

@app.route('/')
def home():
    """Hiển thị form nhập liệu."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Xử lý yêu cầu dự đoán từ form."""
    if model is None or scaler is None:
        return jsonify({"error": "Model or Scaler not loaded."}), 500

    try:
        # Lấy dữ liệu từ form
        data = request.form.to_dict()

        # Đảm bảo thứ tự cột phải khớp với thứ tự huấn luyện
        # ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        features = [
            float(data['Pregnancies']),
            float(data['Glucose']),
            float(data['BloodPressure']),
            float(data['SkinThickness']),
            float(data['Insulin']),
            float(data['BMI']),
            float(data['DiabetesPedigreeFunction']),
            float(data['Age'])
        ]

        # Chuyển đổi thành mảng numpy và chuẩn hóa
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)

        # Dự đoán
        prediction_prob = model.predict(features_scaled)[0][0]
        prediction_class = "Có (Dương tính)" if prediction_prob >= 0.5 else "Không (Âm tính)"

        result = {
            "prediction_class": prediction_class,
            "probability": f"{prediction_prob:.4f}",
            "model_used": MODEL_PATH
        }

        return render_template('index.html',
                               prediction_text=f'Kết quả dự đoán: {prediction_class} (Xác suất: {prediction_prob:.4f})')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Lỗi xử lý dữ liệu: {e}')
        # return jsonify({"error": str(e)}), 400


# ---------------- Chạy ứng dụng ----------------
if __name__ == '__main__':
    # Tạo thư mục 'templates' nếu chưa có
    os.makedirs('templates', exist_ok=True)

    # Nếu chưa có file mô hình hoặc scaler, app sẽ không chạy
    if model and scaler:
        print("Mô hình và Scaler đã sẵn sàng. Chạy Flask...")
        app.run(debug=True)
    else:
        print("Lỗi: Không tìm thấy mô hình hoặc scaler. Vui lòng kiểm tra file đầu vào.")