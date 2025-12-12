import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import tensorflow as tf

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# --- Cấu hình và tải mô hình ---
MODEL_FILE_JOBLIB = 'best_model.joblib'
MODEL_FILE_KERAS = 'best_model.keras'
PREPROCESSOR_FILE = 'preprocessor.joblib'
LABEL_ENCODER_FILE = 'label_encoder.joblib'

# Kiểm tra xem các file cần thiết có tồn tại không
if not os.path.exists(PREPROCESSOR_FILE) or not os.path.exists(LABEL_ENCODER_FILE):
    print("Lỗi: Không tìm thấy các file tiền xử lý và mã hóa. Vui lòng chạy file đánh giá mô hình trước.")
    exit()

# Tải các thành phần đã lưu
try:
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    label_encoder = joblib.load(LABEL_ENCODER_FILE)

    # Tải mô hình tốt nhất, kiểm tra cả hai định dạng
    if os.path.exists(MODEL_FILE_JOBLIB):
        best_model = joblib.load(MODEL_FILE_JOBLIB)
        model_type = "joblib"
        print(f"Đã tải mô hình tốt nhất từ '{MODEL_FILE_JOBLIB}'.")
    elif os.path.exists(MODEL_FILE_KERAS):
        best_model = tf.keras.models.load_model(MODEL_FILE_KERAS)
        model_type = "keras"
        print(f"Đã tải mô hình tốt nhất từ '{MODEL_FILE_KERAS}'.")
    else:
        print("Lỗi: Không tìm thấy file mô hình. Vui lòng đảm bảo bạn đã lưu mô hình tốt nhất.")
        exit()

except Exception as e:
    print(f"Lỗi khi tải các file đã lưu: {e}")
    exit()


# Định nghĩa các tuyến đường (routes)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Lấy dữ liệu từ form
            input_data = {
                'age': float(request.form['age']),
                'height_cm': float(request.form['height_cm']),
                'weight_kg': float(request.form['weight_kg']),
                'years_experience': float(request.form['years_experience']),
                'salary': float(request.form['salary']),
                'gender': request.form['gender'],
                'job': request.form['job'],
                'education_level': request.form['education_level']
            }

            # Chuyển dữ liệu đầu vào thành DataFrame
            features = ['age', 'height_cm', 'weight_kg', 'years_experience', 'salary', 'gender', 'job',
                        'education_level']
            input_df = pd.DataFrame([input_data], columns=features)

            # Áp dụng bộ tiền xử lý đã lưu
            input_processed = preprocessor.transform(input_df)

            # Dự đoán
            if model_type == "keras":
                # Reshape cho các mô hình CNN/LSTM nếu cần
                if "CNN" in best_model.name:
                    input_processed = input_processed.reshape(1, input_processed.shape[1], 1)
                elif "LSTM" in best_model.name:
                    input_processed = input_processed.reshape(1, 1, input_processed.shape[1])

                # Dự đoán xác suất và tìm lớp có xác suất cao nhất
                prediction_encoded = best_model.predict(input_processed)
                prediction_index = np.argmax(prediction_encoded)
                # Chuyển đổi chỉ số thành nhãn
                prediction = label_encoder.inverse_transform([[prediction_index]])[0][0]
            else:  # joblib
                prediction = best_model.predict(input_processed)[0]

            return jsonify({'prediction': prediction})

        except Exception as e:
            return jsonify({'error': str(e)}), 400

    # Trả về trang HTML cho yêu cầu GET
    return render_template('index.html')


if __name__ == '__main__':
    # Chạy ứng dụng Flask
    app.run(debug=True)
