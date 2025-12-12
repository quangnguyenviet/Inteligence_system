from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
# Nguyen Viet Quang
# Load model
model = pickle.load(open("diabetes_model.sav", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json
        sample = pd.DataFrame([data.values()],
                              columns=['Glucose','BMI','Age','Pregnancies','SkinThickness'])
        prediction = model.predict(sample)[0]
        proba = model.predict_proba(sample)[0][prediction]

        result = "🩺 Diabetic" if prediction == 1 else "✅ Non-Diabetic"
        return jsonify({"result": result, "confidence": f"{proba:.2%}"})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
