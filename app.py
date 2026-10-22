from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd
from preprocessing import preprocess_input
import subprocess
import sys

app = Flask(__name__)

model = joblib.load("final_model.pkl")

label_map = {
    0: "low",
    1: "medium",
    2: "high"
}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_input = request.get_json()

        if not user_input:
            return jsonify({"error": "No JSON input received"}), 400

        X_new = preprocess_input(user_input)

        pred_class = int(model.predict(X_new)[0])
        probs = model.predict_proba(X_new)[0]

        result = {
            "predicted_class": pred_class,
            "predicted_category": label_map[pred_class],
            "probabilities": {
                "low": float(probs[0]),
                "medium": float(probs[1]),
                "high": float(probs[2])
            }
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/feedback", methods=["POST"])
def feedback():
    global model

    try:
        feedback_input = request.get_json()

        if not feedback_input:
            return jsonify({"error": "No JSON input received"}), 400

        required_fields = [
            "age",
            "grade",
            "ses_quartile",
            "math_score",
            "reading_score",
            "science_score",
            "attendance_rate",
            "study_hours",
            "internet_access",
            "extracurricular",
            "part_time_job",
            "parent_support",
            "romantic",
            "free_time",
            "go_out",
            "gender",
            "race",
            "school_type",
            "parental_education",
            "locale",
            "true_label"
        ]

        for field in required_fields:
            if field not in feedback_input:
                return jsonify({"error": f"Missing field: {field}"}), 400

        feedback_file = "feedback_data.csv"
        row = pd.DataFrame([feedback_input])

        if os.path.exists(feedback_file):
            existing = pd.read_csv(feedback_file)
            updated = pd.concat([existing, row], ignore_index=True)
            updated.to_csv(feedback_file, index=False)
        else:
            updated = row
            updated.to_csv(feedback_file, index=False)

        feedback_count = len(updated)

        retrained = False
        if feedback_count % 5 == 0:
            subprocess.run([sys.executable, "retrain_model.py"], check=True)
            model = joblib.load("final_model.pkl")
            retrained = True

        return jsonify({
            "message": "Feedback saved successfully",
            "feedback_count": feedback_count,
            "retrained": retrained
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
