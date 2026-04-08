from flask import Flask, request, jsonify
import joblib
import os

# -------------------------
# ✅ Load model and vectorizer
# -------------------------
MODEL_PATH = os.path.join("model", "model.pkl")
VECTORIZER_PATH = os.path.join("model", "vectorizer.pkl")

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✅ Model and vectorizer loaded successfully")
except Exception as e:
    print(f"❌ Error loading model/vectorizer: {e}")
    raise e

# -------------------------
# Flask App
# -------------------------
app = Flask(__name__)

# Home route
@app.route("/", methods=["GET"])
def home():
    return "Sentiment Analysis API is running!"

# Test route
@app.route("/test", methods=["GET"])
def test():
    return "Flask server is working fine!"

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)  # Debug log

        # Input validation
        if not data or "review" not in data:
            return jsonify({"error": "Missing 'review' field in JSON"}), 400

        text = data["review"]
        if not isinstance(text, str) or len(text.strip()) == 0:
            return jsonify({"error": "'review' must be a non-empty string"}), 400

        # Transform text
        text_vec = vectorizer.transform([text])
        prediction = model.predict(text_vec)[0]

        return jsonify({"review": text, "sentiment": prediction}), 200

    except Exception as e:
        print("Error during prediction:", e)  # Debug log
        return jsonify({"error": str(e)}), 500

# -------------------------
# Run Flask App
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)