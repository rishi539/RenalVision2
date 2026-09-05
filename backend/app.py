import os
import sys

# Ensure backend/src is first in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from flask import Flask, request, jsonify, session
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction_pipeline import PredictionPipeline
from cnnClassifier.utils.db_manager import (
    init_db,
    register_user,
    authenticate_user,
    get_user_history,
    add_scan_record
)

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8') 

app = Flask(__name__)
app.secret_key = 'super-secret-renalvision-key'
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5173", "http://127.0.0.1:5174"]}}, supports_credentials=True)

# Initialize SQLAlchemy SQLite database
init_db()


class ClientApp:
    def __init__(self):
        self.filename = os.path.join(os.path.dirname(__file__), "inputImage.jpg")
        self.classifier = PredictionPipeline(self.filename)


# Initialize classifier singleton
clApp = None


def get_classifier():
    global clApp
    if clApp is None:
        clApp = ClientApp()
    return clApp


@app.route("/", methods=['GET'])
def home():
    return jsonify({
        "service": "Renalvision Deep Learning Patient Portal API",
        "status": "online",
        "version": "2.1.0"
    })


@app.route("/api/auth/register", methods=['POST'])
def registerRoute():
    data = request.json or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    name = data.get("name", "").strip()
    email = data.get("email", "").strip()

    if not username or not password or not name or not email:
        return jsonify({"error": "Name, username, email, and password are required"}), 400

    result = register_user(username=username, email=email, password=password, name=name)
    if "error" in result:
        return jsonify(result), 400

    session['user_id'] = result['id']

    return jsonify({
        "status": "success",
        "user": result
    }), 201


@app.route("/api/auth/login", methods=['POST'])
def loginRoute():
    data = request.json or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    user = authenticate_user(username=username, password=password)
    if not user:
        return jsonify({"error": "Invalid username or password."}), 401

    session['user_id'] = user['id']

    return jsonify({
        "status": "success",
        "user": user
    }), 200

@app.route("/api/auth/logout", methods=['POST'])
def logoutRoute():
    session.clear()
    return jsonify({"status": "success", "message": "Logged out"}), 200


@app.route("/api/user/<int:user_id>/history", methods=['GET'])
def getUserHistoryRoute(user_id):
    history = get_user_history(user_id)
    if not history:
        return jsonify({"error": "User record not found"}), 404
    return jsonify(history)


@app.route("/api/user/<int:user_id>/recommendations/gemini", methods=['POST', 'OPTIONS'])
def triggerGeminiRecommendationRoute(user_id):
    if request.method == 'OPTIONS':
        return '', 204
    from cnnClassifier.utils.db_manager import trigger_gemini_recommendation_for_user
    try:
        rec = trigger_gemini_recommendation_for_user(user_id)
        if not rec:
            return jsonify({"error": "Failed to generate recommendation"}), 400
        return jsonify({"status": "success", "recommendations": rec})
    except Exception as e:
        return jsonify({"error": f"Gemini AI Service Error: {str(e)}"}), 500



@app.route("/api/scan/<int:scan_id>", methods=['DELETE', 'OPTIONS'])
def deleteScanRoute(scan_id):
    if request.method == 'OPTIONS':
        return '', 204
    from cnnClassifier.utils.db_manager import delete_scan_record
    success = delete_scan_record(scan_id)
    if success:
        return jsonify({"status": "success", "message": "Scan deleted"}), 200
    else:
        return jsonify({"error": "Failed to delete scan"}), 500


@app.route("/predict", methods=['POST'])
def predictRoute():
    data = request.json or {}
    image_b64 = data.get('image')
    if not image_b64:
        return jsonify({"error": "Image data is required"}), 400

    user_id = data.get('user_id')
    client = get_classifier()

    decodeImage(image_b64, client.filename)
    result = client.classifier.predict()

    # Automatically save scan to logged in patient's personal history
    if user_id and result:
        res_data = result[0]
        try:
            add_scan_record(
                user_id=int(user_id),
                prediction=res_data.get("image", "Unknown"),
                confidence=res_data.get("confidence", 0.0),
                probabilities=res_data.get("probabilities", {}),
                gradcam_b64=res_data.get("gradcam"),
                original_b64=image_b64
            )
        except Exception as e:
            print(f"Failed to log scan record: {e}")

    return jsonify(result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
