from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Create Flask app
app = Flask(__name__)

# Load model
model_pack = joblib.load("models/student_risk_model.pkl")
model = model_pack['model']
columns = model_pack['feature_columns']

print("✅ Model loaded successfully")
print(f"📊 Expecting {len(columns)} features")

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/', methods=['GET'])
def home():
    """API home page"""
    return jsonify({
        'message': 'Student Risk Prediction API',
        'version': '1.0',
        'endpoints': {
            '/predict': 'POST - Get risk prediction for single student',
            '/predict_batch': 'POST - Get predictions for multiple students',
            '/health': 'GET - Check API health'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict risk for a single student
    
    Expected JSON format:
    {
        "attendance_percent": 65,
        "avg_internal_marks": 72,
        "num_backlogs": 1,
        "marks_trend": -5,
        ...
    }
    """
    try:
        # Get data from request
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Prepare features (same as training)
        df = pd.get_dummies(df, drop_first=True)
        df = df.reindex(columns=columns, fill_value=0)
        
        # Predict
        risk_score = model.predict_proba(df)[0, 1]
        risk_category = 'High' if risk_score >= 0.7 else ('Medium' if risk_score >= 0.4 else 'Low')
        
        return jsonify({
            'risk_score': float(risk_score),
            'risk_category': risk_category,
            'success': True
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict risk for multiple students
    
    Expected JSON format:
    {
        "students": [
            {"attendance_percent": 65, "avg_internal_marks": 72, ...},
            {"attendance_percent": 80, "avg_internal_marks": 85, ...}
        ]
    }
    """
    try:
        # Get data from request
        data = request.json
        
        if not data or 'students' not in data:
            return jsonify({'error': 'No students data provided'}), 400
        
        students = data['students']
        
        # Convert to DataFrame
        df = pd.DataFrame(students)
        
        # Prepare features
        df = pd.get_dummies(df, drop_first=True)
        df = df.reindex(columns=columns, fill_value=0)
        
        # Predict
        risk_scores = model.predict_proba(df)[:, 1]
        
        # Create results
        results = []
        for i, score in enumerate(risk_scores):
            results.append({
                'student_index': i,
                'risk_score': float(score),
                'risk_category': 'High' if score >= 0.7 else ('Medium' if score >= 0.4 else 'Low')
            })
        
        return jsonify({
            'predictions': results,
            'total': len(results),
            'success': True
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

# ============================================
# RUN SERVER
# ============================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 STARTING STUDENT RISK PREDICTION API")
    print("="*60)
    print("\nAPI will be available at:")
    print("  Local:   http://localhost:5000")
    print("  Network: http://0.0.0.0:5000")
    print("\nEndpoints:")
    print("  GET  /         - API information")
    print("  GET  /health   - Health check")
    print("  POST /predict  - Single prediction")
    print("  POST /predict_batch - Batch predictions")
    print("\n" + "="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)