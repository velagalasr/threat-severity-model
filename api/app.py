"""
Flask REST API for Threat Severity Model

Provides endpoints for:
- /predict: Real-time threat severity prediction
- /explain: Detailed SHAP explanation
- /health: Service health check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import logging
import time
import joblib
import sys
from pathlib import Path
from datetime import datetime

from api.schemas import (
    PredictionRequest, PredictionResponse,
    ExplainRequest, ExplainResponse,
    HealthResponse, ErrorResponse,
    FeatureContribution
)
from api.utils import get_risk_level, get_recommended_action, validate_features
from src.config import API_CONFIG

# Fix pickle module references for models saved with old import paths
import src.model_training
import src.feature_engineering
import src.explainability
sys.modules['model_training'] = src.model_training
sys.modules['feature_engineering'] = src.feature_engineering
sys.modules['explainability'] = src.explainability

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for loaded models
model = None
explainer = None
feature_engineer = None
feature_names = None
start_time = datetime.now()


def load_models():
    """Load trained model and explainer from disk."""
    global model, explainer, feature_engineer, feature_names
    
    try:
        logger.info("Loading models...")
        
        # Load main model
        model_path = API_CONFIG['model_path']
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        model = joblib.load(model_path)
        logger.info(f"[OK] Model loaded from {model_path}")
        
        # Load feature engineer (scaler)
        scaler_path = API_CONFIG['scaler_path']
        if scaler_path.exists():
            feature_engineer = joblib.load(scaler_path)
            logger.info(f"[OK] Feature engineer loaded from {scaler_path}")
        else:
            logger.warning(f"Feature engineer not found at {scaler_path}, using raw features")
        
        # Load SHAP explainer
        explainer_path = API_CONFIG['explainer_path']
        if explainer_path.exists():
            explainer = joblib.load(explainer_path)
            logger.info(f"[OK] SHAP explainer loaded from {explainer_path}")
        else:
            logger.warning(f"SHAP explainer not found at {explainer_path}, explanations disabled")
        
        logger.info("All models loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


@app.route('/')
def index():
    """Root endpoint."""
    return jsonify({
        "service": "Threat Severity Model API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "explain": "/explain"
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    
    Returns:
        JSON response with service health status
    """
    try:
        uptime = (datetime.now() - start_time).total_seconds()
        
        # Get last update time safely
        last_update = None
        try:
            if API_CONFIG['model_path'].exists():
                last_update = datetime.fromtimestamp(
                    API_CONFIG['model_path'].stat().st_mtime
                ).isoformat()
        except Exception as e:
            logger.warning(f"Could not get model update time: {e}")
        
        # Return simple dict instead of Pydantic model
        response_dict = {
            "status": "healthy" if model is not None else "unhealthy",
            "model_loaded": model is not None,
            "model_version": "1.0.0",
            "last_update": last_update,
            "uptime_seconds": uptime,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_dict), 200
    
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        return jsonify({
            "error": "HEALTH_CHECK_ERROR",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict threat severity for network traffic.
    
    Request JSON:
        {
            "features": [f1, f2, ..., f49],
            "include_explanation": false
        }
    
    Returns:
        JSON response with threat score, risk level, and optional explanation
    """
    start_inference = time.time()
    
    try:
        # Validate request
        if not request.is_json:
            error = ErrorResponse(
                error="INVALID_REQUEST",
                message="Request must be JSON"
            )
            return jsonify(error.model_dump()), 400
        
        data = request.get_json()
        
        # Parse request
        try:
            pred_request = PredictionRequest(**data)
        except Exception as e:
            error = ErrorResponse(
                error="VALIDATION_ERROR",
                message=str(e)
            )
            return jsonify(error.model_dump()), 400
        
        # Validate features
        is_valid, error_msg = validate_features(pred_request.features)
        if not is_valid:
            error = ErrorResponse(
                error="INVALID_FEATURES",
                message=error_msg
            )
            return jsonify(error.model_dump()), 400
        
        # Convert to DataFrame with proper column names
        from src.config import COLUMN_NAMES
        feature_cols = COLUMN_NAMES[:-2]  # Exclude attack_type and difficulty_level
        X = pd.DataFrame([pred_request.features], columns=feature_cols)
        
        # Apply feature engineering if available
        if feature_engineer is not None:
            X = feature_engineer.transform(X)
        
        # Make prediction
        threat_score = float(model.predict(X)[0])
        threat_score = np.clip(threat_score, 0, 10)  # Ensure valid range
        
        # Determine risk level and action
        risk_level = get_risk_level(threat_score)
        recommended_action = get_recommended_action(risk_level)
        
        # Get explanation if requested
        top_contributors = None
        if pred_request.include_explanation and explainer is not None:
            try:
                shap_values = explainer.explainer.shap_values(X)
                
                # Get top contributors
                contributions = pd.DataFrame({
                    'feature': X.columns,
                    'shap_value': shap_values[0]
                })
                contributions = contributions.reindex(
                    contributions['shap_value'].abs().sort_values(ascending=False).index
                )
                
                top_contributors = [
                    FeatureContribution(
                        feature=row['feature'],
                        contribution=float(row['shap_value'])
                    ).model_dump()
                    for _, row in contributions.head(5).iterrows()
                ]
                
            except Exception as e:
                logger.warning(f"Failed to generate explanation: {e}")
        
        # Calculate inference time
        inference_time_ms = (time.time() - start_inference) * 1000
        
        # Create response
        response = PredictionResponse(
            threat_score=threat_score,
            risk_level=risk_level,
            recommended_action=recommended_action,
            top_contributors=top_contributors,
            inference_time_ms=inference_time_ms
        )
        
        logger.info(f"Prediction: score={threat_score:.2f}, risk={risk_level}, time={inference_time_ms:.2f}ms")
        
        return jsonify(response.model_dump())
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        error = ErrorResponse(
            error="PREDICTION_ERROR",
            message=str(e)
        )
        return jsonify(error.model_dump()), 500


@app.route('/explain', methods=['POST'])
def explain():
    """
    Get detailed SHAP explanation for prediction.
    
    Request JSON:
        {
            "features": [f1, f2, ..., f49],
            "top_k": 10
        }
    
    Returns:
        JSON response with detailed feature contributions
    """
    start_inference = time.time()
    
    try:
        if explainer is None:
            error = ErrorResponse(
                error="EXPLAINER_NOT_AVAILABLE",
                message="SHAP explainer not loaded"
            )
            return jsonify(error.model_dump()), 503
        
        # Validate request
        if not request.is_json:
            error = ErrorResponse(
                error="INVALID_REQUEST",
                message="Request must be JSON"
            )
            return jsonify(error.model_dump()), 400
        
        data = request.get_json()
        
        # Parse request
        try:
            explain_request = ExplainRequest(**data)
        except Exception as e:
            error = ErrorResponse(
                error="VALIDATION_ERROR",
                message=str(e)
            )
            return jsonify(error.model_dump()), 400
        
        # Convert to DataFrame
        X = pd.DataFrame([explain_request.features])
        
        # Apply feature engineering if available
        if feature_engineer is not None:
            X = feature_engineer.transform(X)
        
        # Make prediction
        threat_score = float(model.predict(X)[0])
        
        # Calculate SHAP values
        shap_values = explainer.explainer.shap_values(X)
        base_value = explainer.explainer.expected_value if hasattr(explainer.explainer, 'expected_value') else 0
        
        # Create detailed contributions
        contributions = []
        for i, (feature, value, shap_val) in enumerate(zip(X.columns, X.iloc[0], shap_values[0])):
            contributions.append({
                'feature': feature,
                'value': float(value),
                'shap_value': float(shap_val),
                'abs_shap_value': float(abs(shap_val))
            })
        
        # Sort by absolute SHAP value
        contributions.sort(key=lambda x: x['abs_shap_value'], reverse=True)
        
        # Take top k
        contributions = contributions[:explain_request.top_k]
        
        # Calculate inference time
        inference_time_ms = (time.time() - start_inference) * 1000
        
        # Create response
        response = ExplainResponse(
            threat_score=threat_score,
            feature_contributions=contributions,
            base_value=float(base_value),
            inference_time_ms=inference_time_ms
        )
        
        return jsonify(response.model_dump())
        
    except Exception as e:
        logger.error(f"Explanation error: {e}", exc_info=True)
        error = ErrorResponse(
            error="EXPLANATION_ERROR",
            message=str(e)
        )
        return jsonify(error.model_dump()), 500


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    error = ErrorResponse(
        error="NOT_FOUND",
        message="Endpoint not found"
    )
    return jsonify(error.model_dump()), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {e}", exc_info=True)
    error = ErrorResponse(
        error="INTERNAL_ERROR",
        message=str(e)
    )
    return jsonify(error.model_dump()), 500


def create_app():
    """Create and configure Flask app."""
    load_models()
    return app


if __name__ == '__main__':
    # Load models on startup
    load_models()
    
    # Run Flask app
    logger.info(f"Starting API server on {API_CONFIG['host']}:{API_CONFIG['port']}")
    app.run(
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        debug=API_CONFIG['debug']
    )

