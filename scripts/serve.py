"""
API Server Launcher

Starts the Flask API server for inference.
"""

import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.app import create_app
from src.config import API_CONFIG

if __name__ == '__main__':
    app = create_app()
    
    # Debug: Print all registered routes
    print("\n[DEBUG] Registered Flask routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule.methods} {rule.rule}")
    
    print(f"\n{'='*60}")
    print("THREAT SEVERITY MODEL API")
    print(f"{'='*60}")
    print(f"Server: http://{API_CONFIG['host']}:{API_CONFIG['port']}")
    print(f"\nEndpoints:")
    print(f"  POST /predict    - Predict threat severity")
    print(f"  POST /explain    - Get detailed SHAP explanation")
    print(f"  GET  /health     - Health check")
    print(f"\nExample:")
    print(f'  curl -X POST http://localhost:5000/predict \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"features": [0.1, 0.2, ..., 0.9], "include_explanation": true}}\'')
    print(f"{'='*60}\n")
    
    app.run(
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        debug=API_CONFIG['debug']
    )
