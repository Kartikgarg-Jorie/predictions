"""
Startup script for Medical Coding Automation Web Application
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.api.em_api import start_server

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" MEDICAL CODING AUTOMATION - WEB APPLICATION")
    print("="*70)
    print("\n Starting FastAPI server...")
    print("\n Access the application at:")
    print("   ► Web Interface: http://localhost:8000")
    print("   ► API Documentation: http://localhost:8000/docs")
    print("   ► API Health Check: http://localhost:8000/health")
    print("\n" + "="*70)
    print("Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    start_server(host="0.0.0.0", port=8000)
