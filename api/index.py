import os
import sys

# Add project root directory to Python module search path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Expose FastAPI application instance for Vercel serverless runtime
from web.server import app
