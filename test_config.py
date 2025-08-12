#!/usr/bin/env python3
"""
Simple configuration test script
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(__file__))

try:
    from app.core.config import settings
    print("✅ Configuration loaded successfully!")
    print(f"App name: {settings.app_name}")
    print(f"Version: {settings.version}")
    print(f"Debug: {settings.debug}")
    print(f"Database host: {settings.postgresql_host}")
    print(f"Database port: {settings.postgresql_port}")
    print(f"Database name: {settings.postgresql_db}")
    print(f"Database user: {settings.postgresql_user}")
    print("✅ All configuration values are accessible!")
    
except Exception as e:
    print(f"❌ Configuration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 