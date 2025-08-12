#!/usr/bin/env python3
"""
Database initialization script for PostgreSQL
This script will create all necessary tables for the dog recognition system
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(__file__))

from app.core.database import init_db, close_db
from app.core.config import settings

async def main():
    """Initialize the database"""
    print(f"Initializing database: {settings.postgresql_db}")
    print(f"Host: {settings.postgresql_host}")
    print(f"User: {settings.postgresql_user}")
    print(f"Port: {settings.postgresql_port}")
    
    try:
        # Initialize database tables
        await init_db()
        print("✅ Database initialized successfully!")
        
        # Close connections
        await close_db()
        print("✅ Database connections closed")
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 