#!/usr/bin/env python3
"""
Test PostgreSQL integration
This script tests the database connection and basic operations
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(__file__))

from app.core.database import get_db, init_db, close_db
from app.core.config import settings

async def test_database_connection():
    """Test basic database connection"""
    print("🔍 Testing PostgreSQL integration...")
    print(f"📊 Database URL: {settings.postgresql_host}:{settings.postgresql_port}/{settings.postgresql_db}")
    print(f"👤 User: {settings.postgresql_user}")
    print(f"🔒 Port: {settings.postgresql_port}")
    print("=" * 50)
    
    try:
        # Test database initialization
        print("1️⃣ Testing database initialization...")
        await init_db()
        print("✅ Database initialization successful")
        
        # Test getting a database session
        print("2️⃣ Testing database session...")
        async for session in get_db():
            # Test a simple query
            result = await session.execute("SELECT 1 as test")
            row = result.fetchone()
            if row and row.test == 1:
                print("✅ Database session and query successful")
            else:
                print("❌ Database query failed")
            break
        
        # Close connections
        await close_db()
        print("✅ Database connections closed")
        
        print("\n🎉 All PostgreSQL tests passed!")
        
    except Exception as e:
        print(f"❌ PostgreSQL test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

async def main():
    """Run all tests"""
    await test_database_connection()

if __name__ == "__main__":
    asyncio.run(main()) 