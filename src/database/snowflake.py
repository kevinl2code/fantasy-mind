import os
import snowflake.connector
from fastapi import HTTPException

def get_snowflake_connection():
    """Connect to Snowflake database"""
    try:
        conn = snowflake.connector.connect(
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            user=os.getenv('SNOWFLAKE_USER'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
            database=os.getenv('SNOWFLAKE_DATABASE'),
            schema=os.getenv('SNOWFLAKE_SCHEMA'),
            role=os.getenv('SNOWFLAKE_ROLE')
        )
        return conn
    except Exception as e:
        print(f"Snowflake connection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Snowflake connection failed: {e}")

# Optional: Add other Snowflake-specific utilities
def close_snowflake_connection(conn):
    """Safely close Snowflake connection"""
    try:
        if conn:
            conn.close()
    except Exception as e:
        print(f"Error closing Snowflake connection: {e}")

def execute_snowflake_query(query: str, params=None):
    """Execute a query and return results"""
    conn = get_snowflake_connection()
    try:
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        results = cursor.fetchall()
        return results
    finally:
        close_snowflake_connection(conn)