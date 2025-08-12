# main.py
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import snowflake.connector
import pandas as pd
import os

from calculate_def_rankings import add_defensive_rankings

# Load environment variables from .env file
load_dotenv()

# Create FastAPI app
app = FastAPI(title="Snowflake Test", version="1.0.0")


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
        print(f"Connection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")


@app.get("/")
async def home():
    """Simple home page"""
    return {"message": "Snowflake Test API is running!"}


@app.get("/test-connection")
async def test_connection():
    """Test if we can connect to Snowflake"""
    try:
        conn = get_snowflake_connection()
        conn.close()
        return {"status": "success", "message": "Connected to Snowflake successfully!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/show-tables")
async def show_tables():
    """Show what tables exist in our database"""
    try:
        conn = get_snowflake_connection()

        # Simple query to see what tables we have
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()

        # Convert to a simple list
        table_list = [table[1] for table in tables]  # table[1] is the table name

        cursor.close()
        conn.close()

        return {
            "status": "success",
            "tables": table_list,
            "count": len(table_list)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/sample-data/{table_name}")
async def get_sample_data(table_name: str):
    """Get a few rows from a specific table"""
    try:
        conn = get_snowflake_connection()

        # Get first 5 rows from the table
        query = f"SELECT * FROM {table_name} LIMIT 5"
        df = pd.read_sql(query, conn)

        conn.close()

        # Convert to dictionary format
        data = df.to_dict('records')

        return {
            "status": "success",
            "table": table_name,
            "row_count": len(data),
            "columns": list(df.columns),
            "sample_data": data
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/calculate-rankings")
async def calculate_rankings():
    """Load data, calculate rankings, and update the Snowflake table"""
    try:
        conn = get_snowflake_connection()

        # Step 1: Load all data from the table
        query = "SELECT * FROM YEARLY_TEAM_RUN_DEF"
        df = pd.read_sql(query, conn)

        print(f"Loaded {len(df)} rows from Snowflake")

        # Step 2: Apply your ranking function
        # Note: your function expects lowercase column names
        df_lowercase = df.copy()
        df_lowercase.columns = df_lowercase.columns.str.lower()

        df_with_rankings = add_defensive_rankings(df_lowercase)

        print("Rankings calculated successfully")

        # Step 3: Update the table with new columns
        cursor = conn.cursor()

        # Add new columns if they don't exist
        try:
            cursor.execute("ALTER TABLE YEARLY_TEAM_RUN_DEF ADD COLUMN COMPOSITE_DEF_SCORE FLOAT")
            print("Added COMPOSITE_DEF_SCORE column")
        except:
            print("COMPOSITE_DEF_SCORE column already exists")

        try:
            cursor.execute("ALTER TABLE YEARLY_TEAM_RUN_DEF ADD COLUMN DEFENSIVE_RANK INTEGER")
            print("Added DEFENSIVE_RANK column")
        except:
            print("DEFENSIVE_RANK column already exists")

        # Step 4: Update each row with the calculated values
        for _, row in df_with_rankings.iterrows():
            update_query = """
            UPDATE YEARLY_TEAM_RUN_DEF 
            SET COMPOSITE_DEF_SCORE = %s, DEFENSIVE_RANK = %s 
            WHERE YEAR = %s AND TEAM_ID = %s
            """
            cursor.execute(update_query, (
                row['composite_def_score'],
                row['defensive_rank'],
                row['year'],
                row['team_id']
            ))

        # Commit the changes
        conn.commit()
        cursor.close()
        conn.close()

        return {
            "status": "success",
            "message": f"Updated {len(df_with_rankings)} rows with defensive rankings",
            "sample_rankings": df_with_rankings[
                ['team', 'year', 'composite_def_score', 'defensive_rank']].head().to_dict('records')
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)