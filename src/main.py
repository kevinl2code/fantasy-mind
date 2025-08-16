# main.py
from fastapi import FastAPI

import pandas as pd
from dotenv import load_dotenv
from calculate_def_rankings import add_defensive_rankings
from database import get_snowflake_connection
from api import (
    # auth_router,
    train_router,
)
load_dotenv()

# Create FastAPI app
app = FastAPI(title="FantasyMind API", version="1.0.0")

# Include routers
# app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(train_router, prefix="/train", tags=["ml-training"])

@app.get("/")
async def root():
    """API health check"""
    return {"message": "FantasyMind API is running!"}


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