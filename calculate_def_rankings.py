import pandas as pd


def add_defensive_rankings(df):
    """
    Add composite_def_score and defensive_rank columns to dataframe.

    Args:
        df: DataFrame with columns for year, yards_allowed_previous_season,
            ypc_previous_season, tds_allowed_previous_season

    Returns:
        DataFrame with added composite_def_score and defensive_rank columns
    """
    df = df.copy()

    # Calculate defensive rankings by year
    for year in df['year'].unique():
        year_mask = df['year'] == year
        year_data = df[year_mask]

        # Rank each metric (lower = better defense)
        yards_rank = year_data['yards_allowed_previous_season'].rank(ascending=True)
        ypc_rank = year_data['ypc_previous_season'].rank(ascending=True)
        tds_rank = year_data['tds_allowed_previous_season'].rank(ascending=True)

        # Convert to percentiles and create weighted average
        # TDs weighted highest (50%), Yards (35%), YPC (15%)
        num_teams = len(year_data)
        composite_score = (
                0.5 * (tds_rank / num_teams) +  # TDs allowed - 50% weight
                0.45 * (yards_rank / num_teams) +  # Yards allowed - 35% weight
                0.05 * (ypc_rank / num_teams)  # YPC allowed - 15% weight
        )
        defensive_rank = composite_score.rank(ascending=True)

        # Add to dataframe
        df.loc[year_mask, 'composite_def_score'] = composite_score.round(3)
        df.loc[year_mask, 'defensive_rank'] = defensive_rank.astype(int)

    return df