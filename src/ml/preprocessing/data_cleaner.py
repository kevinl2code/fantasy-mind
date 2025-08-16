
class DataCleaner:
    @staticmethod
    def clean_yards_column(yards_str):
        """Clean the Yards column by removing commas and converting to int"""
        if isinstance(yards_str, str):
            # Remove commas and convert to int
            return int(yards_str.replace(',', ''))
        return yards_str

    def clean_dataframe(self, df):
        """Clean the entire dataframe"""
        df = df.copy()
        df['Yards'] = df['Yards'].apply(self.clean_yards_column)
        return df