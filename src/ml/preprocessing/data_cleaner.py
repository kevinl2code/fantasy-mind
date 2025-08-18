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

        # Convert column names to lowercase for consistency
        df.columns = df.columns.str.lower()

        # Now clean the yards column (lowercase)
        df['yards'] = df['yards'].apply(self.clean_yards_column)
        return df