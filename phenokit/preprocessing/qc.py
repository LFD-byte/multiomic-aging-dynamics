import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class MissingValueAnalyzer:
    """
    A class specifically used for generating missing-value statistical reports.
    """
    def __init__(self, df):
        self.df = df

    def report_features(self, top_n=10):
        """
        “The missing rate of a feature (column).
        """
        # Calculate the missing rate of each column.
        missing_series = self.df.isnull().mean()
        
        stats = pd.DataFrame({
            'missing_rate': missing_series,
            'missing_count': self.df.isnull().sum()
        })
        
        # Sort in descending order by missing rate.
        stats = stats.sort_values(by='missing_rate', ascending=False)
        
        print(f"=== Feature missing-rate report. (Top {top_n}) ===")
        print(stats.head(top_n))
        return stats

    def report_samples(self, top_n=10):
        """
        The missing rate of a sample (row).
        """
        # axis=1 computes the missing values by row.
        missing_series = self.df.isnull().mean(axis=1)
        
        stats = pd.DataFrame({
            'missing_rate': missing_series,
            'missing_count': self.df.isnull().sum(axis=1)
        })
        
        stats = stats.sort_values(by='missing_rate', ascending=False)
        
        print(f"=== Sample missing-rate report. (Top {top_n}) ===")
        print(stats.head(top_n))
        return stats
    
    def overall_missing_rate(self):
        """
        Calculate the overall missing rate of the DataFrame.
        """
        total_elements = self.df.size
        total_missing = self.df.isnull().sum().sum()
        overall_rate = total_missing / total_elements
        
        print(f"Overall missing rate: {overall_rate:.4f} ({total_missing} out of {total_elements})")
        return overall_rate
    
    def report_heatmap(self):
        """
        Generate a heatmap of missing values.
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.df.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.xlabel('Features')
        plt.ylabel('Samples')
        plt.show()