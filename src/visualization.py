import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from src.logger import logger


def plot_histograms(df, num_cols, output_dir):
    try:
        df[num_cols].hist(figsize=(14, 14))
        plt.savefig(os.path.join(output_dir, 'histograms.png'))
        plt.close()
        logger.info("Histograms saved successfully")
    except Exception as e:
        logger.error(f"Error plotting histograms: {e}")
        raise


def plot_categorical(df, cat_cols, output_dir):
    try:
        for i in cat_cols:
            if i != 'Attrition':
                (pd.crosstab(df[i], df['Attrition'], normalize='index') * 100).plot(kind='bar', figsize=(8,4), stacked=True)
                plt.ylabel('Percentage Attrition %')
                plt.savefig(os.path.join(output_dir, f'{i}_attrition.png'))
                plt.close()
        logger.info("Categorical plots saved successfully")
    except Exception as e:
        logger.error(f"Error plotting categorical variables: {e}")
        raise


def plot_correlation(df, num_cols, output_dir):
    try:
        plt.figure(figsize=(15, 8))
        sns.heatmap(df[num_cols].corr(), annot=True, fmt='0.2f', cmap='YlGnBu')
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
        plt.close()
        logger.info("Correlation heatmap saved successfully")
    except Exception as e:
        logger.error(f"Error plotting correlation heatmap: {e}")
        raise
