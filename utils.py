import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess the NSSE dataset.
    
    Args:
        filepath (str): Path to the Excel file
        
    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    # Load data
    df = pd.read_excel(filepath, engine="openpyxl")
    
    # Clean string columns
    string_cols = ['MAJfirst', 'MAJsecond', 'gi_another_txt', 'so_another_txt', 
                   'gr_another_txt', 'group1', 'group4']
    for col in string_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Convert numeric columns and handle missing values
    numeric_cols = df.select_dtypes(include=['object']).columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='ignore')
    
    # Fill remaining NaN values with median
    df = df.apply(lambda col: col.fillna(col.median()) if col.name not in string_cols else col)
    
    return df

def calculate_effect_size(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size for two groups.
    
    Args:
        group1 (np.ndarray): First group's data
        group2 (np.ndarray): Second group's data
        
    Returns:
        float: Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_se

def perform_t_test(group1: np.ndarray, group2: np.ndarray) -> Dict[str, Any]:
    """
    Perform independent samples t-test with effect size.
    
    Args:
        group1 (np.ndarray): First group's data
        group2 (np.ndarray): Second group's data
        
    Returns:
        Dict containing t-statistic, p-value, and effect size
    """
    t_stat, p_value = stats.ttest_ind(group1, group2)
    effect_size = calculate_effect_size(group1, group2)
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'effect_size': effect_size
    }

def perform_anova(groups: List[np.ndarray]) -> Dict[str, Any]:
    """
    Perform one-way ANOVA with effect size (eta-squared).
    
    Args:
        groups (List[np.ndarray]): List of arrays containing group data
        
    Returns:
        Dict containing F-statistic, p-value, and eta-squared
    """
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Calculate eta-squared
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = np.sum((all_data - grand_mean)**2)
    eta_squared = ss_between / ss_total
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'eta_squared': eta_squared
    }

def create_correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Heatmap") -> None:
    """
    Create and display a correlation heatmap for numeric columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        title (str): Title for the heatmap
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def perform_chi_square_test(contingency_table: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform chi-square test of independence with Cramer's V.
    
    Args:
        contingency_table (pd.DataFrame): Contingency table
        
    Returns:
        Dict containing chi-square statistic, p-value, and Cramer's V
    """
    chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
    
    # Calculate Cramer's V
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramer_v = np.sqrt(chi2 / (n * min_dim))
    
    return {
        'chi2_statistic': chi2,
        'p_value': p_value,
        'cramer_v': cramer_v
    }

def create_group_comparison_plot(df: pd.DataFrame, 
                               group_col: str, 
                               value_col: str, 
                               title: str = None) -> None:
    """
    Create a box plot comparing groups.
    
    Args:
        df (pd.DataFrame): Input dataframe
        group_col (str): Column name for grouping
        value_col (str): Column name for values
        title (str): Title for the plot
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=group_col, y=value_col, data=df)
    plt.title(title or f"{value_col} by {group_col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show() 