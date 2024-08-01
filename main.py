import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Tuple
from bayes_opt import BayesianOptimization

def score_transformation(odds: np.ndarray, pdo: float, centroid_odds: float, centroid: float, 
                         min_score: int = 300, max_score: int = 850) -> np.ndarray:
    """
    Transform odds to scores based on given parameters.
    
    :param odds: Array of odds to be transformed
    :param pdo: Points to Double the Odds
    :param centroid_odds: Odds at the centroid score
    :param centroid: Centroid score
    :param min_score: Minimum score (default 300)
    :param max_score: Maximum score (default 850)
    :return: Array of transformed scores
    """
    factor = pdo / np.log(2)
    offset = centroid - (factor * np.log(centroid_odds))
    scores = offset + (factor * np.log(odds))
    return np.clip(scores, min_score, max_score)

def calculate_chi_squared_distance(actual: np.ndarray, expected: np.ndarray) -> float:
    """
    Calculate chi-squared distance between actual and expected distributions.
    
    :param actual: Actual distribution
    :param expected: Expected distribution
    :return: Chi-squared distance
    """
    return np.sum((actual - expected) ** 2 / (actual + expected))

def optimize_transformation(df: pd.DataFrame, target_distribution: List[float], 
                            score_bins: List[int], n_iterations: int = 100) -> dict:
    """
    Optimize score transformation parameters using Bayesian Optimization.
    
    :param df: DataFrame containing 'probability' column
    :param target_distribution: Target distribution percentages
    :param score_bins: Bins for score distribution
    :param n_iterations: Number of optimization iterations
    :return: Dictionary of optimal parameters
    """
    def objective(pdo: float, centroid_odds: float, centroid: float) -> float:
        odds = df['probability'] / (1 - df['probability'])
        scores = score_transformation(odds, pdo, centroid_odds, centroid)
        hist, _ = np.histogram(scores, bins=score_bins, density=True)
        actual_dist = hist / hist.sum() * 100
        return -calculate_chi_squared_distance(actual_dist, target_distribution)

    pbounds = {'pdo': (1, 100), 'centroid_odds': (1, 100), 'centroid': (400, 800)}
    optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=42)
    optimizer.maximize(init_points=10, n_iter=n_iterations)
    
    return optimizer.max['params']

def apply_transformation(df: pd.DataFrame, pdo: float, centroid_odds: float, centroid: float, 
                         prob_column: str = 'probability', 
                         score_column: str = 'score') -> pd.DataFrame:
    """
    Apply score transformation to a DataFrame.
    
    :param df: Input DataFrame
    :param pdo: Points to Double the Odds
    :param centroid_odds: Odds at the centroid score
    :param centroid: Centroid score
    :param prob_column: Name of probability column
    :param score_column: Name of new score column
    :return: DataFrame with new score column
    """
    odds = df[prob_column] / (1 - df[prob_column])
    df[score_column] = score_transformation(odds, pdo, centroid_odds, centroid)
    return df

# Example usage:
# Assuming 'df' is your DataFrame with a 'probability' column

target_dist = [1, 0.5, 1.5, 3.5, 8, 15.5, 26.5, 25.5, 10, 4.5, 2, 1.5]
score_bins = [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900]

optimal_params = optimize_transformation(df, target_dist, score_bins)
transformed_df = apply_transformation(df, **optimal_params)
