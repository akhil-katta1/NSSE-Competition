import pytest
import pandas as pd
import numpy as np
from utils import (
    calculate_effect_size,
    perform_t_test,
    perform_anova,
    perform_chi_square_test
)

def test_calculate_effect_size():
    # Test with known values
    group1 = np.array([1, 2, 3, 4, 5])
    group2 = np.array([2, 3, 4, 5, 6])
    effect_size = calculate_effect_size(group1, group2)
    assert isinstance(effect_size, float)
    assert not np.isnan(effect_size)

def test_perform_t_test():
    # Test with known values
    group1 = np.array([1, 2, 3, 4, 5])
    group2 = np.array([2, 3, 4, 5, 6])
    result = perform_t_test(group1, group2)
    assert isinstance(result, dict)
    assert 't_statistic' in result
    assert 'p_value' in result
    assert 'effect_size' in result
    assert not np.isnan(result['t_statistic'])
    assert not np.isnan(result['p_value'])
    assert not np.isnan(result['effect_size'])

def test_perform_anova():
    # Test with known values
    groups = [
        np.array([1, 2, 3]),
        np.array([2, 3, 4]),
        np.array([3, 4, 5])
    ]
    result = perform_anova(groups)
    assert isinstance(result, dict)
    assert 'f_statistic' in result
    assert 'p_value' in result
    assert 'eta_squared' in result
    assert not np.isnan(result['f_statistic'])
    assert not np.isnan(result['p_value'])
    assert not np.isnan(result['eta_squared'])
    assert 0 <= result['eta_squared'] <= 1

def test_perform_chi_square_test():
    # Test with known values
    contingency_table = pd.DataFrame({
        'A': [10, 20],
        'B': [15, 25]
    })
    result = perform_chi_square_test(contingency_table)
    assert isinstance(result, dict)
    assert 'chi2_statistic' in result
    assert 'p_value' in result
    assert 'cramer_v' in result
    assert not np.isnan(result['chi2_statistic'])
    assert not np.isnan(result['p_value'])
    assert not np.isnan(result['cramer_v'])
    assert 0 <= result['cramer_v'] <= 1

def test_edge_cases():
    # Test with empty arrays
    with pytest.raises(ValueError):
        calculate_effect_size(np.array([]), np.array([1, 2, 3]))
    
    # Test with single value arrays
    effect_size = calculate_effect_size(np.array([1]), np.array([2]))
    assert not np.isnan(effect_size)
    
    # Test with identical arrays
    effect_size = calculate_effect_size(np.array([1, 2, 3]), np.array([1, 2, 3]))
    assert effect_size == 0 