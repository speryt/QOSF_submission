import numpy as np
import pytest

from prepare_state import prepare_state

def test_normalization_enforced():
    """Tests that a non-normalized vector is correctly normalized."""
    amplitudes = [1, 1, 1, 1]
    normalized_state = prepare_state(amplitudes)

    # Use existing NumPy function to compare results
    norm = np.linalg.norm(normalized_state)
    
    # Use np.isclose() for floating point comparison
    assert np.isclose(norm, 1.0)

def test_output_dimension():
    """Tests that the output vector has the correct dimension."""
    amplitudes = [1, 0, 0, 0]
    state = prepare_state(amplitudes)
    assert len(state) == 4

def test_already_normalized_state():
    """Tests output of already normalized input."""
    amplitudes = [1/np.sqrt(2), 0, 1/np.sqrt(2), 0]
    state = prepare_state(amplitudes)
    
    expected = np.array([1/np.sqrt(2), 0, 1/np.sqrt(2), 0])

    # Use np.allclose() for single value return floating point arrays comparison
    assert np.allclose(state, expected)

def test_normalization_enforced_3_qubit():
    """Tests that a non-normalized vector is correctly normalized."""
    amplitudes_3_qubit = [1, 0, 0, 1, 0, 0, 1, 0]
    prepared_state = prepare_state(amplitudes_3_qubit)

    # Use existing NumPy function to compare results
    norm = np.linalg.norm(prepared_state)

    # Use np.isclose() for floating point comparison
    assert np.isclose(norm, 1.0)

def test_output_dimension_3_qubit():
    """Tests that the output vector has the correct dimension."""
    amplitudes_3_qubit = [1, 0, 0, 1, 0, 0, 1, 0]
    prepared_state = prepare_state(amplitudes_3_qubit)
    assert len(prepared_state) == 8

def test_already_normalized_state_3_qubit():
    """Tests output of already normalized input."""
    amplitudes_3_qubit = [1, 0, 0, 1, 0, 0, 1, 0]
    expected = np.array([1 / np.sqrt(3), 0, 0, 1 / np.sqrt(3), 0, 0, 1 / np.sqrt(3), 0])
    prepared_state = prepare_state(amplitudes_3_qubit)

    # Use np.allclose() for single value return floating point arrays comparison
    assert np.allclose(prepared_state, expected)

def test_prepare_state_empty_list():
    """Tests that an empty amplitude list correctly raises a ValueError."""
    empty_amplitudes = []
    with pytest.raises(ValueError, match="Empty input vector."):
        prepare_state(empty_amplitudes)

def test_prepare_state_zero_norm_vector():
    """Tests that a 4-amplitude vector with norm 0 correctly raises a ValueError."""
    zero_norm_amplitudes = [0, 0, 0, 0]
    with pytest.raises(ValueError, match="Input amplitudes result in a zero-norm vector."):
        prepare_state(zero_norm_amplitudes)
