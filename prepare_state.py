from typing import List
import numpy as np

def prepare_state(amplitudes: List) -> np.ndarray:
    """
    Prepares a normalized N-qubits quantum state vector.

    Parameters:
        amplitudes (list): A list of complex or real numbers.

    Returns:
        np.ndarray: Normalized state vector.
    """
    if not amplitudes:
        raise ValueError("Empty input vector.")
    
    # Convert input to a complex NumPy array
    state_vector = np.array(amplitudes, dtype=complex)
    
    # Calculate the norm
    sqr_magnitudes = np.abs(state_vector)**2
    norm = np.sqrt(np.sum(sqr_magnitudes))
    
    # Handle the zero-vector case to avoid division by zero
    if np.isclose(norm, 0.0):
        raise ValueError("Input amplitudes result in a zero-norm vector.")
        
    # Normalize and return
    normalized_vector = state_vector / norm
    
    return normalized_vector