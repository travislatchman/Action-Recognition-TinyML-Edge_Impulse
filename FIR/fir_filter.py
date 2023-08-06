import numpy as np


def fir_filter(data, coeffs):
    """
    Applies a Finite Impulse Response (FIR) filter to the input data.

    Args:
        data (list or ndarray): The input data to be filtered.
        coeffs (list or ndarray): The filter coefficients.

    Returns:
        ndarray: The filtered data.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> coeffs = [0.2, 0.2, 0.2, 0.2, 0.2]
        >>> fir_filter(data, coeffs)
        array([1., 2., 3., 4., 5.])
    """
    filtered_data = np.convolve(data, coeffs, mode='same')
    return filtered_data
