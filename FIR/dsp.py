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


def generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq, fir=[0.2, 0.2, 0.2, 0.2, 0.2]):
    """
    Generate features from raw IMU data and optionally visualize filtered data.
    
    The function applies FIR filtering to the raw data and can also produce graphs of the
    filtered data if required.

    Args:
        implementation_version (int): Version number of the implementation.
        draw_graphs (bool): If True, generates and returns graphs of the filtered data.
        raw_data (ndarray): Input raw data matrix to be processed.
        axes (list): List of axis names or identifiers (e.g., ["x", "y", "z"]).
        sampling_freq (float): The sampling frequency of the raw data.
        fir (list, optional): Coefficients for the FIR filter. Defaults to [0.2, 0.2, 0.2, 0.2, 0.2].

    Returns:
        dict: A dictionary containing:
            - features (list): The filtered and processed features.
            - graphs (list): List of dictionaries with graph data if `draw_graphs` is True.
            - output_config (dict): Configuration for the output feature shape.

    Example:
        >>> raw_data = np.array([1, 2, 3, 4, 5])
        >>> axes = ["x"]
        >>> sampling_freq = 50.0
        >>> features = generate_features(1, True, raw_data, axes, sampling_freq)
        >>> features['features']
        [1., 2., 3., 4., 5.]
    """

    # features is a 1D array, reshape so we have a matrix with one raw per axis
    raw_data = raw_data.reshape(int(len(raw_data) / len(axes)), len(axes))

    features = []
    filtered_graph = {}

    # split out the data from all axes
    for ax in range(0, len(axes)):
        X = []
        for ix in range(0, raw_data.shape[0]):
            X.append(raw_data[ix][ax])

        # X now contains only the current axis
        fx = np.array(X)

        # Apply the FIR filter to the raw data
        fx = fir_filter(fx, fir)

        # we save bandwidth by only drawing graphs when needed
        if (draw_graphs):
            filtered_graph[axes[ax]] = list(fx)

        # we need to return a 1D array again, so flatten here again
        for f in fx:
            features.append(f)

    # draw the graph with time in the window on the Y axis, and the values on the X axes
    # note that the 'suggestedYMin/suggestedYMax' names are incorrect, they describe
    # the min/max of the X axis
    graphs = []
    if (draw_graphs):
        graphs.append({
            'name': 'Filtered',
            'X': filtered_graph,
            'y': np.linspace(0.0, raw_data.shape[0] * (1 / sampling_freq) * 1000, raw_data.shape[0] + 1).tolist(),
            'suggestedYMin': -20,
            'suggestedYMax': 20
        })
    
    
    return {'features': features, 'graphs': graphs, 'output_config': {
        # type can be 'flat', 'image' or 'spectrogram'
            'type': 'flat',
            'shape': {
                # shape should be { width, height, channels } for image, { width, height } for spectrogram
                'width': len(features)
            }
            }}
