import numpy as np
from pykalman import KalmanFilter


def apply_kalman_filter(y):
    """
    Apply Kalman filtering on a given 1D data series.
    
    The function uses the KalmanFilter class, assuming an initial state mean of 0 and
    one-dimensional observation. It then filters the provided data and returns the filtered values.

    Args:
        y (ndarray or list): Input data series to be filtered.

    Returns:
        ndarray: Filtered values after applying the Kalman filter.

    Example:
        >>> data_series = np.array([1, 2, 3, 2, 1])
        >>> filtered_data = apply_kalman_filter(data_series)
        >>> print(filtered_data)
        [0.9, 1.8, 2.7, 2.6, 2.5]  # This is a mock output; actual values may vary.

    Note:
        The function is designed for one-dimensional data. Make sure to pass 1D arrays or lists.
    """
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    filtered_values, _ = kf.filter(y)
    return filtered_values.squeeze()


def generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq, scale_axes, kalman_filter=True):
    """
    Generates features by processing the provided raw data, and optionally applies Kalman filtering.

    This function reshapes the raw data to separate out data from all specified axes, scales the values
    according to `scale_axes`, and optionally applies Kalman filtering on each axis. If `draw_graphs` is
    set to True, the function prepares graphs for the processed data.

    Args:
        implementation_version (str): Version information for the implementation.
        draw_graphs (bool): Whether to prepare and return graph data for the processed features.
        raw_data (ndarray): Input raw data to process.
        axes (list): List of axis names to separate the raw data.
        sampling_freq (float): Sampling frequency of the data.
        scale_axes (float): Scaling factor for the data values.
        kalman_filter (bool, optional): Whether to apply Kalman filtering on the data. Defaults to True.

    Returns:
        dict: Dictionary containing features, graphs (if `draw_graphs` is True), and output configuration.
              The output configuration specifies the type ('flat' in this case) and the shape of the features.

    Example:
        >>> raw_data_sample = np.array([1, 2, 3, 4, 5])
        >>> axes_sample = ['x']
        >>> result = generate_features("v1.0", True, raw_data_sample, axes_sample, 10.0, 2.0)
        >>> print(result['features'])
        [2.0, 4.0, 6.0, 8.0, 10.0]  # Mock output; actual values might vary.

    Note:
        Ensure that `raw_data` is compatible with the provided axes and the reshape operation.
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

        # first scale the values
        fx = fx * scale_axes

        # if filtering is enabled, apply the Kalman filter
        if (kalman_filter):
            fx = apply_kalman_filter(fx)

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
