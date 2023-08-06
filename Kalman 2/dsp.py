import numpy as np


class KalmanFilter:
    """
    Represents a basic Kalman Filter implementation.

    Attributes:
        A (ndarray): State transition matrix.
        B (ndarray): Control-input matrix.
        H (ndarray): Observation model matrix.
        Q (ndarray): Covariance of the process noise.
        R (ndarray): Covariance of the observation noise.
        P (ndarray): Covariance matrix.
        x (ndarray): Initial state estimate.
    """

    def __init__(self, A, B, H, Q, R, P, x0):
        """
        Initializes a new instance of the KalmanFilter class.

        Args:
            A (ndarray): State transition matrix.
            B (ndarray): Control-input matrix.
            H (ndarray): Observation model matrix.
            Q (ndarray): Covariance of the process noise.
            R (ndarray): Covariance of the observation noise.
            P (ndarray): Covariance matrix.
            x0 (ndarray): Initial state estimate.
        """
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x0

    def predict(self, u):
        """
        Predicts the next state based on the state transition matrix and control input.

        Args:
            u (ndarray): Control input.

        Returns:
            ndarray: Predicted state estimate.
        """
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        """
        Updates the state estimate and covariance matrix based on the observed measurement.

        Args:
            z (ndarray): Observed measurement.
        """
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.A.shape[0])
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P),
                        (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)


def apply_kalman_filter(y):
    """
    Applies a simple Kalman Filter to a one-dimensional data series.

    Args:
        y (ndarray): Input data series to be filtered.

    Returns:
        ndarray: Kalman-filtered data series.

    Notes:
        This function uses a simple Kalman Filter with 1D state representation and assumes
        that there is no control input (u is always zero). The matrices are initialized 
        with identity values for simplicity.
    """
    A = np.eye(1)
    B = np.zeros((1, 1))
    H = np.eye(1)
    Q = np.eye(1)
    R = np.eye(1)
    P = np.eye(1)
    x0 = np.zeros(1)
    kf = KalmanFilter(A, B, H, Q, R, P, x0)
    filtered_values = []
    for value in y:
        kf.predict(np.zeros(1))
        kf.update(value)
        filtered_values.append(kf.x[0])
    return np.array(filtered_values)


def generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq, scale_axes, kalman_filter=True):
    """
    Generates features from raw data, applies scaling, and optionally uses Kalman filter for noise reduction.

    Args:
        implementation_version (str): Version or descriptor of the implementation.
        draw_graphs (bool): Determines if graphs should be generated for the processed data.
        raw_data (ndarray): 1D array containing raw data points.
        axes (list): List of axes names for the data, e.g., ['x', 'y', 'z'].
        sampling_freq (float): Sampling frequency of the raw data.
        scale_axes (float): Scaling factor to be applied to raw data.
        kalman_filter (bool, optional): If True, the Kalman filter is applied to the data for noise reduction. 
                                        Defaults to True.

    Returns:
        dict: Dictionary containing the generated features, any drawn graphs, and configuration details for the output.

    Notes:
        This function reshapes the 1D raw data into a matrix with rows corresponding to different axes.
        If Kalman filtering is enabled, the data is passed through the `apply_kalman_filter` function for each axis.
        Graphs, if requested, are generated to visualize the processed data over time.
        The returned dictionary provides a structured representation of the processed data, including the features,
        any generated graphs, and configuration details for how the data should be interpreted (e.g., as 'flat', 
        'image', or 'spectrogram').
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
