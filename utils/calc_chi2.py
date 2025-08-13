import numpy as np





def calc_chi2(points):
    """
    Fit a 3D line to a set of points using least squares method and calculate chi^2.

    Parameters:
    points (numpy.ndarray): A Nx3 array of points in 3D space.

    Returns:
    dict: A dictionary containing the line direction vector, a point on the line,
          and chi^2 value.
    """
    # Compute the centroid of the points
    centroid = np.mean(points, axis=0)

    # Center the points around the centroid
    centered_points = points - centroid

    # Perform singular value decomposition (SVD)
    _, _, vh = np.linalg.svd(centered_points)

    # The line direction is given by the first right singular vector
    direction = vh[0]

    # Compute chi^2 as the sum of squared distances of points to the line
    distances = np.linalg.norm(np.cross(centered_points, direction), axis=1)
    chi2 = np.sum(distances**2)

    return {
        "direction": direction,
        "point": centroid,
        "chi2": chi2
    }