import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg


def get_normalization_matrix(x):
    """
    get_normalization_matrix Returns the transformation matrix used to normalize
    the inputs x
    Normalization corresponds to subtracting mean-position and positions
    have a mean distance of sqrt(2) to the center
    """
    # Input: x 3*N
    # 
    # Output: T 3x3 transformation matrix of points

    # Average (x, y), 1) coordinates of the point values 
    # We'll also reshape it so that we can use it for the following
    # subtraction.
    center_point = np.mean(x, axis=1).reshape((3, 1))

    # Get the average mean distance to the center point.
    # We'll interpret 'distance' to mean the euclidean norm
    distances = np.linalg.norm(x - center_point, axis=0) # 1xn matrix of distance *per point*
    avg_distance = np.mean(distances) # scalar of mean distance

    # Recall the structure of an affine transformation matrix:
    # | a b x |
    # | c d y |
    # | 0 0 1 |
    # Where ((x), (y)) encodes the translation, ((a, b), (c, d))$ a rotation,
    # and $a, d$ a scaling operation.
    # matrix.

    # As we want the average sum of distances to be equal to sqrt(2), we scale
    # appropriately. We use the same scaling factor for both directions as we
    # do not want to distort the image.
    # We further translate by -center (scaled with the scale factor) to have
    # coordinates centered around (0, 0).
    scale_factor = np.sqrt(2) / avg_distance
    T = np.array(
            [
                [scale_factor, 0,            -center_point[0, 0] * scale_factor],
                [0,            scale_factor, -center_point[1, 0] * scale_factor],
                [0,            0,            1]
            ]
    )

    return T


def eight_points_algorithm(x1, x2, normalize=True):
    """
    Calculates the fundamental matrix between two views using the normalized 8 point algorithm
    Inputs:
                    x1      3xN     homogeneous coordinates of matched points in view 1
                    x2      3xN     homogeneous coordinates of matched points in view 2
    Outputs:
                    F       3x3     fundamental matrix
    """
    N = x1.shape[1]

    if normalize:
        # Construct transformation matrices to normalize the coordinates
        t1 = get_normalization_matrix(x1)
        t2 = get_normalization_matrix(x2)

        # Normalize inputs
        x1 = np.matmul(t1, x1)
        x2 = np.matmul(t2, x2)

    # 1 row per point, 9 columns as per the paper
    A = np.zeros((x1.shape[1], 9))
    # Recall that the fundamental matrix, when defined as:
    # (x', y') * F * (x, y) = 0
    # Will be the transformation from the coordinate system of (x', y') to (x,
    # y).
    # For our case, let (x, y) := x1, (x', y') := x2
    # The fundamental matrix for the other direction follows as the transpose.
    # Column 1 = x' * x
    A[:, 0] = x1[0] * x2[0]
    # Column 2 = x' * y
    A[:, 1] = x1[1] * x2[0]
    # Column 3 = x'
    A[:, 2] = x2[0]
    # Column 4 = y' * x
    A[:, 3] = x1[0] * x2[1]
    # Column 5 = y' * y
    A[:, 4] = x1[1] * x2[1]
    # Column 6 = y'
    A[:, 5] = x2[1]
    # Column 7 = x
    A[:, 6] = x1[0]
    # Column 8 = y
    A[:, 7] = x1[1]
    # Column 9 = 1
    A[:, 8] = 1

    # Solve for f using SVD
    # No clue how SVD helps us solve this equation system, this is just stolen
    # from some slides I found.
    U, S, VH = np.linalg.svd(A)
    F = VH.transpose()[:, 8].reshape(3, 3)
    print(f"Norm of F after solving with SVD: {np.linalg.norm(F)}")

    # Enforce that rank(F)=2
    # This requires us to take the SVD of F, throw out all but two of the
    # singular values, and reconstruct F.
    U, S, VH = np.linalg.svd(F)
    # We'll discard the smallest singular value, which is the last entry of S.
    S[2] = 0
    # SVD leads to decomposition F = (U * S) @ VH, so we'll rebuild it this way
    F = np.matmul(U * S, VH)
    print(f"Norm of F after enforcing rank 2: {np.linalg.norm(F)}")

    if normalize:
        # t_2^t * F * t_1
        F = np.matmul(np.matmul(t2.transpose(), F), t1)
        print(f"Norm of F after denormalization: {np.linalg.norm(F)}")

    return F


def right_epipole(F):
    """
    Computes the (right) epipole from a fundamental matrix F.
    (Use with F.T for left epipole.)
    """

    # The epipole is the null space of F (F * e = 0)

    # Get basis of null space of F
    e = scipy.linalg.null_space(F)
    # Normalize it so it's proper homogenous coordinates
    e = e / e[2]

    return e


def plot_epipolar_line(im, F, x, e, plot):
    """
    Plot the epipole and epipolar line F*x=0 in an image. F is the fundamental matrix
    and x a point in the other image.
    """
    m, n = im.shape[:2]

    # Recall:
    # - The baseline is the line connectin the two camera origins.
    # - An epipole is the point where the baseline intersects the image plane.
    # - An epipolar line is the intersection of the image plane with the
    #   epipolar plane.

    # Epipolar line corresponding to a point x is Fx
    epipolar_line = np.dot(F, x)

    # Recall that a line in a (projected) plane can be defined as:
    # s*x + t*y + u*z = 0
    # These values for (s, t, u) are what F * x gave us
    s, t, u = epipolar_line

    # Get two x values close to the edges of the image. Maximizing the distance
    # between will lower the impact any floating-point errors have on the line.
    x_coordinates = np.array([10, n - 10])

    # Solving the above for y gives us:
    # y = -(sx + u*z) / t
    # And as we're on a projected plane, z = 0
    y_coordinates = np.array([-(s * x + u) / t for x in x_coordinates])

    plot.plot(x_coordinates, y_coordinates)
