import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import lsmr

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
    # print(f"Norm of F after solving with SVD: {np.linalg.norm(F)}")

    # Enforce that rank(F)=2
    # This requires us to take the SVD of F, throw out all but two of the
    # singular values, and reconstruct F.
    U, S, VH = np.linalg.svd(F)
    # We'll discard the smallest singular value, which is the last entry of S.
    S[2] = 0
    # SVD leads to decomposition F = (U * S) @ VH, so we'll rebuild it this way
    F = np.matmul(U * S, VH)
    # print(f"Norm of F after enforcing rank 2: {np.linalg.norm(F)}")

    if normalize:
        # t_2^t * F * t_1
        F = np.matmul(np.matmul(t2.transpose(), F), t1)
        # print(f"Norm of F after denormalization: {np.linalg.norm(F)}")

    return F


def ransac(x1, x2, threshold, num_steps=1000, random_seed=42):
    if random_seed is not None:
        np.random.seed(random_seed)  # we are using a random seed to make the results reproducible

    cur_best_indices = np.array([]) # Indices of points to use to create model by means of 8-point-algorithm
    cur_best_inliers = np.array([]) # Actual inliers
    cur_max_inlier_count = 0        # Number of inliers in selection

    # Number of total points in input
    n = x1.shape[1]

    for _ in range(num_steps):
        # Select 8 distinct (replace = False) indices of points, on which we'll
        # run the eight-point algorithm.
        # We'll use the bare minimum (ie 8) required to have the algorithm
        # work, to minimize chances of including an outlier.
        random_point_indices = np.random.choice(n, 8, replace=False)

        # run eight point algorithm on these points to get model
        model = eight_points_algorithm(
                x1[:, random_point_indices],
                x2[:, random_point_indices]
        )

        # Estimate accuracy of model by calculating, for all points, an error value.
        # We do this by taking the SSD of where the points actually are,
        # compared with where the model predicts them to be.
        distances = np.square(np.sum(x2 * np.matmul(model, x1), axis=0))

        # And get indices of those which are below the input threshold
        inlier_candidates = distances < threshold
        # `inlier_candidates` is a 2D array with boolean values, indicating for
        # which `distances < threshold` held. We can sum these, which has the
        # effect of counting the number of `True` values - ie the number of
        # selected inliers.
        candidate_count = inlier_candidates.sum()

        if candidate_count > cur_max_inlier_count:
            # Seems we found a selection of points leading to a better model
            cur_best_indices = random_point_indices
            cur_best_inliers = inlier_candidates
            cur_max_inlier_count = candidate_count

    while True:
        # Iteratively apply eight-point algorithm to get best fundamental matrix
        fundamental_matrix = eight_points_algorithm(
                x1[:, cur_best_inliers],
                x2[:, cur_best_inliers]
        )
        distances = np.square(np.sum(x2 * np.matmul(fundamental_matrix, x1), axis=0))

        inlier_candidates = distances < threshold
        if (inlier_candidates == cur_best_inliers).all():
            # Seems we found the optimal fundamental matrix for the given
            # selection of inliers.
            break

        cur_best_inliers = inlier_candidates

    return fundamental_matrix, cur_best_inliers

def decompose_essential_matrix(E, x1, x2):
    """
    Decomposes E into a rotation and translation matrix using the
    normalized corresponding points x1 and x2.
    """

    # Fix left camera-matrix
    Rl = np.eye(3)
    tl = np.array([[0, 0, 0]]).T
    Pl = np.concatenate((Rl, tl), axis=1)

    # Decomposition of E into the two possible rotation and translation
    # matrices.
    # Partially as per Hartley and Zisserman's book, partially as per some
    # notes found online at
    # https://www.eecis.udel.edu/~cer/arv/readings/old_mkss.pdf

    pi_half_rotation_matrix = np.array(
            [
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1],
            ]
    )

    # Get SVD of essential matrix
    # By the defiition of SVD, the returned v is, strictly speaking, the
    # transpose of v. But hey. :)
    u, s, v = np.linalg.svd(E)

    # SVD may yield matrices with a negative determinant, which is incompatible
    # with the definition of a proper rotational matrix. If this happens, we'll
    # just multiply the whole thing with -1.
    if np.linalg.det(u) < 0:
        u = -u
    if np.linalg.det(v) < 0:
        v = -v

    R1 = u @ pi_half_rotation_matrix @ v
    R2 = u @ pi_half_rotation_matrix.T @ v

    # t1, t2 are +- third column of u, reshaped into appropriate (column) form.
    t1 = u[:, 2].reshape(3, 1)
    t2 = -u[:, 2].reshape(3, 1)

    # Four possibilities
    Pr = [np.concatenate((R1, t1), axis=1),
          np.concatenate((R1, t2), axis=1),
          np.concatenate((R2, t1), axis=1),
          np.concatenate((R2, t2), axis=1)]

    # Compute reconstructions for all possible right camera-matrices
    X3Ds = [infer_3d(x1[:, 0:1], x2[:, 0:1], Pl, x) for x in Pr]

    # Compute projections on image-planes and find when both cameras see point
    test = [np.prod(np.hstack((Pl @ np.vstack((X3Ds[i], [[1]])), Pr[i] @ np.vstack((X3Ds[i], [[1]])))) > 0, 1) for i in
            range(4)]
    test = np.array(test)
    idx = np.where(np.hstack((test[0, 2], test[1, 2], test[2, 2], test[3, 2])) > 0.)[0][0]

    # Choose correct matrix
    Pr = Pr[idx]

    return Pl, Pr


def infer_3d(x1, x2, Pl, Pr):
    # INFER3D Infers 3d-positions of the point-correspondences x1 and x2, using
    # the rotation matrices Rl, Rr and translation vectors tl, tr. Using a
    # least-squares approach.

    M = x1.shape[1]
    # Extract rotation and translation
    Rl = Pl[:3, :3]
    tl = Pl[:3, 3]
    Rr = Pr[:3, :3]
    tr = Pr[:3, 3]

    # Construct matrix A with constraints on 3d points
    row_idx = np.tile(np.arange(4 * M), (3, 1)).T.reshape(-1)
    col_idx = np.tile(np.arange(3 * M), (1, 4)).reshape(-1)

    A = np.zeros((4 * M, 3))
    A[:M, :3] = x1[0:1, :].T @ Rl[2:3, :] - np.tile(Rl[0:1, :], (M, 1))
    A[M:2 * M, :3] = x1[1:2, :].T @ Rl[2:3, :] - np.tile(Rl[1:2, :], (M, 1))
    A[2 * M:3 * M, :3] = x2[0:1, :].T @ Rr[2:3, :] - np.tile(Rr[0:1, :], (M, 1))
    A[3 * M:4 * M, :3] = x2[1:2, :].T @ Rr[2:3, :] - np.tile(Rr[1:2, :], (M, 1))

    A = sparse.csr_matrix((A.reshape(-1), (row_idx, col_idx)), shape=(4 * M, 3 * M))

    # Construct vector b
    b = np.zeros((4 * M, 1))
    b[:M] = np.tile(tl[0], (M, 1)) - x1[0:1, :].T * tl[2]
    b[M:2 * M] = np.tile(tl[1], (M, 1)) - x1[1:2, :].T * tl[2]
    b[2 * M:3 * M] = np.tile(tr[0], (M, 1)) - x2[0:1, :].T * tr[2]
    b[3 * M:4 * M] = np.tile(tr[1], (M, 1)) - x2[1:2, :].T * tr[2]

    # Solve for 3d-positions in a least-squares way
    w = lsmr(A, b)[0]
    x3d = w.reshape(M, 3).T

    return x3d
