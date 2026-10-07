################################################################################

import numpy as np
import warnings
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigvalsh
from scipy.linalg import eigh
from sklearn.utils.extmath import randomized_svd
from numbers import Integral

from robust_mixed_dist.quantitative import (
    euclidean_dist_matrix, 
    euclidean_dist, 
    minkowski_dist_matrix, 
    minkowski_dist, 
    canberra_dist_matrix, 
    canberra_dist, 
    pearson_dist_matrix, 
    mahalanobis_dist_matrix,
    mahalanobis_dist,
    robust_mahalanobis_dist_matrix,
    robust_mahalanobis_dist, 
    S_robust
)
from robust_mixed_dist.binary import (
    sokal_dist_matrix, 
    sokal_dist, 
    jaccard_dist_matrix, 
    jaccard_dist
)
from robust_mixed_dist.multiclass import (
    hamming_dist_matrix, 
    hamming_dist
)

################################################################################

def get_dist_matrix_objects():
        
    return {
        'euclidean': euclidean_dist_matrix, 
        'minkowski': minkowski_dist_matrix,
        'canberra': canberra_dist_matrix,
        'pearson': pearson_dist_matrix,
        'mahalanobis': mahalanobis_dist_matrix,
        'robust_mahalanobis': robust_mahalanobis_dist_matrix,
        'sokal': sokal_dist_matrix,
        'jaccard': jaccard_dist_matrix,
        'hamming': hamming_dist_matrix
    }

################################################################################

def get_dist_objects():

    return {
        'euclidean': euclidean_dist, 
        'minkowski': minkowski_dist,
        'canberra': canberra_dist,
        'mahalanobis': mahalanobis_dist,
        'robust_mahalanobis': robust_mahalanobis_dist,
        'sokal': sokal_dist,
        'jaccard': jaccard_dist,
        'hamming': hamming_dist        
    }

################################################################################

def simple_gower_dist(xi, xr, rng, p1, p2, p3) :
    """
    Compute method.
    
    Parameters:
        xi, xr: a pair of mixed data vectors. They represent a couple of statistical observations.
        X: a pandas/polars data-frame or a numpy array. It represents a data matrix.
        p1, p2, p3: number of quantitative, binary and multi-class variables in the considered data matrix, respectively. Must be a non negative integer.

    Returns:
        dist: the Simple Gower distance between the observations `xi` and `xr`.
    """    

    #if hasattr(X, "to_numpy"):
    #    X = X.to_numpy()
    xi = ensure_flat_array(xi)
    xr = ensure_flat_array(xr)

    dist_objects = get_dist_objects()

    xi_quant = xi[0:p1] ; xr_quant = xr[0:p1] ; 
    xi_bin = xi[(p1):(p1+p2)] ; xr_bin = xr[(p1):(p1+p2)]
    xi_multi = xi[(p1+p2):(p1+p2+p3)] ; xr_multi = xr[(p1+p2):(p1+p2+p3)]

    rng[rng == 0] = 1  # evitar división por cero
    
    dist1 = np.sum(np.abs(xi_quant - xr_quant)/rng) if p1 > 0 else 0
    dist2 = dist_objects['jaccard'](xi_bin, xr_bin) if p2 > 0 else 0
    dist3 = dist_objects['hamming'](xi_multi, xr_multi) if p3 > 0 else 0
    dist = dist1 + dist2 + dist3

    return dist

################################################################################

def simple_gower_dist_matrix(X, p1, p2, p3):
    """
    Cálculo matricial de la distancia simple de Gower entre todas las filas de X.

    Parameters:
        X: np.ndarray o DataFrame (se convierte a np.ndarray).
        p1: número de columnas numéricas.
        p2: número de columnas binarias.
        p3: número de columnas categóricas (multi-clase).

    Returns:
        D: matriz de distancias (n x n) con la distancia de Gower simple entre observaciones.
    """

    if hasattr(X, "to_numpy"):
        X = X.to_numpy()

    dist_matrix_objects = get_dist_matrix_objects()

    # Separar bloques
    X_quant = X[:, 0:p1] if p1 > 0 else None
    X_bin = X[:, p1:p1 + p2] if p2 > 0 else None
    X_multi = X[:, p1 + p2:p1 + p2 + p3] if p3 > 0 else None

    n = X.shape[0]
    D = np.zeros((n, n))

    # Distancia cuantitativa: Manhattan normalizada por rango
    if p1 > 0:
        rng = np.max(X_quant, axis=0) - np.min(X_quant, axis=0)
        rng[rng == 0] = 1  # evitar división por cero
        X_quant_norm = X_quant / rng
        dist_quant = dist_matrix_objects['minkowski'](X_quant_norm, q=1)
        D += dist_quant

    # Distancia binaria: Jaccard
    if p2 > 0:
        dist_bin = dist_matrix_objects['jaccard'](X_bin)
        D += dist_bin

    # Distancia categórica: Hamming (simple coincidencia)
    if p3 > 0:
        dist_multi = dist_matrix_objects['hamming'](X_multi)
        D += dist_multi

    return D

################################################################################

def geometric_variability(D_2, weights=None):
    """
    Calculates the geometric variability of the squared distance matrix passed as input.

    Parameters
    ----------
    D_2 : np.ndarray
        A square matrix (n x n) representing squared distances.
    weights : np.ndarray, optional
        A 1D array of weights. If None, uniform weights are assumed.

    Returns
    -------
    float
        The geometric variability value.
    
    Raises
    ------
    ValueError
        If the input matrix is not square or weight dimensions mismatch.
    """
    if D_2.ndim != 2 or D_2.shape[0] != D_2.shape[1]:
        raise ValueError("D_2 must be an squared matrix of 2 dimension.")
    
    n = D_2.shape[0]
    if weights is None:
        GV = np.sum(D_2) / (2 * (n**2))
        return GV

    if weights.shape[0] != n:
            raise ValueError(f"Weights dimension ({weights.shape[0]}) does not match with D_2 dimension ({n}).")
    
    # Normalize weights if needed: ensure sum is 1.0 for the standard formula
    sum_weights = np.sum(weights)
    if not np.isclose(sum_weights, 1.0):
        if np.isclose(sum_weights, 0.0):
             raise ValueError("Sum of weights cannot be zero.")
        weights = weights / sum_weights

    GV_w = (weights @ D_2 @ weights) / 2
    return GV_w

################################################################################

VALID_D1 = ['euclidean', 'minkowski', 'pearson', 'canberra', 'mahalanobis', 'robust_mahalanobis']

def _normalize_p1(p1):
    """
    Normalises p1 into a list of block sizes.
    An integer is treated as a single quantitative block.
    """
    if isinstance(p1, Integral):
        p1_list = [int(p1)]
    elif isinstance(p1, (list, tuple, np.ndarray)):
        p1_list = [int(s) for s in p1]
    else:
        raise TypeError(f"p1 must be an int or a list/tuple of ints, got {type(p1)}.")

    if len(p1_list) == 0:
        raise ValueError("p1 cannot be an empty list.")
    if any(s < 0 for s in p1_list):
        raise ValueError(f"All quantitative block sizes must be >= 0, got {p1_list}.")

    return p1_list


def _get_quantitative_blocks(X, p1, p2, p3):
    """
    Parses p1 (int or list) and returns a list of sub-arrays for each
    quantitative block, plus the binary and multi-class sub-arrays.
    Works both for data matrices (n, p) and for single observations (p,).

    Returns:
        quant_blocks : list of k arrays, one per quantitative block.
        X_bin        : binary part, or None if p2 == 0.
        X_multi      : multi-class part, or None if p3 == 0.
    """
    X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    p1_blocks = _normalize_p1(p1)

    n_cols_required = sum(p1_blocks) + p2 + p3
    if n_cols_required > X_arr.shape[-1]:
        raise ValueError(
            f"p1, p2, p3 require {n_cols_required} columns "
            f"(p1={p1_blocks}, p2={p2}, p3={p3}), but X has {X_arr.shape[-1]}."
        )

    col = 0
    quant_blocks = []
    for size in p1_blocks:
        quant_blocks.append(X_arr[..., col:col + size])
        col += size

    X_bin   = X_arr[..., col:col + p2]           if p2 > 0 else None
    X_multi = X_arr[..., col + p2:col + p2 + p3] if p3 > 0 else None

    return quant_blocks, X_bin, X_multi


def _normalize_d1_per_block(d1, k):
    """
    Normalises d1 into a list of length k (one distance per quantitative block).

    Parameters:
        d1 : str or list/tuple of str.
             - If a single string, it is replicated across all k blocks.
             - If a list/tuple, its length must equal k, and each element
               must be a valid distance name.
        k  : number of quantitative blocks.

    Returns:
        list of k strings, each one of VALID_D1.

    Raises:
        ValueError if a list/tuple is given with a length different from k,
        or if any distance name is not recognised.
    """
    if isinstance(d1, str):
        d1_list = [d1] * k
    elif isinstance(d1, (list, tuple)):
        if len(d1) != k:
            raise ValueError(
                f"d1 was given as a list of length {len(d1)}, but there are "
                f"{k} quantitative block(s) defined by p1. Provide either a "
                f"single string (applied to all blocks) or a list/tuple of "
                f"length {k}."
            )
        d1_list = list(d1)
    else:
        raise TypeError(
            f"d1 must be a str or a list/tuple of str, got {type(d1)}."
        )

    for i, d in enumerate(d1_list):
        if d not in VALID_D1:
            raise ValueError(
                f"Invalid distance '{d}' for quantitative block {i + 1}. "
                f"Must be one of {VALID_D1}."
            )

    return d1_list

def _normalize_per_block(param, k, name):
    """
    Normalises a per-block parameter into a list of length k.
      - list/tuple -> its length must be k.
      - anything else (scalar, None, np.ndarray) -> replicated k times.
    Note: a covariance matrix must be passed as np.ndarray (not as a nested
    list), otherwise it would be interpreted as a per-block list.
    """
    if isinstance(param, (list, tuple)):
        if len(param) != k:
            raise ValueError(
                f"{name} was given as a list of length {len(param)}, but there are "
                f"{k} quantitative block(s) defined by p1. Provide either a single "
                f"value (applied to all blocks) or a list/tuple of length {k}."
            )
        return list(param)
    return [param] * k


################################################################################

def compute_distances(xi, xr, p1, p2, p3, d1, d2, d3, q=1, S=None):
    """
    Calculates the per-block distances between two observations that are
    involved in the Generalized Gower distance.

    Parameters:
        xi, xr : 1D array-like. A pair of mixed observations.
        p1 : int or list of ints (quantitative block sizes).
        p2, p3 : number of binary and multi-class variables.
        d1 : str or list/tuple of str of length k.
        d2, d3 : distances for binary and multi-class variables.
        q  : int or list of ints of length k.
        S  : covariance matrix (np.ndarray) if k == 1, or a list of length k
             with one covariance matrix (or None) per block. Required for
             blocks whose distance is 'mahalanobis' or 'robust_mahalanobis'
             (see `compute_S_by_block`).

    Returns:
        tuple of k + 2 floats: (dist_q1, ..., dist_qk, dist_bin, dist_multi).
    """
    xi = ensure_flat_array(xi)
    xr = ensure_flat_array(xr)

    xi_q, xi_bin, xi_multi = _get_quantitative_blocks(xi, p1, p2, p3)
    xr_q, xr_bin, xr_multi = _get_quantitative_blocks(xr, p1, p2, p3)
    k = len(xi_q)
    d1_list = _normalize_d1_per_block(d1, k)
    q_list  = _normalize_per_block(q, k, 'q')
    S_list  = _normalize_per_block(S, k, 'S')

    dist_objects = get_dist_objects()
    dist_list = []

    for j, (a, b, d1_j, q_j, S_j) in enumerate(zip(xi_q, xr_q, d1_list, q_list, S_list), start=1):
        if a.shape[0] == 0:
            dist_list.append(0.0)
            continue
        if d1_j not in dist_objects:
            raise NotImplementedError(
                f"Distance '{d1_j}' (block {j}) has no point-to-point implementation."
            )
        if d1_j in ('mahalanobis', 'robust_mahalanobis') and S_j is None:
            raise ValueError(
                f"Quantitative block {j} uses '{d1_j}' and requires its covariance "
                f"matrix in S (see compute_S_by_block)."
            )

        if d1_j == 'minkowski':
            dist_list.append(dist_objects[d1_j](a, b, q=q_j))
        elif d1_j == 'robust_mahalanobis':
            dist_list.append(dist_objects[d1_j](a, b, S_robust=S_j))
        elif d1_j == 'mahalanobis':
            dist_list.append(dist_objects[d1_j](a, b, S=S_j))
        else:
            dist_list.append(dist_objects[d1_j](a, b))

    dist_list.append(dist_objects[d2](xi_bin, xr_bin)     if p2 > 0 else 0.0)
    dist_list.append(dist_objects[d3](xi_multi, xr_multi) if p3 > 0 else 0.0)

    return tuple(dist_list)

################################################################################

def compute_dist_matrices(
        X, p1, p2, p3, d1, d2, d3, q=1, 
        robust_method='trimmed', epsilon=0.05, alpha=0.05, n_iters=20, weights=None
):
    """
    Calculates the distance matrices that are involved in the Generalized Gower distance.
            
    Parameters:
      X: a pandas/polars data-frame or a numpy array. Represents a data matrix.
      p1, p2, p3: number of quantitative, binary and multi-class variables in the considered data matrix, respectively. Must be a non negative integer.
      d1: name of the distance to be computed for quantitative variables. Must be an string in ['euclidean', 'minkowski', 'canberra', 'mahalanobis', 'robust_mahalanobis']. 
      d2: name of the distance to be computed for binary variables. Must be an string in ['sokal', 'jaccard'].
      d3: name of the distance to be computed for multi-class variables. Must be an string in ['matching'].
      q: the parameter that defines the Minkowski distance. Must be a positive integer.
      robust_method: the robust_method to be used for computing the robust covariance matrix. Only needed when d1 = 'robust_mahalanobis'.
      epsilon: parameter used by the Delvin algorithm that is used when computing the robust covariance matrix. Only needed when d1 = 'robust_mahalanobis'.
      n_iter: maximum number of iterations used by the Delvin algorithm. Only needed when d1 = 'robust_mahalanobis'.
      weights: the sample weights. Only used if provided and d1 = 'robust_mahalanobis'.  
        
    Returns:
      D1, D2, D3: the distances matrices associated to the quantitative, binary and multi-class variables, respectively.
    """
 
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()

    dist_matrix_objects = get_dist_matrix_objects()

    n = len(X)
    X_quant = X[:, 0:p1] 
    X_bin = X[:, (p1):(p1+p2)]
    X_multi = X[:, (p1+p2):(p1+p2+p3)]

    # Define D1 based on d1 and p1
    D1 = np.zeros((n, n))
    if p1 > 0:
        if d1 == 'minkowski':
            D1 = dist_matrix_objects[d1](X_quant, q)
        elif d1 == 'robust_mahalanobis':
            S_robust_est = S_robust(X=X_quant, method=robust_method, alpha=alpha, epsilon=epsilon, n_iters=n_iters, weights=weights)
            D1 = dist_matrix_objects[d1](X_quant, S_robust=S_robust_est)
        else:
            D1 = dist_matrix_objects[d1](X_quant)

    # Define D2 based on p2
    D2 = dist_matrix_objects[d2](X_bin) if p2 > 0 else np.zeros((n, n)) 
    # Define D3 based on p3
    D3 = dist_matrix_objects[d3](X_multi) if p3 > 0 else np.zeros((n, n))

    return D1, D2, D3

################################################################################

def compute_dist_matrices_by_block(
        X, p1, p2, p3, d1, d2, d3,
        q=1, robust_method='trimmed', epsilon=0.05, alpha=0.05, n_iters=20, weights=None
):
    """
    Computes one distance matrix per block: k quantitative blocks (defined by p1),
    plus the binary block and the multi-class block.

    Parameters:
        X  : pandas/polars DataFrame or numpy array of shape (n, sum(p1) + p2 + p3).
        p1 : int or list of ints (block sizes).
        p2, p3 : number of binary and multi-class variables.
        d1 : str or list/tuple of str of length k (distance per quantitative block).
        d2, d3 : distances for binary and multi-class variables.
        q  : int or list of ints of length k (Minkowski parameter per block).
        robust_method, epsilon, alpha, n_iters, weights : robust covariance
            parameters, global across blocks (used when a block uses 'robust_mahalanobis').

    Returns:
        dist_list : list of k + 2 (n x n) arrays [D_q1, ..., D_qk, D_bin, D_multi].
                    Absent blocks (0 columns, p2 = 0, p3 = 0) are zero matrices.
    """
    quant_blocks, X_bin, X_multi = _get_quantitative_blocks(X, p1, p2, p3)
    k = len(quant_blocks)
    d1_list = _normalize_d1_per_block(d1, k)
    q_list  = _normalize_per_block(q, k, 'q')
    n = quant_blocks[0].shape[0]   # always exists, even with 0 columns

    dist_matrix_objects = get_dist_matrix_objects()
    dist_list = []

    for X_q, d1_i, q_i in zip(quant_blocks, d1_list, q_list):
        if X_q.shape[1] == 0:
            dist_list.append(np.zeros((n, n)))
            continue
        D_q, _, _ = compute_dist_matrices(
            X=X_q, p1=X_q.shape[1], p2=0, p3=0, d1=d1_i, d2=d2, d3=d3,
            q=q_i, robust_method=robust_method, epsilon=epsilon,
            alpha=alpha, n_iters=n_iters, weights=weights
        )
        dist_list.append(D_q)

    dist_list.append(dist_matrix_objects[d2](X_bin)   if p2 > 0 else np.zeros((n, n)))
    dist_list.append(dist_matrix_objects[d3](X_multi) if p3 > 0 else np.zeros((n, n)))

    return dist_list

################################################################################

def compute_S_by_block(
        X, p1, p2, p3, d1,
        robust_method='trimmed', alpha=0.05, epsilon=0.05, n_iters=20, weights=None
):
    """
    Computes the covariance matrix required by each quantitative block.

    Returns:
        S_list : list of length k. Element j is
                 - the robust covariance matrix if d1[j] == 'robust_mahalanobis',
                 - the classical covariance matrix if d1[j] == 'mahalanobis',
                 - None otherwise (or if the block has 0 columns).
                 Can be passed directly as `S` to `generalized_gower_dist`.
    """
    quant_blocks, _, _ = _get_quantitative_blocks(X, p1, p2, p3)
    d1_list = _normalize_d1_per_block(d1, len(quant_blocks))

    S_list = []
    for X_q, d in zip(quant_blocks, d1_list):
        if X_q.shape[1] == 0:
            S_list.append(None)
        elif d == 'robust_mahalanobis':
            S_list.append(S_robust(X=X_q, method=robust_method, alpha=alpha,
                                   epsilon=epsilon, n_iters=n_iters, weights=weights))
        elif d == 'mahalanobis':
            S_list.append(np.atleast_2d(np.cov(X_q, rowvar=False)))
        else:
            S_list.append(None)

    return S_list

################################################################################

def ensure_flat_array(x):
    """
    Converts input (DataFrame, Series, List, etc.) to a flattened 1D NumPy array.
    
    Optimized to avoid hard dependencies on Pandas/Polars and to use
    zero-copy views (.ravel()) whenever possible.
    """
    # 1. Convert to NumPy using duck typing
    # This works for Pandas, Polars, and anything with a .to_numpy() method
    if hasattr(x, "to_numpy"):
        arr = x.to_numpy()
    else:
        # Fallback for lists, tuples, or raw numpy arrays
        arr = np.asarray(x)

    # 2. Flatten only if necessary
    # If it's a DataFrame (2D), this flattens it.
    # If it's a Series (1D), it stays as is.
    if arr.ndim > 1:
        return arr.flatten()   
    
    return arr

################################################################################

def compute_distances(xi, xr, p1, p2, p3, d1, d2, d3, q=1, S=None):
    """
    Calculates the per-block distances between two observations that are
    involved in the Generalized Gower distance.

    Parameters:
        xi, xr : 1D array-like. A pair of mixed observations.
        p1 : int or list of ints (quantitative block sizes).
        p2, p3 : number of binary and multi-class variables.
        d1 : str or list/tuple of str of length k.
        d2, d3 : distances for binary and multi-class variables.
        q  : int or list of ints of length k.
        S  : covariance matrix (np.ndarray) if k == 1, or a list of length k
             with one covariance matrix (or None) per block. Required for
             blocks whose distance is 'mahalanobis' or 'robust_mahalanobis'
             (see `compute_S_by_block`).

    Returns:
        tuple of k + 2 floats: (dist_q1, ..., dist_qk, dist_bin, dist_multi).
    """
    xi = ensure_flat_array(xi)
    xr = ensure_flat_array(xr)

    xi_q, xi_bin, xi_multi = _get_quantitative_blocks(xi, p1, p2, p3)
    xr_q, xr_bin, xr_multi = _get_quantitative_blocks(xr, p1, p2, p3)
    k = len(xi_q)
    d1_list = _normalize_d1_per_block(d1, k)
    q_list  = _normalize_per_block(q, k, 'q')
    S_list  = _normalize_per_block(S, k, 'S')

    dist_objects = get_dist_objects()
    dist_list = []

    for j, (a, b, d1_j, q_j, S_j) in enumerate(zip(xi_q, xr_q, d1_list, q_list, S_list), start=1):
        if a.shape[0] == 0:
            dist_list.append(0.0)
            continue
        if d1_j not in dist_objects:
            raise NotImplementedError(
                f"Distance '{d1_j}' (block {j}) has no point-to-point implementation."
            )
        if d1_j in ('mahalanobis', 'robust_mahalanobis') and S_j is None:
            raise ValueError(
                f"Quantitative block {j} uses '{d1_j}' and requires its covariance "
                f"matrix in S (see compute_S_by_block)."
            )

        if d1_j == 'minkowski':
            dist_list.append(dist_objects[d1_j](a, b, q=q_j))
        elif d1_j == 'robust_mahalanobis':
            dist_list.append(dist_objects[d1_j](a, b, S_robust=S_j))
        elif d1_j == 'mahalanobis':
            dist_list.append(dist_objects[d1_j](a, b, S=S_j))
        else:
            dist_list.append(dist_objects[d1_j](a, b))

    dist_list.append(dist_objects[d2](xi_bin, xr_bin)     if p2 > 0 else 0.0)
    dist_list.append(dist_objects[d3](xi_multi, xr_multi) if p3 > 0 else 0.0)

    return tuple(dist_list)
    
################################################################################

def compute_geometric_var(
        X, p1, p2, p3, d1, d2, d3, 
        q=1, robust_method='trimmed', epsilon=0.05, alpha=0.05, n_iters=20, weights=None
    ): 
    """
    Calculates the geometric variability of an Generalized Gower distance matrix.

    Parameters:
        X: a pandas/polars data-frame or a numpy array. Represents a data matrix.
        p1, p2, p3: number of quantitative, binary and multi-class variables in the considered data matrix, respectively. Must be a non negative integer.
        d1: name of the distance to be computed for quantitative variables. Must be an string in ['euclidean', 'minkowski', 'canberra', 'mahalanobis', 'robust_mahalanobis']. 
        d2: name of the distance to be computed for binary variables. Must be an string in ['sokal', 'jaccard'].
        d3: name of the distance to be computed for multi-class variables. Must be an string in ['matching'].
        q: the parameter that defines the Minkowski distance. Must be a positive integer.
        robust_method: the robust_method to be used for computing the robust covariance matrix. Only needed when d1 = 'robust_mahalanobis'.
        epsilon: parameter used by the Delvin algorithm that is used when computing the robust covariance matrix. Only needed when d1 = 'robust_mahalanobis'.
        n_iter: maximum number of iterations used by the Delvin algorithm. Only needed when d1 = 'robust_mahalanobis'.
        weights: the sample weights. Only used if provided and d1 = 'robust_mahalanobis'.  
            
    Returns:
        Tuple of k + 2 geometric variabilities, in the same order as the blocks (k quantitative, binary, multi-class). For an integer p1 this is (VG1, VG2, VG3).
    """

    dist_list = compute_dist_matrices_by_block(
        X=X, p1=p1, p2=p2, p3=p3, d1=d1, d2=d2, d3=d3,
        q=q, robust_method=robust_method, epsilon=epsilon,
        alpha=alpha, n_iters=n_iters, weights=weights
    )
    return tuple(geometric_variability(D ** 2, weights) for D in dist_list)

################################################################################

def generalized_gower_dist_matrix(
        X, p1, p2, p3, d1, d2, d3, 
        q=1, robust_method='trimmed', alpha=0.05, epsilon=0.05, n_iters=20, weights=None,
        return_combined_distances = False
    ):        
    """
    Calculates the Generalized Gower matrix for a data matrix.
    
    Parameters:
        X: a pandas/polars data-frame or a numpy array. Represents a data matrix.
        p1 : int or list of ints. Number of quantitative variables (single block)
             or list [p1_1, ..., p1_k] defining k quantitative blocks.
             If a list is provided, GGower is applied across all k+2 blocks.
        p2, p3: number of binary and multi-class variables in the considered data matrix, respectively. Must be a non negative integer.
        d1 : distance for quantitative blocks. Either:
             - a single string from ['euclidean', 'minkowski', 'canberra',
               'pearson', 'mahalanobis', 'robust_mahalanobis'], applied to every
               quantitative block, or
             - a list/tuple of strings of the same length as the number of
               quantitative blocks (i.e. len(p1) if p1 is a list), giving a
               distance per block (e.g. block 1 -> 'euclidean', block 2 ->
               'mahalanobis').
        d2: name of the distance to be computed for binary variables. Must be an string in ['sokal', 'jaccard'].
        d3: name of the distance to be computed for multi-class variables. Must be an string in ['hamming'].
        q: the parameter that defines the Minkowski distance. Must be a positive integer.
        robust_method: the robust_method to be used for computing the robust covariance matrix. Only needed when d1 = 'robust_mahalanobis'.
        alpha : a real number in [0,1] that is used if `robust_method` is 'trimmed' or 'winsorized'. Only needed when d1 = 'robust_mahalanobis'.
        epsilon : parameter used by the Delvin transformation. epsilon=0.05 is recommended. Only needed when d1 = 'robust_mahalanobis'.
        n_iter : maximum number of iterations run by the Delvin algorithm. Only needed when d1 = 'robust_mahalanobis'.
        weights: the sample weights. Only used if provided and d1 = 'robust_mahalanobis'.  
    
    Returns:
        dist : (n x n) Generalized Gower distance matrix.
        dist_list : only if return_combined_distances=True. List of k + 2
                    (n x n) matrices [D_q1, ..., D_qk, D_bin, D_multi]
                    (non-standardised, non-squared).
    """
    dist_list = compute_dist_matrices_by_block(
        X=X, p1=p1, p2=p2, p3=p3, d1=d1, d2=d2, d3=d3,
        q=q, robust_method=robust_method, epsilon=epsilon,
        alpha=alpha, n_iters=n_iters, weights=weights
    )

    n = dist_list[0].shape[0]
    dist_2_std_sum = np.zeros((n, n))

    for D in dist_list:
        D_2 = D ** 2
        geom_var = geometric_variability(D_2, weights=weights)
        dist_2_std_sum += D_2 / geom_var if geom_var > 1e-10 else D_2

    dist = np.sqrt(dist_2_std_sum)

    if return_combined_distances:
        return dist, dist_list
    return dist

################################################################################

def generalized_gower_dist(
        xi, xr, p1, p2, p3, d1, d2, d3, q=1, S=None,
        geom_vars=None, geom_var_1=None, geom_var_2=None, geom_var_3=None
):
    """
    Calculates the Generalized Gower distance between a pair of mixed observations.

    Parameters:
        ... (p1 / d1 / q como en la versión matricial) ...
        S : covariance matrix (k == 1) or list of k covariance matrices / None
            (see compute_S_by_block).
        geom_vars : sequence of k + 2 geometric variabilities, in block order
                    (e.g. the output of compute_geometric_var).
        geom_var_1, geom_var_2, geom_var_3 : legacy arguments, only valid when
                    p1 defines a single quantitative block. Ignored if geom_vars
                    is given.
        A geometric variability equal to None (or <= 1e-10) means that block
        is not standardised.

    Returns:
        dist : the Generalized Gower distance between `xi` and `xr`.
    """
    dist_list = compute_distances(
        xi=xi, xr=xr, p1=p1, p2=p2, p3=p3, d1=d1, d2=d2, d3=d3, q=q, S=S
    )
    m = len(dist_list)

    if geom_vars is None:
        if m != 3:
            raise ValueError(
                f"p1 defines {m - 2} quantitative blocks: pass `geom_vars` as a "
                f"sequence of length {m} (e.g. the output of compute_geometric_var)."
            )
        geom_vars = (geom_var_1, geom_var_2, geom_var_3)
    elif len(geom_vars) != m:
        raise ValueError(f"geom_vars has length {len(geom_vars)}, expected {m}.")

    dist_2_std_sum = 0.0
    for d, gv in zip(dist_list, geom_vars):
        d_2 = d ** 2
        dist_2_std_sum += d_2 / gv if (gv is not None and gv > 1e-10) else d_2

    return np.sqrt(dist_2_std_sum)


################################################################################

def compute_gram_matrix(dist, centering_matrix):
    """
    Algebraic formula: G = -(1/2) * J * D * J
    CRITICAL PERFORMANCE BOTTLENECK:
    1. Complexity: O(n^3) due to double matrix multiplication (@).
    2. Memory: Inefficient. Requires allocating and storing the dense 
       'centering_matrix' (J) of size (n, n), which is heavy for large datasets. 
    gram_matrix = -(1/2)*(centering_matrix @ dist @ centering_matrix)
    """
    gram_matrix = -(1/2)*(centering_matrix @ dist @ centering_matrix)

    return gram_matrix

################################################################################

def compute_gram_matrix_faster(dist):
    """
    Optimized implementation using Vectorization and Broadcasting.
    Mathematically equivalent to G = -(1/2) * J * D * J, but computationally superior.
    """
    # 1. Calculate means
    # Complexity: O(n^2). This is linear with respect to the number of elements.
    dist_row_means = np.mean(dist, axis=1, keepdims=True)
    dist_col_means = np.mean(dist, axis=0, keepdims=True)
    dist_mean = np.mean(dist)
    
    # 2. Vectorized Double Centering
    # Algebraic expansion: 
    # (J * D * J)_{ij} = d_{ij} - \mu_{i.} - \mu_{.j} + \mu_{d}
    # J * D * J = D - dist_row_means - dist_col_means + dist_mean
    # BROADCASTING MECHANISM:
    # NumPy performs this operation conceptually AS IF all operands were 
    # full (n, n) matrices. It implicitly expands the dimensions.

    # OPTIMIZATION GAINS:
    # - Speed: Reduces complexity from Cubic O(n^3) to Quadratic O(n^2).
    # - Memory: This expansion is VIRTUAL. NumPy does not actually allocate 
    #           memory for the expanded matrices. It subtracts elements on-the-fly,
    #           avoiding the creation of huge intermediate matrices.
    dist_double_centered = dist - dist_row_means - dist_col_means + dist_mean

    # 3. Scaling to obtain Gram Matrix
    dist_gram_matrix = -(1/2) * dist_double_centered

    return dist_gram_matrix

################################################################################

def check_gram_matrix_psd(gram_matrix, atol=1e-10):
    """
    Checks if a Gram matrix is Positive Semi-Definite (PSD).
    Optimized for stability and speed using LAPACK symmetric routines.
    """
    try:
        # subset_by_index=[0, 0] calcula SOLO el 1er autovalor (el más pequeño)
        # Es determinista, exacto y no calcula autovectores.
        eig_min_val = eigvalsh(gram_matrix, subset_by_index=[0, 0])[0]
        
    except Exception:
        # Fallback de máxima seguridad (calcula todos los autovalores simétricos)
        # Usamos eigvalsh en lugar de np.linalg.eigvals porque es mucho más
        # rápido al asumir que la matriz es simétrica.
        eig_min_val = np.min(eigvalsh(gram_matrix))

    is_psd = eig_min_val >= -atol

    return eig_min_val, is_psd

################################################################################

def gram_matrix_psd_transformation(dist_2_std, eig_min_val, d=2.5):
    """
    Applies the Lingoes (1971) correction to force a PSD Gram matrix.
    
    The transformation D_new^2 = D^2 + omega(11' - I) shifts all eigenvalues 
    by omega/2. To ensure the smallest eigenvalue becomes non-negative:
        lambda_new = lambda_min + omega/2 >= 0
    
    Since omega is defined as d * |lambda_min|, this implies:
        d * |lambda_min| / 2 >= |lambda_min|  ==>  d >= 2
    
    Parameters
    ----------
    d : float
        Controls the magnitude of the correction. 
        Must be >= 2 to theoretically guarantee PSD properties.
        A value of 2.5 is recommended to account for floating-point numerical errors.
    """
    omega = d * np.abs(eig_min_val)
    
    # Apply additive constant correction: D_new^2 = D^2 + omega(1 - I)
    # Instead of creating full ones/identity matrices, we add scalar and subtract from diagonal
    dist_2_std = dist_2_std + omega
    np.fill_diagonal(dist_2_std, np.diag(dist_2_std) - omega)
    
    # Recompute Gram Matrix with new distances
    gram_matrix = compute_gram_matrix_faster(dist_2_std)

    return gram_matrix

################################################################################

def compute_gram_matrix_sqrt(gram_matrix):
    # Compute Square Root of Gram Matrix: G^(1/2)
    
    # SVD Decomposition: G = U @ diag(S) @ V.T
    # Note: np.linalg.svd returns V transposed (Vt) automatically.
    # U: (n, n) - Left Singular Vectors
    # S: (n,)   - Singular Values (sorted descending)
    # Vt: (n, n) - Right Singular Vectors (already transposed)
    U, S, Vt = np.linalg.svd(gram_matrix)
    
    # Clip numerical noise (singular values of Gram matrix must be >= 0)
    S = np.clip(S, 0, None)
    
    # Reconstruct sqrt(G) = U @ diag(sqrt(S)) @ Vt
    # For symmetric PSD matrices, U is effectively equal to V (up to sign),
    # making this equivalent to an eigendecomposition.
    gram_matrix_sqrt = U @ np.diag(np.sqrt(S)) @ Vt

    return gram_matrix_sqrt

################################################################################

def compute_gram_matrix_sqrt_faster(gram_matrix, n_components=30):

    # Compute Square Root of Gram Matrix: G^(1/2)
    n = len(gram_matrix)
    
    # OPTIMIZATION: Use Truncated approximation if n_components is set
    # Condition: components are fewer than full rank, and matrix is large enough to justify overhead
    if n_components is not None and n_components < n - 1 and n > 100: 
        # Efficient rank-k approximation using Randomized SVD.
        # This avoids ARPACK (Lanczos) convergence issues on matrices with clustered eigenvalues.
        
        # Sigma_rsvd (k,) ~ Sigma_svd (n,)
        # U_rsvd (n, k) ~ U_svd (n, n)
        # VT_rsvd (k, n) ~ Vt_svd (n, n)
        U, Sigma, VT = randomized_svd(gram_matrix, n_components=n_components, random_state=42)
        
        # Extract pseudo-eigenvalues and pseudo-eigenvectors
        # Clip negative values to handle numerical noise
        evals = np.clip(Sigma, 0, None) 
        evecs = U
        
        # Reconstruct sqrt(G) ~= U @ diag(sqrt(Sigma)) @ U.T 
        # Since G is symmetric, U == V, so we can use U.T instead of VT
        # Result dims: (n, k) @ (k, k) @ (k, n) -> (n, n) [Rank-k approximation]
        gram_matrix_sqrt = evecs @ np.diag(np.sqrt(evals)) @ evecs.T 
       
    else:
        # Fallback to full decomposition
        # eigh is highly optimized for symmetric matrices (faster than full SVD)
        
        # S_eigh (n,) = S_svd (n,)
        # Q_eigh (n, n) = U_svd (n, n)
        # Q.T_eigh (n, n) = Vt_svd (n, n)
        evals, evecs = eigh(gram_matrix) # Q = evecs, S = evals
        
        # Clip negative eigenvalues (numerical noise)
        evals = np.clip(evals, 0, None)
        
        # Reconstruct sqrt(G) = Q @ diag(sqrt(S)) @ Q.T (where Q = eigenvectors)
        # Result dims: (n, n) @ (n, n) @ (n, n) -> (n, n) [Exact reconstruction]
        gram_matrix_sqrt = evecs @ np.diag(np.sqrt(evals)) @ evecs.T

    return gram_matrix_sqrt

################################################################################

def compute_cross_product_sum(sqrtG1, sqrtG2, sqrtG3):
    return sqrtG1@sqrtG2 + sqrtG1@sqrtG3 + sqrtG2@sqrtG1 + sqrtG2@sqrtG3 + sqrtG3@sqrtG1 + sqrtG3@sqrtG2

################################################################################

def compute_cross_product_sum_faster(matrices: list[np.ndarray]) -> np.ndarray:
    """
    Efficiently computes the sum of cross-products between all matrices in the list.
    Formula: (Sum(M))^2 - Sum(M^2)
    
    Parameters
    ----------
    matrices : list[np.ndarray]
        A list of square matrices [A, B, C, ...].
        
    Returns
    -------
    np.ndarray
        The result of sum(Mi @ Mj) for all i != j.
    """
    # 1. Sum all matrices (Very cheap: O(N^2))
    # S = A + B + C
    sum_of_matrices = sum(matrices)
    
    # 2. Square the total sum (1 Matrix Multiplication)
    # Total = S @ S
    squared_sum = sum_of_matrices @ sum_of_matrices
    
    # 3. Calculate sum of individual squares (k Matrix Multiplications)
    # Individual = A@A + B@B + C@C
    sum_of_squares = sum(m @ m for m in matrices)
    
    # 4. Subtract to isolate cross-terms
    cross_product_sum = squared_sum - sum_of_squares
    
    return cross_product_sum

################################################################################

def related_metric_scaling_dist_matrix(
        X, p1, p2, p3, d1, d2, d3,
        q=1, robust_method='trimmed', epsilon=0.05, alpha=0.05, n_iters=20,
        weights=None, Gs_PSD_transformation=True, return_combined_distances=False
):
    """
    Calculates the Related Metric Scaling matrix for a data matrix.

    Parameters:
        X  : pandas/polars DataFrame or numpy array of shape (n, p1_total + p2 + p3).
        p1 : int or list of ints. Number of quantitative variables (single block)
             or list [p1_1, ..., p1_k] defining k quantitative blocks.
             If a list is provided, RelMS is applied across all k+2 blocks.
        p2 : number of binary variables.
        p3 : number of multi-class variables.
        d1 : distance for quantitative blocks. Either:
             - a single string from ['euclidean', 'minkowski', 'canberra',
               'pearson', 'mahalanobis', 'robust_mahalanobis'], applied to every
               quantitative block, or
             - a list/tuple of strings of the same length as the number of
               quantitative blocks (i.e. len(p1) if p1 is a list), giving a
               distance per block (e.g. block 1 -> 'euclidean', block 2 ->
               'mahalanobis').
        d2 : distance for binary variables.    One of ['sokal', 'jaccard'].
        d3 : distance for multi-class variables. One of ['hamming'].
        q  : Minkowski parameter (positive int).
        robust_method : robust covariance method; used when d1='robust_mahalanobis'.
                         Global across all blocks.
        alpha         : trimming/winsorising level; used when d1='robust_mahalanobis'.
                         Global across all blocks.
        epsilon       : Delvin transformation parameter.
        n_iters       : max iterations for the Delvin algorithm.
        weights       : sample weights; used when d1='robust_mahalanobis'.
                         Global across all blocks.
        Gs_PSD_transformation : whether to force PSD Gram matrices.
        return_combined_distances : if True, also return the individual distance
                                    matrices as a tuple.

    Returns:
        dist : (n x n) Generalized Gower distance matrix.
        dist_list : only if return_combined_distances=True. List of k + 2
                    (n x n) matrices [D_q1, ..., D_qk, D_bin, D_multi]
                    (non-standardised, non-squared).
    """

    # ------------------------------------------------------------------ #
    # 1. Build per-block distance matrices                               #
    # ------------------------------------------------------------------ #

    dist_list = compute_dist_matrices_by_block(
        X=X, p1=p1, p2=p2, p3=p3, d1=d1, d2=d2, d3=d3,
        q=q, robust_method=robust_method, epsilon=epsilon,
        alpha=alpha, n_iters=n_iters, weights=weights
    )

    # ------------------------------------------------------------------ #
    # 2. Gram matrices (one per block)                                   #
    # ------------------------------------------------------------------ #
    m = len(dist_list)   # total number of blocks = k + 2
    n = dist_list[0].shape[0]
    ones     = np.ones((n, 1))
    ones_T   = np.ones((1, n))
    ones_mat = np.ones((n, n))
    I        = np.identity(n)
    H        = I - (1 / n) * (ones @ ones_T)   # centering matrix

    gram_matrix_list      = []
    gram_matrix_sqrt_list = []

    for i, dist in enumerate(dist_list, start=1):
        dist_2   = dist ** 2
        geom_var = geometric_variability(dist_2)
        dist_2_std = dist_2 / geom_var if geom_var > 1e-10 else dist_2
        gram_matrix = compute_gram_matrix(dist_2_std, H)

        if Gs_PSD_transformation:
            v = np.real(np.linalg.eigvals(gram_matrix))
            v[np.isclose(v, 0, atol=1e-10)] = 0
            if not np.all(v >= 0):
                warnings.warn(
                    f'Gram matrix for block {i} is not PSD; '
                    f'a transformation will be applied.'
                )
            omega      = 2.5 * np.abs(np.min(v))
            dist_2_std = dist_2_std + omega * ones_mat - omega * I
            gram_matrix = -(1 / 2) * (H @ dist_2_std @ H)

        gram_matrix_sqrt = compute_gram_matrix_sqrt(gram_matrix)
        gram_matrix_list.append(gram_matrix)
        gram_matrix_sqrt_list.append(gram_matrix_sqrt)

    # ------------------------------------------------------------------ #
    # 3. RelMS combination:  G* = Σ Gj  -  (1/m) Σ_{i≠j} Gi^½ Gj^½       #
    # ------------------------------------------------------------------ #
    gram_matrices_sum = sum(gram_matrix_list)

    # Cross-product sum for arbitrary number of blocks
    cross_product_sum = np.zeros((n, n))
    for i in range(m):
        for j in range(m):
            if i != j:
                cross_product_sum += gram_matrix_sqrt_list[i] @ gram_matrix_sqrt_list[j]

    gram_matrix_final = gram_matrices_sum - (1 / m) * cross_product_sum

    # ------------------------------------------------------------------ #
    # 4. Recover distances from G*                                       #
    # ------------------------------------------------------------------ #
    g      = np.diag(gram_matrix_final).reshape(-1, 1)
    g_T    = g.T
    dist_2_final = g @ ones_T + ones @ g_T - 2 * gram_matrix_final
    dist_2_final[np.isclose(dist_2_final, 0, atol=1e-10)] = 0
    dist_final = np.sqrt(dist_2_final)

    if return_combined_distances:
        return dist_final, dist_list
    return dist_final


################################################################################

def related_metric_scaling_dist_matrix_faster(
        X, p1, p2, p3, d1, d2, d3,
        q=1, robust_method='trimmed', epsilon=0.05, alpha=0.05, n_iters=20,
        weights=None, Gs_PSD_transformation=True, n_components=10,
        atol=1e-7, return_combined_distances=False
):
    """
    Calculates a faster estimation of the Related Metric Scaling matrix.

    Parameters: identical to related_metric_scaling_dist_matrix (including the
    per-block d1 as str or list/tuple), plus:
        n_components : number of components for the truncated SVD used in the
                       faster Gram-matrix square-root approximation.
        atol         : numerical tolerance for zero-clamping.
    """

    # ------------------------------------------------------------------ #
    # 1. Build per-block distance matrices                               #
    # ------------------------------------------------------------------ #

    dist_list = compute_dist_matrices_by_block(
        X=X, p1=p1, p2=p2, p3=p3, d1=d1, d2=d2, d3=d3,
        q=q, robust_method=robust_method, epsilon=epsilon,
        alpha=alpha, n_iters=n_iters, weights=weights
    )

    # ------------------------------------------------------------------ #
    # 2. Gram matrices                                                   #
    # ------------------------------------------------------------------ #
    m = len(dist_list)
    n = dist_list[0].shape[0]

    gram_matrix_list      = []
    gram_matrix_sqrt_list = []

    for i, dist in enumerate(dist_list, start=1):
        if np.sum(dist) == 0:
            gram_matrix_sqrt_list.append(np.zeros((n, n)))
            gram_matrix_list.append(np.zeros((n, n)))
            continue

        dist_2     = dist ** 2
        geom_var   = geometric_variability(dist_2)
        dist_2_std = dist_2 / geom_var if geom_var > atol else dist_2
        gram_matrix = compute_gram_matrix_faster(dist_2_std)

        if Gs_PSD_transformation:
            eig_min_val, is_psd = check_gram_matrix_psd(gram_matrix, atol)
            if not is_psd:
                warnings.warn(
                    f'Gram matrix for block {i} is not PSD '
                    f'(min eig={eig_min_val:.2e}). Transformation applied.'
                )
                gram_matrix = gram_matrix_psd_transformation(dist_2_std, eig_min_val)

        gram_matrix_sqrt = compute_gram_matrix_sqrt_faster(gram_matrix, n_components)
        gram_matrix_list.append(gram_matrix)
        gram_matrix_sqrt_list.append(gram_matrix_sqrt)

    # ------------------------------------------------------------------ #
    # 3. RelMS combination                                               #
    # ------------------------------------------------------------------ #
    gram_matrices_sum = sum(gram_matrix_list)

    cross_product_sum = np.zeros((n, n))
    for i in range(m):
        for j in range(m):
            if i != j:
                cross_product_sum += gram_matrix_sqrt_list[i] @ gram_matrix_sqrt_list[j]

    gram_matrix_final = gram_matrices_sum - (1 / m) * cross_product_sum

    # ------------------------------------------------------------------ #
    # 4. Recover distances                                               #
    # ------------------------------------------------------------------ #
    g_diag       = np.diag(gram_matrix_final)
    dist_2_final = g_diag[:, None] + g_diag[None, :] - 2 * gram_matrix_final
    dist_2_final[np.abs(dist_2_final) < atol] = 0
    dist_2_final = np.clip(dist_2_final, 0, None)
    dist_final   = np.sqrt(dist_2_final)

    if return_combined_distances:
        return dist_final, dist_list
    return dist_final

################################################################################