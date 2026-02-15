################################################################################

import numpy as np
import warnings
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigvalsh
from scipy.linalg import eigh
from sklearn.utils.extmath import randomized_svd

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

def simple_gower_dist(xi, xr, X, p1, p2, p3) :
    """
    Compute method.
    
    Parameters:
        xi, xr: a pair of mixed data vectors. They represent a couple of statistical observations.
        X: a pandas/polars data-frame or a numpy array. It represents a data matrix.
        p1, p2, p3: number of quantitative, binary and multi-class variables in the considered data matrix, respectively. Must be a non negative integer.

    Returns:
        dist: the Simple Gower distance between the observations `xi` and `xr`.
    """    

    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    xi = ensure_flat_array(xi)
    xr = ensure_flat_array(xr)

    dist_objects = get_dist_objects()

    X_quant = X[:,0:p1]  
    xi_quant = xi[0:p1] ; xr_quant = xr[0:p1] ; 
    xi_bin = xi[(p1):(p1+p2)] ; xr_bin = xr[(p1):(p1+p2)]
    xi_multi = xi[(p1+p2):(p1+p2+p3)] ; xr_multi = xr[(p1+p2):(p1+p2+p3)]
    R = np.max(X_quant, axis=0) - np.min(X_quant, axis=0)

    dist1 = np.sum(np.abs(xi_quant - xr_quant)/R) if p1 > 0 else 0
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
        R = np.max(X_quant, axis=0) - np.min(X_quant, axis=0)
        R[R == 0] = 1  # evitar división por cero
        X_quant_norm = X_quant / R
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
        return np.sum(D_2) / (2 * (n**2))

    if weights.shape[0] != n:
            raise ValueError(f"Weights dimension ({weights.shape[0]}) does not match with D_2 dimension ({n}).")
    
    # Normalize weights if needed: ensure sum is 1.0 for the standard formula
    sum_weights = np.sum(weights)
    if not np.isclose(sum_weights, 1.0):
        if np.isclose(sum_weights, 0.0):
             raise ValueError("Sum of weights cannot be zero.")
        weights = weights / sum_weights

    return (weights @ D_2 @ weights) / 2.0


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
    Calculates the distances between observations that are involved in the Generalized Gower distance.
       
    Parameters:
        xi, xr: a pair of quantitative vectors. They represent a couple of statistical observations.
        p1, p2, p3: number of quantitative, binary and multi-class variables in the considered data matrix, respectively. Must be a non negative integer.
        d1: name of the distance to be computed for quantitative variables. Must be an string in ['euclidean', 'minkowski', 'canberra', 'mahalanobis', 'robust_mahalanobis']. 
        d2: name of the distance to be computed for binary variables. Must be an string in ['sokal', 'jaccard'].
        d3: name of the distance to be computed for multi-class variables. Must be an string in ['matching'].
        q: the parameter that defines the Minkowski distance. Must be a positive integer.
        S: the covariance matrix of the considered data matrix.
        S_robust: the robust covariance matrix of the considered data matrix.
                  
    Returns:
        dist1, dist2, dist3: the distances values associated to the quantitative, binary and multi-class observations, respectively.
    """

    xi = ensure_flat_array(xi)
    xr = ensure_flat_array(xr)

    dist_objects = get_dist_objects()
                   
    xi_quant = xi[0:p1]  
    xr_quant = xr[0:p1] 
    xi_bin = xi[(p1):(p1+p2)] 
    xr_bin = xr[(p1):(p1+p2)]
    xi_multi = xi[(p1+p2):(p1+p2+p3)] 
    xr_multi = xr[(p1+p2):(p1+p2+p3)]
    
    dist1 = 0
    if p1 > 0:
        if d1 == 'minkowski':
            dist1 = dist_objects[d1](xi_quant, xr_quant, q=q)
        elif d1 == 'robust_mahalanobis':
            dist1 = dist_objects[d1](xi_quant, xr_quant, S_robust=S)
        elif d1 == 'mahalanobis':
            dist1 = dist_objects[d1](xi_quant, xr_quant, S=S)
        else:
            dist1 = dist_objects[d1](xi_quant, xr_quant)

    dist2 = dist_objects[d2](xi_bin, xr_bin) if p2 > 0 else 0
    dist3 = dist_objects[d3](xi_multi, xr_multi) if p3 > 0 else 0

    return dist1, dist2, dist3
    
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
        VG1, VG2, VG3: the geometric variabilities of the distances matrices associated to the quantitative, binary and multi-class variables, respectively.
    """

    D1, D2, D3 = compute_dist_matrices(
        X=X, p1=p1, p2=p2, p3=p3, d1=d1, d2=d2, d3=d3,
        q=q, robust_method=robust_method, epsilon=epsilon,
        alpha=alpha, n_iters=n_iters, weights=weights
    )
       
    D1_2, D2_2, D3_2 = D1**2, D2**2, D3**2
    VG1, VG2, VG3 = geometric_variability(D1_2, weights), geometric_variability(D2_2, weights), geometric_variability(D3_2, weights)

    return VG1, VG2, VG3

################################################################################

'''
class GGowerDistMatrix: 
    """
    Calculates the Generalized Gower matrix for a data matrix.
    """

    def __init__(self, p1, p2, p3, d1='euclidean', d2='sokal', d3='matching', q=1, robust_method='trimmed', epsilon=0.05, alpha=0.05, n_iters=20, weights=None):
        """
        Constructor method.
        
        Parameters:
            p1, p2, p3: number of quantitative, binary and multi-class variables in the considered data matrix, respectively. Must be a non negative integer.
            d1: name of the distance to be computed for quantitative variables. Must be an string in ['euclidean', 'minkowski', 'canberra', 'mahalanobis', 'robust_mahalanobis']. 
            d2: name of the distance to be computed for binary variables. Must be an string in ['sokal', 'jaccard'].
            d3: name of the distance to be computed for multi-class variables. Must be an string in ['hamming'].
            q: the parameter that defines the Minkowski distance. Must be a positive integer.
            metrobust_methodhod: the robust_method to be used for computing the robust covariance matrix. Only needed when d1 = 'robust_mahalanobis'.
            alpha : a real number in [0,1] that is used if `robust_method` is 'trimmed' or 'winsorized'. Only needed when d1 = 'robust_mahalanobis'.
            epsilon : parameter used by the Delvin transformation. epsilon=0.05 is recommended. Only needed when d1 = 'robust_mahalanobis'.
            n_iter : maximum number of iterations run by the Delvin algorithm. Only needed when d1 = 'robust_mahalanobis'.
            weights: the sample weights. Only used if provided and d1 = 'robust_mahalanobis'.  
            fast_VG: whether the geometric variability estimation will be full (False) or fast (True).
            VG_sample_size: sample size to be used to make the estimation of the geometric variability.
            VG_n_samples: number of samples to be used to make the estimation of the geometric variability.
            random_state: the random seed used for the (random) sample elements.
        """
        self.p1 = p1 ; self.p2 = p2 ; self.p3 = p3
        self.d1 = d1 ; self.d2 = d2 ; self.d3 = d3
        self.q = q ; self.robust_method = robust_method ; self.alpha = alpha ; 
        self.epsilon = epsilon ; self.n_iters = n_iters; self.weights = weights

    def compute(self, X):
        """
        Compute method.
        
        Parameters:
            X: a pandas/polars data-frame or a numpy array. Represents a data matrix.
            
        Returns:
            D: the Generalized Gower matrix for the data matrix `X`.
        """

        D1, D2, D3 = compute_dist_matrices(X=X, p1=self.p1, p2=self.p2, p3=self.p3, 
                                               d1=self.d1, d2=self.d2, d3=self.d3, 
                                               q=self.q, robust_method=self.robust_method, epsilon=self.epsilon, 
                                               alpha=self.alpha, n_iters=self.n_iters, weights=self.weights)

        D1_2 = D1**2  ; D2_2 = D2**2 ; D3_2 = D3**2

        VG1, VG2, VG3 = geometric_variability(D1_2, self.weights), geometric_variability(D2_2, self.weights), geometric_variability(D3_2, self.weights)

        D1_std = D1_2/VG1 if VG1 > 0 else D1_2 
        D2_std = D2_2/VG2 if VG2 > 0 else D2_2 
        D3_std = D3_2/VG3 if VG3 > 0 else D3_2
        D_2 = D1_std + D2_std + D3_std
        D = np.sqrt(D_2)

        return D 
'''

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
        p1, p2, p3: number of quantitative, binary and multi-class variables in the considered data matrix, respectively. Must be a non negative integer.
        d1: name of the distance to be computed for quantitative variables. Must be an string in ['euclidean', 'minkowski', 'canberra', 'mahalanobis', 'robust_mahalanobis']. 
        d2: name of the distance to be computed for binary variables. Must be an string in ['sokal', 'jaccard'].
        d3: name of the distance to be computed for multi-class variables. Must be an string in ['hamming'].
        q: the parameter that defines the Minkowski distance. Must be a positive integer.
        robust_method: the robust_method to be used for computing the robust covariance matrix. Only needed when d1 = 'robust_mahalanobis'.
        alpha : a real number in [0,1] that is used if `robust_method` is 'trimmed' or 'winsorized'. Only needed when d1 = 'robust_mahalanobis'.
        epsilon : parameter used by the Delvin transformation. epsilon=0.05 is recommended. Only needed when d1 = 'robust_mahalanobis'.
        n_iter : maximum number of iterations run by the Delvin algorithm. Only needed when d1 = 'robust_mahalanobis'.
        weights: the sample weights. Only used if provided and d1 = 'robust_mahalanobis'.  
    
    Returns:
        D: the Generalized Gower matrix for the data matrix `X`.
    """
    dist1, dist2, dist3 = compute_dist_matrices(
        X=X, 
        p1=p1, 
        p2=p2, 
        p3=p3, 
        d1=d1, 
        d2=d2, 
        d3=d3, 
        q=q, 
        robust_method=robust_method, 
        epsilon=epsilon, 
        alpha=alpha, 
        n_iters=n_iters, 
        weights=weights
    )
    
    n = len(dist1)
    dist_2_std_sum = np.zeros((n, n))

    for dist in [dist1, dist2, dist3]:

        dist_2 = dist**2
        geom_var = geometric_variability(dist_2, weights=weights)
        dist_2_std = dist_2 / geom_var if geom_var > 1e-15 else dist_2 
        dist_2_std_sum += dist_2_std
    
    dist = np.sqrt(dist_2_std_sum)

    if return_combined_distances:
        return dist, dist1, dist2, dist3

    return dist 

################################################################################

def generalized_gower_dist(xi, xr, p1, p2, p3, d1, d2, d3, q=1, S=None, geom_var_1=None, geom_var_2=None, geom_var_3=None):
    """
    Calculates the Generalized Gower distance between a pair of mixed data vectors.
    
    Parameters:
        xi, xr: 1D array-like. They represent a couple of statistical observations (mixed data vectors).
        p1, p2, p3: number of quantitative, binary and multi-class variables in the considered data vectors, respectively. Must be a non negative integer.
        d1: name of the distance to be computed for quantitative variables. Must be an string in ['euclidean', 'minkowski', 'canberra', 'mahalanobis', 'robust_mahalanobis']. 
        d2: name of the distance to be computed for binary variables. Must be an string in ['sokal', 'jaccard'].
        d3: name of the distance to be computed for multi-class variables. Must be an string in ['hamming'].
        q: the parameter that defines the Minkowski distance. Must be a positive integer.
        S: the covariance matrix (standard or robust) to be used. Only needed when d1 is 'mahalanobis' or 'robust_mahalanobis'.
        geom_var_1, geom_var_2, geom_var_3: geometric variability of the quantitative, binary, and multi-class distances, respectively. Used to standardize the squared distances.
    
    Returns:
        dist: the Generalized Gower distance between the observations `xi` and `xr`.
    """
   
    dist1, dist2, dist3 = compute_distances(xi=xi, xr=xr, p1=p1, p2=p2, p3=p3, d1=d1, d2=d2, d3=d3, q=q, S=S)
    
    dist_2_std_sum = 0

    for dist, geom_var in zip([dist1, dist2, dist3], [geom_var_1, geom_var_2, geom_var_3]):

        dist_2 = dist**2 
        dist_2_std = dist_2 / geom_var
        dist_2_std_sum += dist_2_std

    dist = np.sqrt(dist_2_std_sum)

    return dist

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

def check_gram_matrix_psd(gram_matrix, atol=1e-15):
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
        X, p1, p2, p3,d1, d2, d3, 
        q=1, robust_method='trimmed', epsilon=0.05, alpha=0.05, n_iters=20, weights=None,
        Gs_PSD_transformation=True, return_combined_distances=False
):
    """
    Calculates the Related Metric Scaling matrix for a data matrix.
    
    Parameters:
        X: a pandas/polars data-frame or a numpy array. Represents a data matrix.
        p1, p2, p3: number of quantitative, binary and multi-class variables in the considered data matrix, respectively. Must be a non negative integer.
        d1: name of the distance to be computed for quantitative variables. Must be an string in ['euclidean', 'minkowski', 'canberra', 'mahalanobis', 'robust_mahalanobis']. 
        d2: name of the distance to be computed for binary variables. Must be an string in ['sokal', 'jaccard'].
        d3: name of the distance to be computed for multi-class variables. Must be an string in ['hamming'].
        q: the parameter that defines the Minkowski distance. Must be a positive integer.
        metrobust_methodhod: the robust_method to be used for computing the robust covariance matrix. Only needed when d1 = 'robust_mahalanobis'.
        alpha : a real number in [0,1] that is used if `robust_method` is 'trimmed' or 'winsorized'. Only needed when d1 = 'robust_mahalanobis'.
        epsilon : parameter used by the Delvin transformation. epsilon=0.05 is recommended. Only needed when d1 = 'robust_mahalanobis'.
        n_iter : maximum number of iterations run by the Delvin algorithm. Only needed when d1 = 'robust_mahalanobis'.
        weights: the sample weights. Only used if provided and d1 = 'robust_mahalanobis'.  
        tol: a tolerance value to round the close-to-zero eigenvalues of the Gramm matrices.
        Gs_PSD_trans: controls if a transformation is applied to enforce positive semi-definite Gramm matrices.
        d: a parameter that controls the omega definition involved in the transformation mentioned above.    

    Returns:
        D: the Related Metric Scaling matrix for the data matrix `X`.
    """

    dist1, dist2, dist3 = compute_dist_matrices(
        X=X, 
        p1=p1, 
        p2=p2, 
        p3=p3, 
        d1=d1, 
        d2=d2, 
        d3=d3, 
        q=q, 
        robust_method=robust_method, 
        epsilon=epsilon, 
        alpha=alpha, 
        n_iters=n_iters, 
        weights=weights)
    
    n = len(dist1)
    ones = np.ones((n, 1)) 
    ones_T = np.ones((1, n))
    ones_matrix = np.ones((n, n))
    identity_matrix = np.identity(n)
    centering_matrix = identity_matrix - (1/n)*(ones @ ones_T)
     
    gram_matrix_list, gram_matrix_sqrt_list = [], []

    for i, dist in enumerate([dist1, dist2, dist3], start=1):

        dist_2 = dist**2
        geom_var = geometric_variability(dist_2)
        dist_2_std = dist_2/geom_var if geom_var > 1e-15 else dist_2 
        gram_matrix = compute_gram_matrix(dist_2_std, centering_matrix)

        if Gs_PSD_transformation == True :
            v = np.real(np.linalg.eigvals(gram_matrix))
            v[np.isclose(v, 0, atol=1e-15)] = 0
            gram_matrix_psd = np.all(v >= 0)
            if not gram_matrix_psd:
                warnings.warn(f'Gram matrix for d{i} is not PSD, a transformation to force it will be applied.')   
            omega = 2.5 * np.abs(np.min(v))  
            dist_2_std  = dist_2_std + omega*ones_matrix - omega*identity_matrix
            gram_matrix = -(1/2)*(centering_matrix @ dist_2_std @ centering_matrix)
        
        gram_matrix_sqrt = compute_gram_matrix_sqrt(gram_matrix)   

        gram_matrix_list.append(gram_matrix)
        gram_matrix_sqrt_list.append(gram_matrix_sqrt)
            
    gram_matrices_sum = sum(gram_matrix_list)
    gram_matrices_sqrt_cross_product_sum = compute_cross_product_sum(sqrtG1=gram_matrix_sqrt_list[0], sqrtG2=gram_matrix_sqrt_list[1], sqrtG3=gram_matrix_sqrt_list[2]) 
    gram_matrix_final = gram_matrices_sum - (1/3) * gram_matrices_sqrt_cross_product_sum
    g = np.diag(gram_matrix_final) 
    g =  np.reshape(g, (len(g), 1))  
    g_T = np.reshape(g, (1, len(g)))   
    dist_2_final = g @ ones_T + ones @ g_T - 2*gram_matrix_final
    dist_2_final[np.isclose(dist_2_final, 0, atol=1e-15)] = 0
    dist_final = np.sqrt(dist_2_final)

    if return_combined_distances:
        return dist_final, dist1, dist2, dist3
    
    return dist_final

################################################################################

def related_metric_scaling_dist_matrix_faster(
        X, p1, p2, p3,d1, d2, d3, 
        q=1, robust_method='trimmed', epsilon=0.05, alpha=0.05, n_iters=20, weights=None,
        Gs_PSD_transformation=True, n_components=10, atol=1e-7, return_combined_distances=False
):
    """
    Calculates a faster estimation of the Related Metric Scaling matrix for a data matrix.
    
    Parameters:
        X: a pandas/polars data-frame or a numpy array. Represents a data matrix.
        p1, p2, p3: number of quantitative, binary and multi-class variables in the considered data matrix, respectively. Must be a non negative integer.
        d1: name of the distance to be computed for quantitative variables. Must be an string in ['euclidean', 'minkowski', 'canberra', 'mahalanobis', 'robust_mahalanobis']. 
        d2: name of the distance to be computed for binary variables. Must be an string in ['sokal', 'jaccard'].
        d3: name of the distance to be computed for multi-class variables. Must be an string in ['hamming'].
        q: the parameter that defines the Minkowski distance. Must be a positive integer.
        robust_method: the robust_method to be used for computing the robust covariance matrix. Only needed when d1 = 'robust_mahalanobis'.
        alpha : a real number in [0,1] that is used if `robust_method` is 'trimmed' or 'winsorized'. Only needed when d1 = 'robust_mahalanobis'.
        epsilon : parameter used by the Delvin transformation. epsilon=0.05 is recommended. Only needed when d1 = 'robust_mahalanobis'.
        n_iter : maximum number of iterations run by the Delvin algorithm. Only needed when d1 = 'robust_mahalanobis'.
        weights: the sample weights. Only used if provided and d1 = 'robust_mahalanobis'.  
        tol: a tolerance value to round the close-to-zero eigenvalues of the Gramm matrices.
        Gs_PSD_trans: controls if a transformation is applied to enforce positive semi-definite Gramm matrices.
        d: a parameter that controls the omega definition involved in the transformation mentioned above.    

    Returns:
        D: the Related Metric Scaling matrix for the data matrix `X`.
    """
    dist1, dist2, dist3 = compute_dist_matrices(
        X=X, 
        p1=p1, 
        p2=p2, 
        p3=p3, 
        d1=d1, 
        d2=d2, 
        d3=d3, 
        q=q, 
        robust_method=robust_method, 
        epsilon=epsilon, 
        alpha=alpha, 
        n_iters=n_iters, 
        weights=weights)
    
    n = len(dist1)

    # Pre-allocate accumulator for the sum of Gram matrices
    gram_matrices_sum = np.zeros((n, n))
    gram_matrix_list, gram_matrix_sqrt_list = [], []

    for i, dist in enumerate([dist1, dist2, dist3], start=1):
        
        # Handle cases where a variable type is missing (empty distance matrix)
        if np.sum(dist) == 0:
            gram_matrix_sqrt_list.append(np.zeros((n, n)))
            continue
        
        dist_2 = dist**2
        geom_var = geometric_variability(dist_2)
        dist_2_std = dist_2/geom_var if geom_var > atol else dist_2 

        gram_matrix = compute_gram_matrix_faster(dist_2_std)

        # PSD Check and Transformation
        if Gs_PSD_transformation:
            eig_min_val, is_psd = check_gram_matrix_psd(gram_matrix, atol)
            if not is_psd:
                warnings.warn(f'Gram matrix for d{i} is not PSD (min eig={eig_min_val:.2e}). Transformation applied.')
                gram_matrix = gram_matrix_psd_transformation(dist_2_std, eig_min_val)
         
        gram_matrix_sqrt = compute_gram_matrix_sqrt_faster(gram_matrix, n_components)  
        
        gram_matrix_list.append(gram_matrix)
        gram_matrix_sqrt_list.append(gram_matrix_sqrt)
    
    # Sum individual gram matrices sum(Gj for j=1,2,3)
    gram_matrices_sum = sum(gram_matrix_list)
    # Calculate cross-products sum using the optimized helper
    # Formula: sum(Gi^1/2 Gj^1/2) for i!=j
    gram_matrices_sqrt_cross_product_sum = compute_cross_product_sum_faster(gram_matrix_sqrt_list) 
    # Final Gram Matrix Combination: G_relms = sum(G_k) - (1/m) * sum_{k!=l} ...
    # m = 3 in this specific implementation context    
    gram_matrix_final = gram_matrices_sum - (1/3) * gram_matrices_sqrt_cross_product_sum

    # Recover Euclidean Distances from Final Gram Matrix
    # Formula: D^2_ij = G_ii + G_jj - 2G_ij
    g_diag = np.diag(gram_matrix_final)
    
    # Use broadcasting for O(n^2) reconstruction
    dist_2_final = g_diag[:, None] + g_diag[None, :] - 2 * gram_matrix_final
    
    # Cleanup numerical noise
    dist_2_final[np.abs(dist_2_final) < atol] = 0
    dist_2_final = np.clip(dist_2_final, 0, None)
    
    # Recover de final distance matrix
    dist_final = np.sqrt(dist_2_final)

    if return_combined_distances:
        return dist_final, dist1, dist2, dist3
    
    return dist_final

################################################################################

def related_metric_scaling_dist(
    xi, xr, 
    p1, p2, p3, d1, d2, d3, 
    q=1, S=None, 
    geom_var_1=None, geom_var_2=None, geom_var_3=None,
    Gs_PSD_transformation=True, 
    n_components=30
):
    """
    Calculates the Related Metric Scaling distance between a pair of observations (xi, xr).
    
    It projects the pair into the RMS spectral space using the provided geometric 
    variabilities (geom_var_k) and covariance matrix (S) derived from the training sample.
    
    Parameters match generalized_gower_dist for consistency in clustering algorithms.
    """
    
    # 1. Compute raw scalar distances between the pair
    # Returns 3 scalar values (one for each variable type)
    d_v1, d_v2, d_v3 = compute_distances(
        xi=xi, xr=xr, 
        p1=p1, p2=p2, p3=p3, 
        d1=d1, d2=d2, d3=d3, 
        q=q, S=S
    )
    
    # Group inputs for iteration
    raw_distances = [d_v1, d_v2, d_v3]
    geom_vars = [geom_var_1, geom_var_2, geom_var_3]
    
    # We are working in a local context of N=2 (xi and xr)
    n = 2 
    gram_matrix_list = []
    gram_matrix_sqrt_list = []
    
    # Adjust n_components: we cannot extract more components than the rank (2)
    k_components = min(n_components, n)

    # 2. Process each metric type
    for i, (dist_val, g_var) in enumerate(zip(raw_distances, geom_vars)):
        
        # Handle missing data or zero variance gracefully
        if dist_val is None or g_var is None or g_var < 1e-15:
            # If a metric is missing/invalid, it contributes 0 to the geometric space
            gram_matrix_list.append(np.zeros((n, n)))
            gram_matrix_sqrt_list.append(np.zeros((n, n)))
            continue

        # Standardize the squared distance using the GLOBAL geometric variability
        dist_2 = dist_val**2
        dist_2_std = dist_2 / g_var
        
        # 3. Construct the 2x2 Squared Distance Matrix
        # We simulate a mini-distance matrix for the pair to apply spectral logic
        # D^2 = [[0,      d^2_std],
        #        [d^2_std, 0     ]]
        D_2_matrix = np.array([
            [0.0, dist_2_std],
            [dist_2_std, 0.0]
        ])
        
        # 4. Apply Optimized Double Centering (Broadcasting)
        # Even for 2x2, this correctly centers the matrix in the spectral space
        gram_matrix = compute_gram_matrix_faster(D_2_matrix)
        
        # 5. PSD Check and Transformation (Robustness)
        if Gs_PSD_transformation:
            eig_min_val, is_psd = check_gram_matrix_psd(gram_matrix)
            if not is_psd:
                gram_matrix = gram_matrix_psd_transformation(D_2_matrix, eig_min_val)
        
        # 6. Compute Matrix Square Root (SVD/Eigen)
        # Using the faster function constrained to k=2 components
        gram_matrix_sqrt = compute_gram_matrix_sqrt_faster(gram_matrix, n_components=k_components)
        
        gram_matrix_list.append(gram_matrix)
        gram_matrix_sqrt_list.append(gram_matrix_sqrt)

    # 7. RMS Aggregation Formula (The Core Logic)
    # G_relms = sum(G_k) - (1/m) * sum_{k!=l} (G_k^1/2 * G_l^1/2)
    
    # Sum of Gram Matrices
    gram_matrices_sum = sum(gram_matrix_list)
    
    # Sum of Cross-Products: uses the optimized helper function
    gram_matrices_sqrt_cross_product_sum = compute_cross_product_sum_faster(gram_matrix_sqrt_list)
    
    # Final Combination (m=3 fixed for this implementation)
    m = 3
    gram_matrix_final = gram_matrices_sum - (1/m) * gram_matrices_sqrt_cross_product_sum
    
    # 8. Recover Euclidean Distance from the Final Gram Matrix
    # We want the distance between index 0 (xi) and index 1 (xr)
    # Formula: D^2_ij = G_ii + G_jj - 2G_ij
    
    g00 = gram_matrix_final[0, 0]
    g11 = gram_matrix_final[1, 1]
    g01 = gram_matrix_final[0, 1]
    
    dist_2_final = g00 + g11 - 2 * g01
    
    # Numerical safety (clip negative zeros)
    if dist_2_final < 1e-15:
        return 0.0
        
    dist_final = np.sqrt(dist_2_final)

    return dist_final

################################################################################
################################################################################
################################################################################


def get_dist_matrix_functions():
        
    dist_matrix = {}
    dist_matrix['euclidean'] = euclidean_dist_matrix
    dist_matrix['minkowski'] = minkowski_dist_matrix
    dist_matrix['canberra'] = canberra_dist_matrix
    dist_matrix['pearson'] = pearson_dist_matrix
    dist_matrix['mahalanobis'] = mahalanobis_dist_matrix
    dist_matrix['robust_mahalanobis'] = robust_mahalanobis_dist_matrix
    dist_matrix['sokal'] = sokal_dist_matrix
    dist_matrix['jaccard'] = jaccard_dist_matrix
    dist_matrix['hamming'] = hamming_dist_matrix

    return dist_matrix

################################################################################

def get_dist_functions():

    dist = {}
    dist['euclidean'] = euclidean_dist
    dist['minkowski'] = minkowski_dist
    dist['canberra'] = canberra_dist
    dist['mahalanobis'] = mahalanobis_dist
    dist['robust_mahalanobis'] = robust_mahalanobis_dist
    dist['sokal'] = sokal_dist
    dist['jaccard'] = jaccard_dist
    dist['hamming'] = hamming_dist

    return dist

################################################################################

def get_distances(xi, xr, p1, p2, p3, d1='euclidean', d2='sokal', d3='matching', q=1, S=None, S_robust=None):
    """
    Calculates the distances between observations that are involved in the Generalized Gower distance.
       
    Parameters:
        xi, xr: a pair of quantitative vectors. They represent a couple of statistical observations.
        p1, p2, p3: number of quantitative, binary and multi-class variables in the considered data matrix, respectively. Must be a non negative integer.
        d1: name of the distance to be computed for quantitative variables. Must be an string in ['euclidean', 'minkowski', 'canberra', 'mahalanobis', 'robust_mahalanobis']. 
        d2: name of the distance to be computed for binary variables. Must be an string in ['sokal', 'jaccard'].
        d3: name of the distance to be computed for multi-class variables. Must be an string in ['matching'].
        q: the parameter that defines the Minkowski distance. Must be a positive integer.
        S: the covariance matrix of the considered data matrix.
        S_robust: the robust covariance matrix of the considered data matrix.
                  
    Returns:
        dist1, dist2, dist3: the distances values associated to the quantitative, binary and multi-class observations, respectively.
    """

    if isinstance(xi, (pl.DataFrame, pd.DataFrame)) :
        xi = xi.to_numpy().flatten()
    elif isinstance(xi, (pd.Series, pl.Series)) :
        xi = xi.to_numpy() 
    if isinstance(xr, (pl.DataFrame, pd.DataFrame)) :
        xr = xr.to_numpy().flatten()
    elif isinstance(xr, (pd.Series, pl.Series)) :
        xr = xr.to_numpy() 

    dist = get_dist_functions()
                   
    xi_quant = xi[0:p1] ; xr_quant = xr[0:p1] ; 
    xi_bin = xi[(p1):(p1+p2)] ; xr_bin = xr[(p1):(p1+p2)]
    xi_multi = xi[(p1+p2):(p1+p2+p3)] ; xr_multi = xr[(p1+p2):(p1+p2+p3)]

    if p1 > 0:
        if d1 == 'minkowski':
            dist1 = dist[d1](xi_quant, xr_quant, q=q)
        elif d1 == 'robust_mahalanobis':
            dist1 = dist[d1](xi_quant, xr_quant, S_robust=S_robust)
        elif d1 == 'mahalanobis':
            dist1 = dist[d1](xi_quant, xr_quant, S=S)
        else:
            dist1 = dist[d1](xi_quant, xr_quant)
    elif p1 == 0:
        dist1 = 0

    dist2 = dist[d2](xi_bin, xr_bin) if p2 > 0 else 0
    dist3 = dist[d3](xi_multi, xr_multi) if p3 > 0 else 0

    return dist1, dist2, dist3

################################################################################

def get_dist_matrices(X, p1, p2, p3, d1='euclidean', d2='sokal', d3='matching', q=1, 
                      robust_method='trimmed', epsilon=0.05, alpha=0.05, n_iters=20, weights=None):
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
 
    if isinstance(X, (pl.DataFrame, pd.DataFrame)):
        X = X.to_numpy()

    dist_matrix = get_dist_matrix_functions()

    n = len(X)
    X_quant = X[:, 0:p1] 
    X_bin = X[:, (p1):(p1+p2)]
    X_multi = X[:, (p1+p2):(p1+p2+p3)]

    # Define D1 based on d1 and p1
    if p1 > 0:
        if d1 == 'minkowski':
            D1 = dist_matrix[d1](X_quant, q)
        elif d1 == 'robust_mahalanobis':
            S_robust_ = S_robust(X=X_quant, method=robust_method, alpha=alpha,
                                    epsilon=epsilon, n_iters=n_iters, 
                                    weights=weights)
            D1 = dist_matrix[d1](X_quant, S_robust=S_robust_)
        else:
            D1 = dist_matrix[d1](X_quant)
    elif p1 == 0:
        D1 = np.zeros((n, n))
    # Define D2 based on p2
    D2 = dist_matrix[d2](X_bin) if p2 > 0 else np.zeros((n, n)) 
    # Define D3 based on p3
    D3 = dist_matrix[d3](X_multi) if p3 > 0 else np.zeros((n, n))

    return D1, D2, D3

################################################################################

def vg(D_2):
    """
    Calculates the geometric variability of the squared distance matrix `D_2` passed as input.

    Parameters (inputs)
    ----------
    D_2: a numpy array. It should represent an squared distance matrix.

    Returns (outputs)
    -------
    VG: the geometric variability of the squared distance matrix `D_2`.
    """
    n = len(D_2)
    VG = (1/(2*(n**2)))*np.sum(D_2)
    # TO DO: version managing weights
    return VG

################################################################################

def vg_ggower_estimation(X, p1, p2, p3, d1='euclidean', d2='sokal', d3='matching', 
                         q=1, robust_method='trimmed', epsilon=0.05, alpha=0.05, 
                         n_iters=20, weights=None): 
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
        VG1, VG2, VG3: the geometric variabilities of the distances matrices associated to the quantitative, binary and multi-class variables, respectively.
    """

    D1, D2, D3 = get_dist_matrices(X=X, p1=p1, p2=p2, p3=p3, d1=d1, d2=d2, d3=d3,
                                   q=q, robust_method=robust_method, epsilon=epsilon,
                                   alpha=alpha, n_iters=n_iters, weights=weights)
       
    D1_2, D2_2, D3_2 = D1**2, D2**2, D3**2
    VG1, VG2, VG3 = vg(D1_2), vg(D2_2), vg(D3_2)

    return VG1, VG2, VG3

################################################################################

def vg_ggower_fast_estimation(X, p1, p2, p3, d1='euclidean', d2='sokal', d3='matching',
                         robust_method='trimmed', epsilon=0.05, alpha=0.05, n_iters=20, q=1,
                         VG_sample_size=300, VG_n_samples=5, random_state=123, weights=None):
    """
    Calculates a fast estimation of the geometric variability of an squared Generalized Gower distance matrix.
            
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
        VG_sample_size: sample size to be used to make the estimation of the geometric variability.
        VG_n_samples: number of samples to be used to make the estimation of the geometric variability.
        random_state: the random seed used for the (random) sample elements.
        
    Returns:
        VG1, VG2, VG3: the geometric variabilities of the distances matrices associated to the quantitative, binary and multi-class variables, respectively.
    """
        
    if isinstance(X, (pl.DataFrame, pd.DataFrame)) :
        X = X.to_numpy()  
    
    n = len(X)
    VG1_list, VG2_list, VG3_list = [], [], []

    for i in range(0, VG_n_samples) :

        np.random.seed(random_state + i)
        index = np.arange(0,n)
        sample_index = np.random.choice(index, size=VG_sample_size)
        X_sample = X[sample_index,:].copy()
        
        if weights is not None:
            sample_weights = weights[sample_index].copy() 
        else:
            sample_weights = None
        
        VG1, VG2, VG3 = vg_ggower_estimation(X=X_sample, p1=p1, p2=p2, p3=p3, d1=d1, d2=d2, d3=d3, q=q,
                                            robust_method=robust_method, epsilon=epsilon, alpha=alpha, 
                                            n_iters=n_iters, weights=sample_weights)
        
        VG1_list.append(VG1) ; VG2_list.append(VG2) ; VG3_list.append(VG3) 

    VG1 = np.mean(VG1_list) ; VG2 = np.mean(VG2_list) ; VG3 = np.mean(VG3_list)

    return VG1, VG2, VG3

################################################################################

class GGowerDistMatrix: 
    """
    Calculates the Generalized Gower matrix for a data matrix.
    """

    def __init__(self, p1, p2, p3, d1='euclidean', d2='sokal', d3='matching', q=1, robust_method='trimmed', epsilon=0.05, alpha=0.05, n_iters=20,
                 fast_VG=False, VG_sample_size=300, VG_n_samples=5, random_state=123, weights=None):
        """
        Constructor method.
        
        Parameters:
            p1, p2, p3: number of quantitative, binary and multi-class variables in the considered data matrix, respectively. Must be a non negative integer.
            d1: name of the distance to be computed for quantitative variables. Must be an string in ['euclidean', 'minkowski', 'canberra', 'mahalanobis', 'robust_mahalanobis']. 
            d2: name of the distance to be computed for binary variables. Must be an string in ['sokal', 'jaccard'].
            d3: name of the distance to be computed for multi-class variables. Must be an string in ['hamming'].
            q: the parameter that defines the Minkowski distance. Must be a positive integer.
            metrobust_methodhod: the robust_method to be used for computing the robust covariance matrix. Only needed when d1 = 'robust_mahalanobis'.
            alpha : a real number in [0,1] that is used if `robust_method` is 'trimmed' or 'winsorized'. Only needed when d1 = 'robust_mahalanobis'.
            epsilon : parameter used by the Delvin transformation. epsilon=0.05 is recommended. Only needed when d1 = 'robust_mahalanobis'.
            n_iter : maximum number of iterations run by the Delvin algorithm. Only needed when d1 = 'robust_mahalanobis'.
            weights: the sample weights. Only used if provided and d1 = 'robust_mahalanobis'.  
            fast_VG: whether the geometric variability estimation will be full (False) or fast (True).
            VG_sample_size: sample size to be used to make the estimation of the geometric variability.
            VG_n_samples: number of samples to be used to make the estimation of the geometric variability.
            random_state: the random seed used for the (random) sample elements.
        """
        self.p1 = p1 ; self.p2 = p2 ; self.p3 = p3
        self.d1 = d1 ; self.d2 = d2 ; self.d3 = d3
        self.q = q ; self.robust_method = robust_method ; self.alpha = alpha ; 
        self.epsilon = epsilon ; self.n_iters = n_iters
        self.VG_sample_size = VG_sample_size; self.VG_n_samples = VG_n_samples
        self.random_state = random_state ; self.fast_VG = fast_VG; self.weights = weights

    def compute(self, X):
        """
        Compute method.
        
        Parameters:
            X: a pandas/polars data-frame or a numpy array. Represents a data matrix.
            
        Returns:
            D: the Generalized Gower matrix for the data matrix `X`.
        """

        D1, D2, D3 = get_dist_matrices(X=X, p1=self.p1, p2=self.p2, p3=self.p3, 
                                               d1=self.d1, d2=self.d2, d3=self.d3, 
                                               q=self.q, robust_method=self.robust_method, epsilon=self.epsilon, 
                                               alpha=self.alpha, n_iters=self.n_iters, weights=self.weights)
     
        D1_2 = D1**2  ; D2_2 = D2**2 ; D3_2 = D3**2

        if self.fast_VG == True:   
            VG1, VG2, VG3 = vg_ggower_fast_estimation(X=X, p1=self.p1, p2=self.p2, p3=self.p3, 
                                                   d1=self.d1, d2=self.d2, d3=self.d3, 
                                                   robust_method=self.robust_method, alpha=self.alpha,
                                                   VG_sample_size=self.VG_sample_size, VG_n_samples=self.VG_n_samples, 
                                                   random_state=self.random_state, weights=self.weights)
        else:
            VG1, VG2, VG3 = vg(D1_2), vg(D2_2), vg(D3_2)

        D1_std = D1_2/VG1 if VG1 > 0 else D1_2 
        D2_std = D2_2/VG2 if VG2 > 0 else D2_2 
        D3_std = D3_2/VG3 if VG3 > 0 else D3_2
        D_2 = D1_std + D2_std + D3_std
        D = np.sqrt(D_2)

        return D 

################################################################################
 
class GGowerDist: 
    """
    Calculates the Generalized Gower distance for a pair of data observations.
    """

    def __init__(self, p1, p2, p3, d1='euclidean', d2='sokal', d3='matching', q=1, robust_method='trimmed', alpha=0.05, epsilon=0.05, n_iters=20,
                 VG_sample_size=300, VG_n_samples=5, random_state=123, weights=None):
        """
        Constructor method.
        
        Parameters:
            p1, p2, p3: number of quantitative, binary and multi-class variables in the considered data matrix, respectively. Must be a non negative integer.
            d1: name of the distance to be computed for quantitative variables. Must be an string in ['euclidean', 'minkowski', 'canberra', 'mahalanobis', 'robust_mahalanobis']. 
            d2: name of the distance to be computed for binary variables. Must be an string in ['sokal', 'jaccard'].
            d3: name of the distance to be computed for multi-class variables. Must be an string in ['matching'].
            q: the parameter that defines the Minkowski distance. Must be a positive integer.
            robust_method: the robust_method to be used for computing the robust covariance matrix. Only needed when d1 = 'robust_mahalanobis'.
            epsilon: parameter used by the Delvin algorithm that is used when computing the robust covariance matrix. Only needed when d1 = 'robust_mahalanobis'.
            n_iter: maximum number of iterations used by the Delvin algorithm. Only needed when d1 = 'robust_mahalanobis'.
            weights: the sample weights. Only used if provided and d1 = 'robust_mahalanobis'.  
            VG_sample_size: sample size to be used to make the estimation of the geometric variability.
            VG_n_samples: number of samples to be used to make the estimation of the geometric variability.
            random_state: the random seed used for the (random) sample elements.
        """
        self.p1 = p1 ; self.p2 = p2 ; self.p3 = p3
        self.d1 = d1 ; self.d2 = d2 ; self.d3 = d3
        self.q = q ; self.robust_method = robust_method ; self.alpha = alpha ; 
        self.epsilon = epsilon ; self.n_iters = n_iters
        self.VG_sample_size = VG_sample_size; self.VG_n_samples = VG_n_samples
        self.random_state = random_state; self.weights = weights

    def fit(self, X) :
        """
        Fit method that computes the geometric variability and covariance matrix to be used in 'compute' method, if needed.
        
        Parameters:
            X: a pandas/polars data-frame or a numpy array. Represents a data matrix.
            
        Returns:
            D: the Generalized Gower matrix for the data matrix `X`.
        """
        p1 = self.p1 ; p2 = self.p2 ; p3 = self.p3
        d1 = self.d1 ; d2 = self.d2 ; d3 = self.d3
        self.S, self.S_robust = None, None

        if d1 in ['robust_mahalanobis', 'mahalanobis']:

            if isinstance(X, (pl.DataFrame, pd.DataFrame)) :
                X = X.to_numpy()
                
            X_quant = X[:, 0:p1] 

            if d1 == 'robust_mahalanobis':
                self.S_robust = S_robust(X=X_quant, method=self.robust_method, alpha=self.alpha, 
                                            epsilon=self.epsilon, n_iters=self.n_iters, weights=self.weights)
            elif d1 == 'mahalanobis':
                self.S = np.cov(X_quant, rowvar=False)

        self.VG1, self.VG2, self.VG3 = vg_ggower_fast_estimation(X=X, p1=p1, p2=p2, p3=p3, d1=d1, d2=d2, d3=d3, robust_method=self.robust_method, 
                                                                 alpha=self.alpha, epsilon=self.epsilon, n_iters=self.n_iters,
                                                                 VG_sample_size=self.VG_sample_size, VG_n_samples=self.VG_n_samples, 
                                                                 random_state=self.random_state, weights=self.weights)
    
    def compute(self, xi, xr):
        """
        Compute method.
        
        Parameters:
            xi, xr: a pair of quantitative vectors. They represent a couple of statistical observations.
            
        Returns:
            dist: the Generalized Gower distance between the observations `xi` and `xr`.
        """
        dist1, dist2, dist3 = get_distances(xi=xi, xr=xr, p1=self.p1, p2=self.p2, p3=self.p3, 
                                            d1=self.d1, d2=self.d2, d3=self.d3, 
                                            q=self.q, S=self.S, S_robust=self.S_robust)
        
        dist1_2 = dist1**2 ; dist2_2 = dist2**2 ; dist3_2 = dist3**2
        dist1_2_std = dist1_2/self.VG1 if self.VG1 > 0 else dist1_2 
        dist2_2_std = dist2_2/self.VG2 if self.VG2 > 0 else dist2_2 
        dist3_2_std = dist3_2/self.VG3 if self.VG3 > 0 else dist3_2 
        dist_2 = dist1_2_std + dist2_2_std + dist3_2_std
        dist = np.sqrt(dist_2)

        return dist
    
################################################################################

import polars as pl
import pandas as pd

def data_preprocessing(X, frac_sample_size, random_state):
    """
    Preprocess data in the way as needed by `FastGG` class.

    Parameters (inputs)
    ----------
    X: a pandas/polars data-frame.
    frac_sample_size: the sample size in proportional terms.
    random_state: the random seed for the random elements of the function.

    Returns (outputs)
    -------
    X_sample: a polars df with the sample of `X`.
    X_out_sample: a polars df with the out of sample of `X`.
    sample_index: the index of the sample observations/rows.
    out_sample_index: the index of the out of sample observations/rows.
    """

    if not (0 < frac_sample_size <= 1):
       raise ValueError('frac_sample_size must be in (0,1].')

    if isinstance(X, (pd.DataFrame, pl.DataFrame)):
        X = X.to_numpy()
    
    n = len(X)

    if frac_sample_size < 1:
        n_sample = int(frac_sample_size*n)
        index = np.arange(0,n)
        np.random.seed(random_state)
        sample_index = np.random.choice(index, size=n_sample, replace=False)
        out_sample_index = np.array([x for x in index if x not in sample_index])
        X_sample = X[sample_index,:] 
        X_out_sample = X[out_sample_index,:] 
    else:
        X_sample = X
        sample_index =  np.arange(0,n)
        X_out_sample = np.array([])
        out_sample_index = np.array([])

    return X_sample, X_out_sample, sample_index, out_sample_index

################################################################################

class FastGGowerDistMatrix:
    """
    Calculates the the Generalized Gower matrix of a sample of a given data matrix.
    """

    def __init__(self, frac_sample_size=0.1, random_state=123, p1=None, p2=None, p3=None, 
                 d1='robust_mahalanobis', d2='jaccard', d3='matching', 
                 robust_method='trimmed', alpha=0.05, epsilon=0.05, n_iters=20, q=1, 
                 fast_VG=False, VG_sample_size=1000, VG_n_samples=5, weights=None) :
        """
        Constructor method.
        
        Parameters:
            frac_sample_size: the sample size in proportional terms.
            p1, p2, p3: number of quantitative, binary and multi-class variables in the considered data matrix, respectively. Must be a non negative integer.
            d1: name of the distance to be computed for quantitative variables. Must be an string in ['euclidean', 'minkowski', 'canberra', 'mahalanobis', 'robust_mahalanobis']. 
            d2: name of the distance to be computed for binary variables. Must be an string in ['sokal', 'jaccard'].
            d3: name of the distance to be computed for multi-class variables. Must be an string in ['matching'].
            q: the parameter that defines the Minkowski distance. Must be a positive integer.
            robust_method: the method to be used for computing the robust covariance matrix. Only needed when d1 = 'robust_mahalanobis'.
            alpha : a real number in [0,1] that is used if `method` is 'trimmed' or 'winsorized'. Only needed when d1 = 'robust_mahalanobis'.
            epsilon: parameter used by the Delvin algorithm that is used when computing the robust covariance matrix. Only needed when d1 = 'robust_mahalanobis'.
            n_iters: maximum number of iterations used by the Delvin algorithm. Only needed when d1 = 'robust_mahalanobis'.
            fast_VG: whether the geometric variability estimation will be full (False) or fast (True).
            VG_sample_size: sample size to be used to make the estimation of the geometric variability.
            VG_n_samples: number of samples to be used to make the estimation of the geometric variability.
            random_state: the random seed used for the (random) sample elements.
            weights: the sample weights. Only used if provided and d1 = 'robust_mahalanobis'.  
        """
        self.random_state = random_state; self.frac_sample_size = frac_sample_size; self.p1 = p1; self.p2 = p2; self.p3 = p3; 
        self.d1 = d1; self.d2 = d2; self.d3 = d3; self.robust_method = robust_method; self.alpha = alpha; self.epsilon = epsilon; 
        self.n_iters = n_iters; self.fast_VG = fast_VG; self.VG_sample_size = VG_sample_size; self.VG_n_samples = VG_n_samples; 
        self.q = q; self.weights = weights

    def compute(self, X):
        """
        Compute method: computes the Generalized Gower function for the defined sample of data.
        
        Parameters:
            X: a pandas/polars data-frame or a numpy array. Represents a data matrix.
        """

        X_sample, X_out_sample, sample_index, out_sample_index = data_preprocessing(X=X, frac_sample_size=self.frac_sample_size, 
                                                                                    random_state=self.random_state)
       
        sample_weights = self.weights[sample_index] if self.weights is not None else None

        GGower_matrix = GGowerDistMatrix(p1=self.p1, p2=self.p2, p3=self.p3, 
                                         d1=self.d1, d2=self.d2, d3=self.d3, q=self.q,
                                         robust_method=self.robust_method, alpha=self.alpha, 
                                         epsilon=self.epsilon, n_iters=self.n_iters,
                                         fast_VG=self.fast_VG, VG_sample_size=self.VG_sample_size, 
                                         VG_n_samples=self.VG_n_samples, weights=sample_weights)
        
        self.D_GGower = GGower_matrix.compute(X=X_sample)
        self.sample_index = sample_index
        self.out_sample_index = out_sample_index
        self.X_sample = X_sample
        self.X_out_sample = X_out_sample

################################################################################

class RelMSDistMatrix: 
    """
    Calculates the Related Metric Scaling matrix for a data matrix.
    """

    def __init__(self, p1,p2,p3,d1='euclidean',d2='sokal',d3='matching',q=1, robust_method='trimmed', 
                 epsilon=0.05, alpha=0.05, n_iters=20, weights=None):
        """
        Constructor method.
        
        Parameters:
            p1, p2, p3: number of quantitative, binary and multi-class variables in the considered data matrix, respectively. Must be a non negative integer.
            d1: name of the distance to be computed for quantitative variables. Must be an string in ['euclidean', 'minkowski', 'canberra', 'mahalanobis', 'robust_mahalanobis']. 
            d2: name of the distance to be computed for binary variables. Must be an string in ['sokal', 'jaccard'].
            d3: name of the distance to be computed for multi-class variables. Must be an string in ['matching'].
            q: the parameter that defines the Minkowski distance. Must be a positive integer.
            robust_method: the robust_method to be used for computing the robust covariance matrix. Only needed when d1 = 'robust_mahalanobis'.
            epsilon: parameter used by the Delvin algorithm that is used when computing the robust covariance matrix. Only needed when d1 = 'robust_mahalanobis'.
            n_iters: maximum number of iterations used by the Delvin algorithm. Only needed when d1 = 'robust_mahalanobis'.
            weights: the sample weights. Only used if provided and d1 = 'robust_mahalanobis'.  
        """
        self.p1 = p1 ; self.p2 = p2 ; self.p3 = p3
        self.d1 = d1 ; self.d2 = d2 ; self.d3 = d3
        self.q = q ; self.robust_method = robust_method ; self.alpha = alpha 
        self.epsilon = epsilon ; self.n_iters = n_iters ; self.weights = weights


    def compute(self, X, tol=1e-6, d=2.5, Gs_PSD_transformation=True):
        """
        Compute method.
        
        Parameters:
            X: a pandas/polars data-frame or a numpy array. Represents a data matrix.
            tol: a tolerance value to round the close-to-zero eigenvalues of the Gramm matrices.
            Gs_PSD_trans: controls if a transformation is applied to enforce positive semi-definite Gramm matrices.
            d: a parameter that controls the omega definition involved in the transformation mentioned above.
            
        Returns:
            D: the Related Metric Scaling matrix for the data matrix `X`.
        """
        D1, D2, D3  = compute_dist_matrices(X=X, p1=self.p1, p2=self.p2, p3=self.p3, 
                                               d1=self.d1, d2=self.d2, d3=self.d3, 
                                               q=self.q, robust_method=self.robust_method, epsilon=self.epsilon, 
                                               alpha=self.alpha, n_iters=self.n_iters, weights=self.weights)
       
        D1_2 = D1**2  ; D2_2 = D2**2 ; D3_2 = D3**2

        VG1, VG2, VG3 =   geometric_variability(D1_2),   geometric_variability(D2_2),   geometric_variability(D3_2)

        D1_std = D1_2/VG1 if VG1 > 0 else D1_2 
        D2_std = D2_2/VG2 if VG2 > 0 else D2_2 
        D3_std = D3_2/VG3 if VG3 > 0 else D3_2
 
        n = len(D1)
        ones = np.ones((n, 1)) 
        ones_T = np.ones((1, n))
        ones_M = np.ones((n, n))
        I = np.identity(n)
        H = I - (1/n)*(ones @ ones_T)
        G_1 = -(1/2)*(H @ D1_std @ H)
        G_2 = -(1/2)*(H @ D2_std @ H) 
        G_3 = -(1/2)*(H @ D3_std @ H)

        if Gs_PSD_transformation == True :

            v1 = np.real(np.linalg.eigvals(G_1))
            v2 = np.real(np.linalg.eigvals(G_2))
            v3 = np.real(np.linalg.eigvals(G_3))
            v1[np.isclose(v1, 0, atol=tol)] = 0 
            v2[np.isclose(v2, 0, atol=tol)] = 0 
            v3[np.isclose(v3, 0, atol=tol)] = 0
            G1_PSD = np.all(v1 >= 0)
            G2_PSD = np.all(v2 >= 0) 
            G3_PSD = np.all(v3 >= 0)

            if not G1_PSD :
                
                print('G1 is not PSD, a transformation to force it will be applied.')

                omega = d * np.abs(np.min(v1))  
                D1_std  = D1_std + omega*ones_M - omega*I
                G_1 = -(1/2)*(H @ D1_std @ H)

            if not G2_PSD :

                print('G2 is not PSD, a transformation to force it will be applied.')
                omega = d * np.abs(np.min(v2)) 
                D2_std = D2_std + omega*ones_M - omega*I
                G_2 = -(1/2)*(H @ D2_std @ H)

            if not G3_PSD :

                print('G3 is not PSD, a transformation to force it will be applied.')
                omega = d * np.abs(np.min(v3))  
                D3_std = D3_std + omega*ones_M - omega*I
                G_3 = -(1/2)*(H @ D3_std @ H) 
        
        U1, S1, V1 = np.linalg.svd(G_1) 
        U2, S2, V2 = np.linalg.svd(G_2)   
        U3, S3, V3 = np.linalg.svd(G_3)
        S1 = np.clip(S1, 0, None)
        S2 = np.clip(S2, 0, None)
        S3 = np.clip(S3, 0, None)
        sqrtG1 = U1 @ np.diag(np.sqrt(S1)) @ V1 
        sqrtG2 = U2 @ np.diag(np.sqrt(S2)) @ V2 
        sqrtG3 = U3 @ np.diag(np.sqrt(S3)) @ V3

        G_cross_product_sum = compute_cross_product_sum(sqrtG1, sqrtG2, sqrtG3)
        G = G_1 + G_2 + G_3 - (1/3) * G_cross_product_sum
        g = np.diag(G) 
        g =  np.reshape(g, (len(g), 1))  
        g_T = np.reshape(g, (1, len(g)))   
        D_2_ = g @ ones_T + ones @ g_T - 2*G
        D_2_[np.isclose(D_2_, 0, atol=tol)] = 0
        D = np.sqrt(D_2_)
 
        return D    

################################################################################
################################################################################
################################################################################