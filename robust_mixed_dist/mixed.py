import numpy as np

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
    VG1, VG2, VG3 = geometric_variability(D1_2), geometric_variability(D2_2), geometric_variability(D3_2)

    return VG1, VG2, VG3

################################################################################

def generalized_gower_dist_matrix(X, p1, p2, p3, d1, d2, d3, q=1, robust_method='trimmed', alpha=0.05, epsilon=0.05, n_iters=20, weights=None):        
    """
    Calculates the Generalized Gower matrix for a data matrix.
    
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
        weights=weights)
    
    for dist in [dist1, dist2, dist3]:

        dist_2 = dist**2
        geom_var = geometric_variability(dist_2)
        dist_2_std = dist_2/geom_var if geom_var > 0 else dist_2 
        dist_2_std_sum += dist_2_std
    
    dist = np.sqrt(dist_2_std_sum)

    return dist 

################################################################################

def generalized_gower_dist(xi, xr, p1, p2, p3, d1, d2, d3, q=1, S=None, geom_var_1=None, geom_var_2=None, geom_var_3=None):
   
    dist1, dist2, dist3 = compute_distances(xi=xi, xr=xr, p1=p1, p2=p2, p3=p3, d1=d1, d2=d2, d3=d3, q=q, S=S)
    
    for dist, geom_var in zip([dist1, dist2, dist3], [geom_var_1, geom_var_2, geom_var_3]):

        dist_2 = dist**2 
        dist_2_std = dist_2 / geom_var
        dist_2_std_sum += dist_2_std

    dist = np.sqrt(dist_2_std_sum)

    return dist

################################################################################
    
def simple_gower_dist(xi, xr, X, p1, p2, p3) :
    """
    Compute method.
    
    Parameters:
        xi, xr: a pair of quantitative vectors. They represent a couple of statistical observations.
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

def compute_cross_product_sum(matrices: list[np.ndarray]) -> np.ndarray:
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
    sum_of_matrices = np.sum(matrices)
    
    # 2. Square the total sum (1 Matrix Multiplication)
    # Total = S @ S
    squared_sum = sum_of_matrices @ sum_of_matrices
    
    # 3. Calculate sum of individual squares (k Matrix Multiplications)
    # Individual = A@A + B@B + C@C
    sum_of_squares = np.sum(m @ m for m in matrices)
    
    # 4. Subtract to isolate cross-terms
    cross_product_sum = squared_sum - sum_of_squares
    
    return cross_product_sum

################################################################################

# TODO: OPTIMIZAR LA FUNCION 

def related_metric_scaling_dist_matrix(
        X, p1, p2, p3,d1, d2, d3, 
        q=1, robust_method='trimmed', epsilon=0.05, alpha=0.05, n_iters=20, weights=None,
        tol=1e-6, d=2.5, Gs_PSD_transformation=True
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
     
    gram_matrix_sqrt = []
    for i, dist in enumerate([dist1, dist2, dist3], start=1):

        dist_2 = dist**2
        geom_var = geometric_variability(dist_2)
        dist_2_std = dist_2/geom_var if geom_var > 0 else dist_2 
        
        gram_matrix = -(1/2)*(centering_matrix @ dist_2_std @ centering_matrix)
        gram_matrix_sum += gram_matrix

        if Gs_PSD_transformation == True :

            v = np.real(np.linalg.eigvals(gram_matrix))
            v[np.isclose(v, 0, atol=tol)] = 0
            gram_matrix_psd = np.all(v >= 0)
            if not gram_matrix_psd:
                raise Warning(f'Gram matrix for d{i} is not PSD, a transformation to force it will be applied.')   
            omega = d * np.abs(np.min(v))  
            dist_2_std  = dist_2_std + omega*ones_matrix - omega*identity_matrix
            gram_matrix = -(1/2)*(centering_matrix)

        U, S, V = np.linalg.svd(gram_matrix)
        S = np.clip(S, 0, None)
        gram_matrix_sqrt.append(U @ np.diag(np.sqrt(S)) @ V)
    
    gram_matrices_cross_product_sum = compute_cross_product_sum([gram_matrix_sqrt[i] for i in range(len(gram_matrix_sqrt))]) 
    gram_matrix = gram_matrix_sum - (1/3) * gram_matrices_cross_product_sum
    g = np.diag(gram_matrix) 
    g =  np.reshape(g, (len(g), 1))  
    g_T = np.reshape(g, (1, len(g)))   
    dist_2 = g @ ones_T + ones @ g_T - 2*gram_matrix
    dist_2[np.isclose(dist_2, 0, atol=tol)] = 0
    dist = np.sqrt(dist_2)

    return dist




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

        G = G_1 + G_2 + G_3 - (1/3)*(sqrtG1@sqrtG2 + sqrtG1@sqrtG3 + sqrtG2@sqrtG1 + sqrtG2@sqrtG3 + sqrtG3@sqrtG1 + sqrtG3@sqrtG2)
        g = np.diag(G) 
        g =  np.reshape(g, (len(g), 1))  
        g_T = np.reshape(g, (1, len(g)))   
        D_2_ = g @ ones_T + ones @ g_T - 2*G
        D_2_[np.isclose(D_2_, 0, atol=tol)] = 0
        D = np.sqrt(D_2_)
 
        return D    

################################################################################

def data_preprocessing(X, frac_sample_size,   ):
    """
    Preprocess data in the way as needed by `FastGG` class.

    Parameters (inputs)
    ----------
    X: a pandas/polars data-frame.
    frac_sample_size: the sample size in proportional terms.
      : the random seed for the random elements of the function.

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
        np.random.seed(  )
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

'''
class FastGGowerDistMatrix:
    """
    Calculates the the Generalized Gower matrix of a sample of a given data matrix.
    """

    def __init__(self, frac_sample_size=0.1,   =123, p1=None, p2=None, p3=None, 
                 d1='robust_mahalanobis', d2='jaccard', d3='matching', 
                 robust_method='trimmed', alpha=0.05, epsilon=0.05, n_iters=20, q=1, 
                  =False,  =1000,  =5, weights=None) :
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
             : whether the geometric variability estimation will be full (False) or fast (True).
             : sample size to be used to make the estimation of the geometric variability.
             : number of samples to be used to make the estimation of the geometric variability.
              : the random seed used for the (random) sample elements.
            weights: the sample weights. Only used if provided and d1 = 'robust_mahalanobis'.  
        """
        self.   =   ; self.frac_sample_size = frac_sample_size; self.p1 = p1; self.p2 = p2; self.p3 = p3; 
        self.d1 = d1; self.d2 = d2; self.d3 = d3; self.robust_method = robust_method; self.alpha = alpha; self.epsilon = epsilon; 
        self.n_iters = n_iters; self.  =  ; self.  =  ; self.  =  ; 
        self.q = q; self.weights = weights

    def compute(self, X):
        """
        Compute method: computes the Generalized Gower function for the defined sample of data.
        
        Parameters:
            X: a pandas/polars data-frame or a numpy array. Represents a data matrix.
        """

        X_sample, X_out_sample, sample_index, out_sample_index = data_preprocessing(X=X, frac_sample_size=self.frac_sample_size, 
                                                                                      =self.  )
       
        sample_weights = self.weights[sample_index] if self.weights is not None else None

        GGower_matrix = GGowerDistMatrix(p1=self.p1, p2=self.p2, p3=self.p3, 
                                         d1=self.d1, d2=self.d2, d3=self.d3, q=self.q,
                                         robust_method=self.robust_method, alpha=self.alpha, 
                                         epsilon=self.epsilon, n_iters=self.n_iters,
                                          =self. ,  =self. , 
                                          =self. , weights=sample_weights)
        
        self.D_GGower = GGower_matrix.compute(X=X_sample)
        self.sample_index = sample_index
        self.out_sample_index = out_sample_index
        self.X_sample = X_sample
        self.X_out_sample = X_out_sample
'''

################################################################################