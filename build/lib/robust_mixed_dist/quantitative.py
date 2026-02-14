import polars as pl
import numpy as np
import pandas as pd
from itertools import product
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform, cdist
from scipy import sparse

################################################################################

def euclidean_dist_matrix(X):
    """
    Calculates the Euclidean distance matrix for a data matrix using SciPy.

    Parameters (inputs)
    ----------
    X: a Pandas or Polars DataFrame or a NumPy array. It represents a data matrix.

    Returns (outputs)
    -------
    M: the Euclidean distance matrix between the rows of `X`.
    """
    
    # Convert to NumPy array if input is a DataFrame.
    if isinstance(X, pl.DataFrame):
        X = X.to_numpy()
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()    
            
    # Compute the pairwise distances using pdist and convert to a square form.
    M = squareform(pdist(X, metric='euclidean'))
    
    return M

################################################################################

def euclidean_dist(xi, xr) :
    """
    Calculates the Euclidean distance between a pair of vectors.

    Parameters (inputs)
    ----------
    xi, xr: a pair of Pandas or Polars Series or DataFrames. 
            They represent a couple of statistical observations of quantitative variables. 

    Returns (outputs)
    -------
    The Euclidean distance between the observations `xi` and `xr`.
    """

    if isinstance(xi, (pl.DataFrame, pd.DataFrame)) :
        xi = xi.to_numpy().flatten()
    elif isinstance(xi, (pd.Series, pl.Series)) :
        xi = xi.to_numpy() 
    if isinstance(xr, (pl.DataFrame, pd.DataFrame)) :
        xr = xr.to_numpy().flatten()
    elif isinstance(xr, (pd.Series, pl.Series)) :
        xr = xr.to_numpy() 

    return distance.euclidean(xi, xr)

################################################################################

def minkowski_dist_matrix(X, q):
    """
    Calculates the Minkowski distance matrix for a data matrix using SciPy.

    Parameters (inputs)
    ----------
    X: a Pandas or Polars DataFrame or a NumPy array. It represents a data matrix.
    q: the parameters that defines the Minkowski form. Some particular cases: q=1 := Manhattan, q=2 := Euclidean.

    Returns (outputs)
    -------
    M: the Minkowski(`q`) distance matrix between the rows of `X`.
    """   
    
    if isinstance(X, pl.DataFrame):
        X = X.to_numpy()
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()    
            
    # Compute the pairwise distances using pdist and convert to a square form.
    M = squareform(pdist(X, metric='minkowski', p=q))
    
    return M

################################################################################

def minkowski_dist(xi, xr, q) :
    """
    Calculates the Minkowski distance between a pair of vectors.

    Parameters (inputs)
    ----------
    xi, xr: a pair of quantitative vectors. They represent a couple of statistical observations.
    q: the parameters that defines the Minkowski form. Some particular cases: q=1 := Manhattan, q=2 := Euclidean.

    Returns (outputs)
    -------
    The Minkowki(`q`) distance between the observations `xi` and `xr`.
    """

    if isinstance(xi, (pl.DataFrame, pd.DataFrame)) :
        xi = xi.to_numpy().flatten()
    elif isinstance(xi, (pd.Series, pl.Series)) :
        xi = xi.to_numpy() 
    if isinstance(xr, (pl.DataFrame, pd.DataFrame)) :
        xr = xr.to_numpy().flatten()
    elif isinstance(xr, (pd.Series, pl.Series)) :
        xr = xr.to_numpy() 

    return distance.minkowski(xi, xr, q)

################################################################################

def canberra_dist_matrix(X):
    """
    Calculates the Canberra distance matrix for a data matrix using SciPy.

    Parameters (inputs)
    ----------
    X: a pandas/polars DataFrame or a NumPy array. It represents a data matrix.

    Returns (outputs)
    -------
    M: the Canberra distance matrix between the rows of `X`.
    """

    if isinstance(X, pl.DataFrame):
        X = X.to_numpy()
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()    
            
    # Compute the pairwise distances using pdist and convert to a square form.
    M = squareform(pdist(X, metric='canberra'))
    
    return M

################################################################################

def canberra_dist(xi, xr) :
    """
    Calculates the Canberra distance between a pair of vectors.

    Parameters (inputs)
    ----------
    xi, xr: a pair of quantitative vectors. They represent a couple of statistical observations.

    Returns (outputs)
    -------
    The Canberra distance between the observations `xi` and `xr`.
    """

    if isinstance(xi, (pl.DataFrame, pd.DataFrame)) :
        xi = xi.to_numpy().flatten()
    elif isinstance(xi, (pd.Series, pl.Series)) :
        xi = xi.to_numpy() 
    if isinstance(xr, (pl.DataFrame, pd.DataFrame)) :
        xr = xr.to_numpy().flatten()
    elif isinstance(xr, (pd.Series, pl.Series)) :
        xr = xr.to_numpy() 

    return distance.canberra(xi, xr)

################################################################################

def pearson_dist_matrix(X):
    """
    Calculates the Pearson distance matrix for a data matrix using SciPy.

    Parameters (inputs)
    ----------
    X: a pandas/polars DataFrame or a NumPy array. It represents a data matrix.

    Returns (outputs)
    -------
    M: the Pearson distance matrix between the rows of X.
    """

    if isinstance(X, pl.DataFrame):
        X = X.to_numpy()
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()    
        
    # Compute the pairwise distances using pdist and convert to a square form.
    M = squareform(pdist(X, metric='seuclidean'))
    
    return M

################################################################################

def mahalanobis_dist_matrix(X):
    """
    Calculates the classical Mahalanobis distance matrix for a data matrix `X`.

    Parameters
    ----------
    X : pandas.DataFrame, polars.DataFrame, or np.ndarray
        Data matrix of shape (n_samples, n_features).

    Returns
    -------
    D : np.ndarray
        Symmetric matrix (n_samples x n_samples) of Mahalanobis distances.
    """
    
    # Convert to numpy array if needed
    if isinstance(X, pl.DataFrame):
        X = X.to_numpy()
    elif isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Classical covariance matrix
    S = np.cov(X_centered, rowvar=False)

    # Use pseudo-inverse for numerical stability
    S_pinv = np.linalg.pinv(S)

    # Symmetrize just in case
    S_pinv = (S_pinv + S_pinv.T) / 2

    # Compute Mahalanobis distance matrix
    D = cdist(X_centered, X_centered, metric='mahalanobis', VI=S_pinv)

    return D

################################################################################

def mahalanobis_dist(xi, xr, S) :
    """
    Calculates the Mahalanobis distance between a pair of vectors.

    Parameters (inputs)
    ----------
    xi, xr: a pair of quantitative vectors. They represent a couple of statistical observations.
    S: the covariance matrix of the data matrix to which `xi` and `xr` belong.

    Returns (outputs)
    -------
    The Mahalanobis distance between the observations `xi` and `xr`.
    """

    if isinstance(xi, (pl.DataFrame, pd.DataFrame)) :
        xi = xi.to_numpy().flatten()
    elif isinstance(xi, (pd.Series, pl.Series)) :
        xi = xi.to_numpy() 
    if isinstance(xr, (pl.DataFrame, pd.DataFrame)) :
        xr = xr.to_numpy().flatten()
    elif isinstance(xr, (pd.Series, pl.Series)) :
        xr = xr.to_numpy() 
    
    S_inv = np.linalg.inv(S) 
    dist = distance.mahalanobis(xi, xr, S_inv)

    return dist

################################################################################

def mad(Xj) :
    """
    Compute the median absolute deviation of a statistical variable.

    Parameters (inputs)
    ----------
    Xj: a vector representing a quantitative statistical variable.  

    Returns (outputs)
    -------
    MAD: median absolute deviation of `Xj`.
    """

    if isinstance(Xj, pl.Series):
        Xj = Xj.to_numpy()
    elif isinstance(Xj, pd.Series):
        Xj = Xj.to_numpy()

    mad_ = np.median(np.abs(Xj - np.median(Xj)))

    return mad_

################################################################################

def Xj_trimmed(Xj, alpha) : 
    """
    Compute the trimmed version of a statistical variable.

    Parameters (inputs)
    ----------
    Xj : a vector representing a quantitative statistical variable.  
    alpha : a real number in [0,1] that defines the trimming level. 

    Returns (outputs)
    -------
    result: the `alpha` trimmed version of `Xj`.
    """

    if isinstance(Xj, pl.Series):
        Xj = Xj.to_numpy()
    elif isinstance(Xj, pd.Series):
        Xj = Xj.to_numpy()

    lower_bound = np.quantile(Xj, q=alpha/2)  
    upper_bound = np.quantile(Xj, q=1-alpha/2) 
    mask = np.logical_and(Xj >= lower_bound, Xj <= upper_bound)
    Xj_trimmed_ = Xj[mask]
    
    return Xj_trimmed_

################################################################################

def Xj_winsorized(Xj, alpha):
    """
    Compute the winsorized version of a quantitative variable.

    Parameters
    ----------
    Xj : a vector representing a quantitative statistical variable.  
    alpha : a real number in [0,1] that defines the winsorizing level. 

    Returns
    -------
    result: the `alpha` winsorized version of Xj.
    """

    if isinstance(Xj, pl.Series):
        Xj = Xj.to_numpy()
    elif isinstance(Xj, pd.Series):
        Xj = Xj.to_numpy()

    # If Xj is a vector of zeros, return Xj.
    if np.all(Xj == 0):
        return Xj

    lower_bound = np.quantile(Xj, q=alpha/2)
    upper_bound = np.quantile(Xj, q=1-alpha/2)

    # Clip the values: values smaller than lower_bound are set to lower_bound,
    #  those larger than upper_bound are set to upper_bound, 
    # and the ones in the middle are not transform.
    Xj_winsorized_ = np.clip(Xj, lower_bound, upper_bound)

    return Xj_winsorized_

################################################################################

def robust_var(Xj, method, alpha=None) :
    """
    Compute the robust variance of `Xj` allowing different methods.

    Parameters
    ----------
    Xj : a vector representing a quantitative statistical variable.  
    method: the method to be used for computing the robust variance of `Xj`. Must be an string in ['MAD', 'trimmed', 'winsorized'].
    alpha : a real number in [0,1] that is used if `method` is 'trimmed' or 'winsorized'. 

    Returns
    -------
    result: the robust variance of `Xj` computed by the method `method`.
    """

    if method == 'MAD' :
        
        return mad(Xj)**2

    if method == 'trimmed' :
        
        return np.var(Xj_trimmed(Xj, alpha))

    if method == 'winsorized' :
        
        return np.var(Xj_winsorized(Xj, alpha))
    
################################################################################

def robust_corr(Xj, Xr, method, alpha=None) :
    """
    Compute the robust correlation between `Xj` and `Xr` by different methods.

    Parameters
    ----------
    Xj, Xr : two vectors representing a quantitative statistical variables.  
    method: the method to be used for computing the robust variance of `Xj`. Must be an string in ['MAD', 'trimmed', 'winsorized'].
    alpha : a real number in [0,1] that is used if `method` is 'trimmed' or 'winsorized'. 

    Returns
    -------
    result: the robust correlation between `Xj` and `Xr` computed by the method `method`.
    """

    if isinstance(Xj, pl.Series):
        Xj = Xj.to_numpy()
    elif isinstance(Xj, pd.Series):
        Xj = Xj.to_numpy()
    if isinstance(Xr, pl.Series):
        Xr = Xr.to_numpy()
    elif isinstance(Xr, pd.Series):
        Xr = Xr.to_numpy()

    # Si la varianza robusta de X_j es cero, la version estandarizada de X_j es la propia X_j.
    if robust_var(Xj, method, alpha) == 0 :
        Xj_std = Xj
    # Si la varianza robusta de X_j es distinta de cero.
    else :
       # Se estandariza X_j como se especifica en la seccion 7.2.2.
        Xj_std = Xj / np.sqrt(robust_var(Xj, method, alpha))
    # Si la varianza robusta de X_r es cero, la version estandarizada de X_r es la propia X_r.
    if np.sqrt(robust_var(Xr, method, alpha)) == 0 :
        Xr_std = Xr
    # Si la varianza robusta de X_res distinta de cero.
    else :
      # Se estandariza X_r como se especifica en la seccion 7.2.2.  
      Xr_std = Xr / np.sqrt(robust_var(Xr, method, alpha))
      
    # Se calcula la correlacion robusta como se especifica en la seccion 7.2.2, evitando problemas de divisionalidad.
    robust_var_3 = robust_var(Xj_std + Xr_std, method, alpha)
    robust_var_4 = robust_var(Xj_std - Xr_std, method, alpha)
    if (robust_var_3 + robust_var_4) == 0 :
        robust_corr = (robust_var_3 - robust_var_4) 
    else : 
        robust_corr = (robust_var_3 - robust_var_4) / (robust_var_3 + robust_var_4)
    return robust_corr

################################################################################

def R_robust(X, method, alpha=None) :
    """
    Computes the robust correlation matrix of a given data matrix `X`.

    Parameters
    ----------
    X : a pandas/polars data-frame or a numpy array. 
    method : the method used to compute the robust correlation matrix.  Must be an string in ['MAD', 'trimmed', 'winsorized']. 
    alpha : a real number in [0,1] that is used if `method` is 'trimmed' or 'winsorized'.

    Returns
    -------
    M : the robust correlation matrix for `X`.
    """

    if isinstance(X, pl.DataFrame):
        X = X.to_numpy()
    elif isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    
    p = X.shape[1]
    M =  np.zeros((p,p))

    for j,r in product(range(0,p), range(0,p)) :

        M[j,r] = robust_corr(Xj=X[:,j], Xr=X[:,r], method=method, alpha=alpha)

    return M

################################################################################

def delvin_trans(M, epsilon=0.05) : 
    """
    Applies the Delvin transformation for the matrix `M` passed as input to make it positive definite or closer to it.

    Parameters (inputs)
    ----------
    M : a pandas/polars data-frame or a numpy array. 
    epsilon : parameter involved in the Delvin transformation that must be a close to zero positive number. epsilon=0.05 is recommended.

    Returns (outputs)
    -------
    M : the Delvin transformation of the input matrix `M`.
    """

    if isinstance(M, pl.DataFrame):
        M = M.to_numpy()
    elif isinstance(M, pd.DataFrame):
        M = M.to_numpy()   

    # Se define la funcion z.
    def z(x) : 
       return np.arctanh(x)
    
    # Se define la funcion z^{-1}
    def z_inv(x) :  
       # La arctanh es la inversa de tanh, por tanto, la inversa de arctanh es tanh.
       return np.tanh(x) 
    
    # Se define la funcion g.
    def g(i,j, M) :
        if i == j :             
            return 1
        else:
            if np.abs(M[i,j]) <= z(epsilon) :  
                    return 0
            elif M[i,j] < - z(epsilon) : 
                    return z_inv(M[i,j] + epsilon)
            elif M[i,j] > z(epsilon) : 
                    return z_inv(M[i,j] - epsilon)
    
    # Se crea una matriz cuyos elementos son el resultado de aplicar la funcion g sobre matrix elemento a elemento.
    p = M.shape[1]
    M_new =  np.zeros((p,p))
    
    for i,j in product(range(0,p), range(0,p)) :
            
            M_new[i,j] = g(i,j, M)

    return M_new  

################################################################################

def delvin_algorithm(M, epsilon, n_iters):
    """
    Applies the Delvin algorithm on the matrix `M` passed as input 
    to make it positive definite by applying on it the Delvin transformation as many iterations as needed.

    Parameters (inputs)
    ----------
    M: a pandas/polars data-frame or a numpy array. 
    epsilon : parameter used by the Delvin transformation. epsilon=0.05 is recommended.
    n_iter : maximum number of iterations run by the algorithm.

    Returns (outputs)
    -------
    M_new : the resulting matrix of applying the Delvin algorithm on `M`.
    """

    M_new = M.copy()
    # Se inicializa i=0 para entrar en el bucle while
    i = 0
    # Mientras i sea inferior o igual a n_iter, el bucle continua ejecutandose.
    while i < n_iters:
        # Si new_matrix ya es definida positiva (todos sus autovalores son positivos), se devuelve new_matrix. 
        # En otro caso, se le aplica la transformacion de Delvin y se vuelve a comprobar si es definida positiva.
        if np.all(np.linalg.eigvals(M_new) > 0):
            return M_new, i
        else:
            M_new = delvin_trans(M=M_new, epsilon=epsilon)
            i = i + 1
    
    return M_new, i

################################################################################

def S_robust(X, method, alpha, epsilon=0.05, n_iters=20, weights=None):
    """
    Computes the robust covariance of the data matrix `X` by different methods.

    Parameters (inputs)
    ----------
    X: a pandas/polars data-frame or a numpy array. 
    method: the method to be used to compute the robust covariance. Must be an string in ['MAD', 'trimmed', 'winsorized']. 
    alpha : a real number in [0,1] that is used if `method` is 'trimmed' or 'winsorized'.
    epsilon : parameter used by the Delvin transformation. epsilon=0.05 is recommended.
    n_iter : maximum number of iterations run by the Delvin algorithm.
    weights: the sample weights. Only used if provided.

    Returns (outputs)
    -------
    S_robust : the robust covariance matrix computed for `X`.
    """

    if isinstance(X, pl.DataFrame):
        X = X.to_numpy()
    elif isinstance(X, pd.DataFrame):
        X = X.to_numpy() 

    if weights is None:
        # Se calcula la matriz de correlaciones robustas para Data.
        R_robust_ = R_robust(X, method, alpha)
        # Se aplica el algoritmo de Delvin a la matriz de correlaciones robustas calculada.
        R_robust_, i = delvin_algorithm(M=R_robust_, epsilon=epsilon, n_iters=n_iters)
        # Se calcula la matriz de covarianzas robustas a partir de la matriz de correlaciones robustas.
        S_robust = np.diag(np.std(X, axis=0)) @ R_robust_ @ np.diag(np.std(X, axis=0)) 

    else:
        w = weights
        n = len(X)
        Dw = sparse.diags(w)
        I = np.identity(n)
        ones_arr = np.ones(n)
        Jw = np.sqrt(Dw) @ (I - ones_arr @ w.T)
        Xw = Jw @ X # Computational problems when n is too large since is an n x n matrix.
        # Se calcula la matriz de correlaciones robustas para Data.
        R_robust_ = R_robust(Xw, method, alpha)
        # Se aplica el algoritmo de Delvin a la matriz de correlaciones robustas calculada.
        R_robust_, i = delvin_algorithm(M=R_robust_, epsilon=epsilon, n_iters=n_iters)
        # Se calcula la matriz de covarianzas robustas a partir de la matriz de correlaciones robustas.
        S_robust = np.diag(np.std(Xw, axis=0)) @ R_robust_ @ np.diag(np.std(Xw, axis=0)) 

    return S_robust

################################################################################

def robust_mahalanobis_dist_matrix(X, S_robust):
    """
    Calculates the Robust Mahalanobis distance matrix for a data matrix `X`
    using a robust estimation of the covariance matrix.

    Parameters
    ----------
    X : pandas.DataFrame, polars.DataFrame, or np.ndarray
        The input data matrix with shape (n_samples, n_features).
    
    S_robust : np.ndarray
        Robust covariance matrix (e.g., from MCD or a trimmed estimator).
        Should be of shape (n_features, n_features).

    Returns
    -------
    D : np.ndarray
        Symmetric matrix (n_samples, n_samples) of Mahalanobis distances.
    """

    # Convert input to NumPy array if needed
    if isinstance(X, pl.DataFrame):
        X = X.to_numpy()
    elif isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    # Center the data (important for Mahalanobis)
    X_centered = X - np.mean(X, axis=0)

    # Use pseudo-inverse for stability
    S_robust_pinv = np.linalg.pinv(S_robust)

    # Force symmetry (sometimes lost by numerical error)
    S_robust_pinv = (S_robust_pinv + S_robust_pinv.T) / 2

    # Compute pairwise Mahalanobis distances with cdist
    D = cdist(X_centered, X_centered, metric='mahalanobis', VI=S_robust_pinv)

    return D

################################################################################

def robust_mahalanobis_dist(xi, xr, S_robust) :
    """
    Calculates the Robust Mahalanobis distance between a pair of vectors.

    Parameters (inputs)
    ----------
    xi, xr: a pair of quantitative vectors. They represent a couple of statistical observations.
    S_robust: the robust covariance matrix of the data matrix to which `xi` and `xr` belong.

    Returns (outputs)
    -------
    The Robust Mahalanobis distance between the observations `xi` and `xr`.
    """

    if isinstance(xi, (pl.DataFrame, pd.DataFrame)) :
        xi = xi.to_numpy().flatten()
    elif isinstance(xi, (pd.Series, pl.Series)) :
        xi = xi.to_numpy() 
    if isinstance(xr, (pl.DataFrame, pd.DataFrame)) :
        xr = xr.to_numpy().flatten()
    elif isinstance(xr, (pd.Series, pl.Series)) :
        xr = xr.to_numpy() 

    X = np.array([xi, xr])
    dist_xi_xr = robust_mahalanobis_dist_matrix(X, S_robust)
    dist_xi_xr = dist_xi_xr[0,1]
    
    return dist_xi_xr

################################################################################