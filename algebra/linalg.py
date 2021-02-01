import numpy as np

class LinAlgebra:
    '''
    Linear algebra calculations.
    '''
    
    def __init__(self, X):
        '''
        Arguments:
        
            X: data matrix.
        '''
        
        self.X = X
        self.D = None # eigenvalues
        self.M = None # orthonormal basis
        self.U = None # left eigenvectors
        self.S = None # singular values
        self.V = None # right eigenvectors
        
    def __svd(self):
        '''
        Compute the singular value decomposition.
        '''
        
        U, S, VT = np.linalg.svd(self.X)
        
        self.U = U
        self.S = S
        self.V = V.T
        
    def __svd_red(self):
        '''
        Compute only the singular values.
        '''
        
        self.S = np.linalg.svd(self.X, compute_uv=False)
        
    def cov_eigenvectors(self, right=True):
        '''
        Compute the eigenvectors of a matrix from SVD.
        
        Arguments:
        
            right: return the right eigenvectors (usually the eigenvectors of the covariance matrix).
        '''
        
        if self.U is None or self.V is None:
            self.__svd()
            
        if right:
            return self.V / self.X.shape[0]
        else:
            return self.U / self.X.shape[0]
        
    def singular_values(self):
        '''
        Return the singular values.
        '''
        
        if self.S is None:
            self.__svd_red()
            
        return self.S
    
    def cov_eigenvalues(self):
        '''
        Return the eigenvalues of the covariance matrix.
        '''
        
        return np.square(self.singular_values()) / self.X.shape[0]
    
    def __eig(self):
        '''
        Compute eigenvalues and eigenvectors of a square matrix.
        '''
        
        assert self.X.shape[0] == self.X.shape[1], 'Matrix is not square!'
        
        self.D, self.M = np.linalg.eig(self.X)
            
        
    def eigenvalues(self):
        '''
        Compute the eigenvalues of a square matrix.
        '''
        
        if self.D is None:
            self.__eig()
            
        return self.D
    
    def eigenvectors(self):
        '''
        Compute the eigenvectors of a square matrix.
        '''
        
        if self.M is None:
            self.__eig()
            
        return self.M
        