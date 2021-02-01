import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Marchenko-Pastur distribution
def marchenko_pastur(x, c, var=1.0):
    '''
    Marchenko-Pastur PDF.
    
    Arguments:
    
        x:   the independent variable,
        c:   the ratio between columns and rows,
        var: the variance of the data matrix.
    '''
    
    N = 2.0 * np.pi * c * x * var
    a = var * np.square(1 - np.sqrt(c))
    b = var * np.square(1 + np.sqrt(c))
    
    if a <= x <= b:
        return np.sqrt((b - x) * (x - a)) / N
    else:
        return 0.0

marchenko_pastur = np.vectorize(marchenko_pastur)

class PCAnalysis:
    '''
    Basic operations for PCA analysis.
    '''
    
    def __init__(self,
                 X,
                 standardise=True
                ):
        '''
        Arguments:
        
            X: the data matrix.
        '''
        
        if isinstance(X, pd.DataFrame):
            self.__X = X.values
        else:
            self.__X = X
            
        # standardise the matrix
        if standardise:
            
            mean = np.mean(self.__X, axis=-1).reshape(-1, 1)
            std  = np.std(self.__X, axis=-1).reshape(-1, 1)

            self.__X = (self.__X - mean) / std
            
        # compute the covariance
        self.cov = np.cov(self.__X, rowvar=False)
        
        self.__evalues  = None
        self.__evectors = None
        
    def data(self):
        '''
        Return the data matrix.
        '''
        
        return self.__X
        
    def __eig(self):
        '''
        Compute the eigenvalues.
        '''
        
        self.__evalues, self.__evectors = np.linalg.eigh(self.cov)
        
        return self
    
    def eigenvalues(self):
        '''
        Return the eigenvalues.
        '''
        
        if self.__evalues is None:
            self.__eig()
        
        return self.__evalues
    
    def eigenvectors(self):
        '''
        Return the eigenvectors.
        '''
        
        if self.__evalues is None:
            self.__eig()
        
        return self.__evectors
        
    def distribution(self,
                     xlabel='eigenvalues',
                     ylabel='density',
                     binwidth=None,
                     title=None,
                     logscale=(False,False),
                     name=None,
                     figsize=(6,5),
                     save=False,
                     ctx=None,
                     **kwargs
                    ):
        '''
        Plot the ditribution of the eigenvalues.
        
        Arguments:
        
            xlabel:   label of the x-axis,
            ylabel:   label of the y-axis,
            binwidth: width of the binning,
            title:    the plot title,
            logscale: tuple containing whether to use log scale on the axes,
            name:     name of the figure,
            figsize:  size of the Matplotlib figure,
            save:     save the figure,
            ctx:      Context,
            **kwargs: addition arguments for Seaborn histplot.
        '''
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.tight_layout()
        
        # histplot
        data = self.eigenvalues()
        sns.histplot(data,
                     binwidth=binwidth,
                     stat='density',
                     color='tab:blue',
                     label='eigenvalues',
                     log_scale=logscale,
                     ax=ax,
                     **kwargs
                    )
        
        x = np.arange(data.min(), data.max() + 1.0, 0.001)
        sns.lineplot(x=x,
                     y=marchenko_pastur(x, c=self.__X.shape[1] / self.__X.shape[0]),
                     linestyle='dashed',
                     color='tab:red',
                     label='MP distribution',
                     ax=ax
                    )
        
        # set attributes
        ax.legend()
        
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        
        # save the figure
        if save:
            fig.savefig(os.path.join(ctx.img(), name + '.pdf'), dpi=72)
            
    def eig_select(self):
        '''
        Select the eigenvectors whose eigenvalue is outside the limiting Marchenko-Pastur distribution.
        '''
        
        c = self.__X.shape[1] / self.__X.shape[0]
        
        bot = np.square(1 - np.sqrt(c))
        top = np.square(1 + np.sqrt(c))
        ev  = self.eigenvalues()
        
        mask = (ev < bot) | (ev > bot)
        
        vec = self.eigenvectors()
        
        return vec[:, mask]