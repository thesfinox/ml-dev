import os
import skopt
import joblib
import statsmodels.api as sm

class LinModel:
    '''
    Use a linear model to fit data.
    '''
    
    def __init__(self,
                 X,
                 y,
                 title=None,
                 ctx=None
                ):
        '''
        Arguments:
        
            X:         the features,
            y:         the labels,
            title:     title of the summary table,
            ctx:       Context.
        '''
        
        self.__X     = X
        self.__y     = y
        self.__y0    = None
        
        if title is None:
            self.__title = 'Least Squares Regression Results'
        else:
            self.__title = title
        
        self.__l1_reg = None # l1 regularisation
        self.__l2_reg = None # l2 regularisation
        self.__alpha  = None # elastic net regularisation
        self.__l1_wt  = None # l1 regularisation ratio
        
        self.__ctx = ctx
        
        self.__model   = None
        self.__results = None
        
    def model(self):
        '''
        Return the model.
        '''
        
        # add intercept if needed
        if self.__y0:
            self.__X = sm.add_constant(self.__X)
        
        
        if self.__model is None:
            
            self.__model = sm.OLS(self.__y, self.__X)
        
        return self.__model
    
    def results(self):
        '''
        Return the results.
        '''
        
        if self.__results is None:
            
            self.__results = self.__fitted()
        
        
        return self.__results
    
    def __fitted(self):
        '''
        Perform the fit.
        '''
        
        model = self.model()
        
        if self.__alpha == 0.0:
            
            results = model.fit()
                
        else:
            
            results = model.fit_regularized(alpha=self.__alpha, L1_wt=self.__l1_wt)
            
        return results
    
    def summary(self):
        '''
        Return the summary of the fit.
        '''
        
        results = self.results()
        
        if self.__alpha == 0.0:
            
            return results.summary(title=self.__title)
        
        else:
            
            return results.params
        
    def fit(self, l1_reg=0.0, l2_reg=0.0, intercept=False, verbose=True):
        '''
        Perform the fit.
        
        Arguments:
        
            l1_reg:    L1 regularisation factor,
            l2_reg:    L2 regularisation factor,
            intercept: add an intercept,
            verbose:   verbose output.
        '''
        
        self.__l1_reg = l1_reg
        self.__l2_reg = l2_reg
        self.__y0     = intercept
        self.__alpha  = self.__l1_reg + self.__l2_reg # elastic net regularisation
        if self.__alpha > 0.0:
            self.__l1_wt  = self.__l1_reg / self.__alpha # l1 regularisation ratio
        else:
            self.__l1_wt = 0.0
        
        # get the results
        results = self.results()
            
        if verbose:

            print(self.summary())
            print('\n\n')

        if self.__ctx is not None:

            self.__ctx.logger().info(self.summary())
            
        return self
    
    def save(self, name):
        '''
        Save the results.
        
        Arguments:
        
            name: name of the model file.
        '''
        
        # add extension and location
        name = name + '.joblib'        
        if self.__ctx is not None:
            
            name = os.path.join(self.__ctx.mod(), name)
            self.__ctx.logger().info(f'Model saved to {name}.')
            
        results = self.results()
        results.save(name)
        
        return self
    
    def predict(self, X):
        '''
        Compute the predictions.
        
        Arguments:
        
            X: the features.
        '''
        
        return self.results().predict(X)