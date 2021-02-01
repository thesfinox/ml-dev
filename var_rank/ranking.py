import os
import re
from time import time, strftime, gmtime
import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class VarRank:
    '''
    Rank variables using boosted decision trees.
    '''
    
    def __init__(self,
                 ctx=None,
                 objective='regression',
                 boosting='gbdt',
                 learning_rate=1.0e-2,
                 num_leaves=31,
                 tree_learner='serial',
                 num_thread=0,
                 device_type='cpu',
                 seed=None,
                 deterministic=False,
                 force_col_wise=False,
                 force_row_wise=False,
                 histogram_pool_size=-1.0,
                 max_depth=-1,
                 min_data_in_leaf=20,
                 min_sum_hessian_in_leaf=1.0e-3,
                 bagging_fraction=1.0,
                 pos_bagging_fraction=1.0,
                 neg_bagging_fraction=1.0,
                 bagging_freq=0,
                 bagging_seed=3,
                 feature_fraction=1.0,
                 feature_fraction_bynode=1.0,
                 feature_fraction_seed=2,
                 extra_trees=False,
                 extra_seed=6,
                 first_metric_only=False,
                 max_delta_step=0.0,
                 lambda_l1=0.0,
                 lambda_l2=0.0,
                 min_gain_to_split=0.0,
                 drop_rate=0.1,
                 max_drop=50,
                 skip_drop=0.5,
                 xgboost_dart_mode=False,
                 uniform_drop=False,
                 drop_seed=4,
                 top_rate=0.2,
                 other_rate=0.1,
                 min_data_per_group=100,
                 max_cat_threshold=32,
                 cat_l2=10.0,
                 cat_smooth=10.0,
                 max_cat_to_onehot=4,
                 top_k=20,
                 monotone_constraints=None,
                 monotone_constraints_method='basic',
                 monotone_penalty=0.0,
                 feature_contrib=None,
                 forcedsplits_filename='',
                 refit_decay_rate=0.9,
                 path_smooth=0,
                 interaction_constraints='',
                 verbosity=0
                ):
        '''
        Arguments:
        
            ctx: Context (provide for logging).
        
        See https://lightgbm.readthedocs.io/en/latest/Parameters.html for a list of parameters of the LightGBM.
        '''
        
        self.__ctx    = ctx
        self.__gbm    = None
        self.__y_pred = None
        self.__obj    = objective
        self.__rank   = None
        self.params   = {'objective': objective,
                         'boosting': boosting,
                         'learning_rate': learning_rate,
                         'num_leaves': num_leaves,
                         'tree_learner': tree_learner,
                         'num_thread': num_thread,
                         'device_type': device_type,
                         'seed': seed,
                         'deterministic': deterministic,
                         'force_col_wise': force_col_wise,
                         'force_row_wise': force_row_wise,
                         'histogram_pool_size': histogram_pool_size,
                         'max_depth': max_depth,
                         'min_data_in_leaf': min_data_in_leaf,
                         'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
                         'bagging_fraction': bagging_fraction,
                         'pos_bagging_fraction': pos_bagging_fraction,
                         'neg_bagging_fraction': neg_bagging_fraction,
                         'bagging_freq': bagging_freq,
                         'bagging_seed': bagging_seed,
                         'feature_fraction': feature_fraction,
                         'feature_fraction_bynode': feature_fraction_bynode,
                         'feature_fraction_seed': feature_fraction_seed,
                         'extra_trees': extra_trees,
                         'extra_seed': extra_seed,
                         'first_metric_only': first_metric_only,
                         'max_delta_step': max_delta_step,
                         'lambda_l1': lambda_l1,
                         'lambda_l2': lambda_l2,
                         'min_gain_to_split': min_gain_to_split,
                         'drop_rate': drop_rate,
                         'max_drop': max_drop,
                         'skip_drop': skip_drop,
                         'xgboost_dart_mode': xgboost_dart_mode,
                         'uniform_drop': uniform_drop,
                         'drop_seed': drop_seed,
                         'top_rate': top_rate,
                         'other_rate': other_rate,
                         'min_data_per_group': min_data_per_group,
                         'max_cat_threshold': max_cat_threshold,
                         'cat_l2': cat_l2,
                         'cat_smooth': cat_smooth,
                         'max_cat_to_onehot': max_cat_to_onehot,
                         'top_k': top_k,
                         'monotone_constraints': monotone_constraints,
                         'monotone_constraints_method': monotone_constraints_method,
                         'monotone_penalty': monotone_penalty,
                         'feature_contrib': feature_contrib,
                         'forcedsplits_filename': forcedsplits_filename,
                         'refit_decay_rate': refit_decay_rate,
                         'path_smooth': path_smooth,
                         'interaction_constraints': interaction_constraints,
                         'verbosity': verbosity,
                        }
        
    def ranker(self):
        '''
        Get the Booster ranker.
        '''
        
        return self.__gbm
        
    def fit(self,
            X,
            y,
            boost_rounds=100,
            early_stopping=0,
            val_data=None
           ):
        '''
        Fit the boosted trees.
        
        Arguments:
        
            X:              the features,
            y:              the labels,
            boost_rounds:   number of boosting rounds,
            early_stopping: number of iterations before early stopping,
            val_data:       validation data (tuple with X and y).
        '''
        
        # create dataset for training
        train_set = lgb.Dataset(X, label=y)
        
        val_names = None
        if val_data is not None:
            val_set   = lgb.Dataset(val_data[0], label=val_data[1])
            val_names = 'validation'
            
        # adjust parameters
        if X.shape[0] > X.shape[1]:
            self.params['force_row_wise'] = True
        else:
            self.params['force_col_wise'] = True
        
        # fit the data
        start = time()
        self.__gbm = lgb.train(self.params,
                               train_set,
                               num_boost_round=boost_rounds,
                               valid_sets=val_data,
                               valid_names=val_names,
                               early_stopping_rounds=early_stopping
                              )
        stop  = time() - start
        delta = strftime("%H hours, %M minutes and %S seconds", gmtime(stop))
        
        if self.__ctx is not None:
            self.__ctx.logger().info(f'Variable ranking trained in {delta}.')
        
        return self
    
    def predict(self, X):
        '''
        Compute predictions.
        
        Arguments:
        
            X: the test features.
        '''
        
        if self.__y_pred is None:
            self.__y_pred = self.__gbm.predict(X)
        
        return self.__y_pred
    
    def evaluate(self, X, y, verbose=True):
        '''
        Evaluate the model.
        
        Arguments:
        
            X: the test features,
            y: the test labels.
        '''
        
        y_true = y
        
        # compute predictions
        shape  = y_true.shape
        y_pred = self.predict(X).reshape(shape)

        ev     = None
        metric = None
        if bool(re.match('regression', self.__obj)):
            
            metric = 'MSE'
            ev     = np.mean(np.square(np.subtract(y_true, y_pred)))
            
        else:
            
            metric = 'ACC'
            ev     = np.mean(np.equal(y_true, y_pred))
            
        if self.__ctx is not None:
            self.__ctx.logger().info(f'Variable ranking evaluated with {metric} = {ev:.3f}.')
            
        return ev
            
    def __compute_ranking(self):
        '''
        Compute the variable ranking.
        '''
        
        features = self.__gbm.feature_name()
        importance = self.__gbm.feature_importance(importance_type='split')
        
        # normalise the importance
        total      = np.sum(importance)
        importance = importance / total
        
        # create a dataframe
        self.__rank = pd.DataFrame({'features': features, 'ranking': importance})
        self.__rank = self.__rank.sort_values(by='ranking', ignore_index=True, ascending=False)
        
        return self
    
    def feature_importances(self, head=None):
        '''
        Return the list of the feature importances.
        
        Arguments:
        
            head: display the first N entries.
        '''
        
        if self.__rank is None:
            self.__compute_ranking()
        
        if head is None:
            return self.__rank
        else:
            return self.__rank.loc[:head, :]
        
    def plot_importance(self,
                        head=None,
                        xlabel='',
                        ylabel='ranking',
                        name=None,
                        figsize=(6,5),
                        save=False,
                        ctx=None,
                        **kwargs
                       ):
        '''
        Plot the feature importances.
        
        Arguments:
        
            head: plots only the first N entries,
            xlabel:   label of the x-axis,
            ylabel:   label of the y-axis,
            name:     name of the figure,
            figsize:  size of the Matplotlib figure,
            save:     save the figure,
            ctx:      Context,
            **kwargs: addition arguments for Seaborn barplot.
        '''
        
        # get the data
        data = self.feature_importances(head=head)
        
        # barplot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.tight_layout()
        
        sns.barplot(data=data,
                    x='features',
                    y='ranking',
                    color='tab:blue',
                    ax=ax,
                    **kwargs
                   )
        
        # rotate the x labels
        plt.xticks(rotation=45, ha='right', va='top')
        
        # set attributes
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        
        # save the figure
        if save and ctx is not None:
            fig.savefig(os.path.join(ctx.img(), name + '.pdf'), dpi=72)