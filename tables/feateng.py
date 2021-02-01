import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

@pd.api.extensions.register_dataframe_accessor('eda')
class EDA:
    '''
    Improve feature engineering with Pandas dataframe.
    '''
    
    def __init__(self, pandas_obj):
        
        self.__obj = pandas_obj
        
    def full_info(self, to='MB', ratio=1024):
        '''
        Return info on the dataset.
        
        Arguments:
        
            to:    convert memory usage to ['KB', 'MB', 'GB'],
            ratio: ratio between 1 KB and 1 B.
        '''
        
        # conversion
        conv = {'KB': 1, 'MB': 2, 'GB': 3}
        mem  = self.__obj.memory_usage(deep=True).sum()
        for _ in range(conv[to]):
            mem = mem / ratio
            
        # shape
        nrows, ncols = self.__obj.shape
        
        # print info
        print(f'No. of rows:    {nrows:d}')
        print(f'No. of columns: {ncols:d}')
        print(f'Index:          {type(self.__obj.index)}')
        print(f'RAM usage:      {mem:.2f} {to}.')
        
        print('\n')
        
        # print info table
        info = {'dtypes':       self.__obj.dtypes,
                'NA cases':     self.__obj.isna().sum(),
                'mean':         self.__compute_mean(),
                'std':          self.__compute_std(),
                '0% (min)':     self.__compute_quantile(0.00),
                '1%':           self.__compute_quantile(0.01),
                '10%':          self.__compute_quantile(0.10),
                '25%':          self.__compute_quantile(0.25),
                '50% (median)': self.__compute_quantile(0.50),
                '75%':          self.__compute_quantile(0.75),
                '90%':          self.__compute_quantile(0.90),
                '99%':          self.__compute_quantile(0.99),
                '100% (max)':   self.__compute_quantile(1.00)
               }
        
        return pd.DataFrame(info, index=self.__obj.columns)
        
    def __compute_mean(self):
        '''
        Compute the mean per column.
        '''
        
        mean = []
        for c in self.__obj.columns:
            
            if self.__obj[c].dtype == 'object':
                mean.append('--')
            else:
                mean.append(self.__obj[c].mean())
                
        return mean
                
    def __compute_std(self):
        '''
        Compute the standard deviation per column.
        '''
        
        std = []
        for c in self.__obj.columns:
            
            if self.__obj[c].dtype == 'object':
                std.append('--')
            else:
                std.append(self.__obj[c].std())
                
        return std
    
    def __compute_quantile(self, q):
        '''
        Compute the quantile per column.
        
        Arguments:
        
            q: the quantile (0 <= q <= 1).
        '''
        
        quantile = []
        for c in self.__obj.columns:
            
            if self.__obj[c].dtype == 'object':
                quantile.append('--')
            else:
                quantile.append(self.__obj[c].quantile(q))
                
        return quantile
    
    def get_na(self, columns, inverse=False):
        '''
        Select NA values in given columns.
        
        Arguments:
        
            columns:   str or list of columns to select NA values,
            inverse:   inverse the selection.
        '''
        
        ids = self.__obj[columns].isna().any(axis=1)
        
        # select the cases
        na_cases       = self.__obj.loc[ids, :]
        complete_cases = self.__obj.loc[~self.__obj.index.isin(na_cases.index), :]
        
        # return the choice
        if not inverse:
            return na_cases
        else:
            return complete_cases
        
    @property
    def loc(self):
        '''
        Locate by property.
        '''
        
        return self.__obj.loc
    
    @property
    def iloc(self):
        '''
        Locate by index.
        '''
        
        return self.__obj.iloc
            
    @property
    def complete_cases(self):
        '''
        Get only the complete cases.
        '''
        
        columns        = self.__obj.columns
        complete_cases = self.get_na(columns=columns, inverse=True)
        
        return complete_cases
    
    def replace_values(self, column, new_values):
        '''
        Replace the values in the column.
        '''
        
        self.__obj.loc[:, column] = new_values
        
        return self.__obj
        
    def convert_dtype(self, columns, dtype):
        '''
        Convert dtype of multiple columns.
        
        Arguments:
        
            columns: str or list of columns,
            dtype:   the new dtype to assign.
        '''
        
        self.__obj.loc[:, columns] = self.__obj.loc[:, columns].astype(dtype)
        
        return self.__obj
    
    def freq_rank(self, column, count=False):
        '''
        List the most frequent values.
        
        Arguments:
        
            column: the column to consider,
            count:  display count instead of frequency.
        '''
            
        # rank the values
        if count:
            rank = self.__obj[column].value_counts()
        else:
            rank = self.__obj[column].value_counts(normalize=True)
            
        rank = rank.reset_index().rename(columns={'index': 'values'})
        
        return rank
    
    def histplot(self,
                 series,
                 xlabel=None,
                 ylabel=None,
                 binwidth=None,
                 logscale=(False,False),
                 name=None,
                 figsize=(6,5),
                 save=False,
                 ctx=None,
                 **kwargs
                ):
        '''
        Plot univariate distributions.
        
        Arguments:
        
            series:   data series to plot,
            xlabel:   label of the x-axis,
            ylabel:   label of the y-axis,
            binwidth: width of the binning,
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
        if not isinstance(series, list):
            series = [series]
            
        for n, s in enumerate(series):
            sns.histplot(data=self.__obj,
                         x=s,
                         binwidth=binwidth,
                         color=sns.color_palette()[n],
                         log_scale=logscale,
                         ax=ax,
                         **kwargs
                        )
        
        # set attributes
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        
        # save the figure
        if save and ctx is not None:
            fig.savefig(os.path.join(ctx.img(), name + '.pdf'), dpi=72)
            
    def barplot(self,
                x,
                y,
                xlabel='',
                ylabel=None,
                logscale=(False,False),
                rotate=None,
                name=None,
                figsize=(6,5),
                save=False,
                ctx=None,
                **kwargs
               ):
        '''
        Plot categorical data in bars.
        
        Arguments:
        
            x:        x-axis data,
            y:        y-axis data,
            xlabel:   label of the x-axis,
            ylabel:   label of the y-axis,
            logscale: tuple containing whether to use log scale on the axes,
            rotate:   rotate the labels on the x-axis (0 < rotate <= 90),
            name:     name of the figure,
            figsize:  size of the Matplotlib figure,
            save:     save the figure,
            ctx:      Context,
            **kwargs: addition arguments for Seaborn barplot.
        '''
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.tight_layout()
        
        # barplot
        if not isinstance(y, list):
            y = [y]
            
        for n, s in enumerate(y):
            sns.barplot(data=self.__obj,
                        x=x,
                        y=s,
                        color=sns.color_palette()[n],
                        order=self.__obj[x],
                        ax=ax,
                        **kwargs
                       )
        
        # set attributes
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if logscale[0]:
            ax.set_xscale('log')
        if logscale[1]:
            ax.set_yscale('log')
        if rotate is not None:
            plt.xticks(rotation=rotate, ha='right', va='top')
        
        # save the figure
        if save and ctx is not None:
            fig.savefig(os.path.join(ctx.img(), name + '.pdf'), dpi=72)
            
    def train_test_split(self,
                         train,
                         test=None,
                         validation=None,
                         stratified=None,
                         to_dict=False,
                         random_state=None,
                         ctx=None,
                         verbose=True
                        ):
        '''
        Split into training and test sets (and validation, if needed).
        
        Arguments:
        
            train:        train fraction of the data to use,
            test:         test fraction of the data to use,
            validation:   validation fraction of the data to use,
            stratified:   perform stratification on label,
            to_dict:      convert to dictionary before returning ({column -> values}),
            random_state: the random state,
            ctx:          Context,
            verbose:      verbose output.
            
        Returns:
        
            train set, val. set, test set
        '''
        
        train_set, val_set, test_set = None, None, None
        
        # distinguish the stratified case
        if stratified is None:
            
            # select the training set
            train_set = self.__obj.sample(frac=train, random_state=random_state)
            oos       = self.__obj.loc[~self.__obj.index.isin(train_set.index), :]

        else:
            
            obj     = self.__obj.groupby(by=stratified, group_keys=False)
            
            # select training set
            train_set = obj.apply(lambda x: x.sample(frac=train, random_state=random_state))
            oos       = self.__obj.loc[~self.__obj.index.isin(train_set.index), :]
            
            
        # select the test set
        if test is not None:
            test_set = oos.sample(frac=round(test / (1.0 - train), 2), random_state=random_state)
            oos     = oos.loc[~oos.index.isin(test_set.index), :]
        else:
            test_set = oos

        # select the validation set
        if validation is not None:
            
            # distinguish the stratified case
            if stratified is None:
                
                val_set = oos.sample(frac=round(validation / (1.0 - train - test), 2), random_state=random_state)
                
            else:
                
                obj     = oos.groupby(by=stratified, group_keys=False)
                val_set = obj.apply(lambda x: x.sample(frac=round(validation / (1.0 - train - test), 2), random_state=random_state))
            
        # verbose output
        if verbose:
            print(f'Training set:   {train_set.shape[0]:d} rows ({100 * train_set.shape[0] / self.__obj.shape[0]:.1f}% ratio)')
            print(f'Test set:        {test_set.shape[0]:d} rows ({100 * test_set.shape[0] / self.__obj.shape[0]:.1f}% ratio)')
                
            if val_set is not None:
                print(f'Validation set: {val_set.shape[0]:d} rows ({100 * val_set.shape[0] / self.__obj.shape[0]:.1f}% ratio)')
                
        # save sets if Context is provided
        if ctx is not None:
            
            train_set.to_json(os.path.join(ctx.dat(), ctx.log_sess() + '_train_set.json.gz'), orient='index')
            test_set.to_json(os.path.join(ctx.dat(), ctx.log_sess() + '_test_set.json.gz'), orient='index')
            
            if val_set is not None:
                val_set.to_json(os.path.join(ctx.dat(), ctx.log_sess() + '_val_set.json.gz'), orient='index')
                
            ctx.logger().info(f'Datasets have been saved to file in {ctx.dat()}.')
            
        if to_dict:
            
            if isinstance(train_set, pd.DataFrame):
                train_set = {c: np.asarray(list(train_set.loc[:, c].values)) for c in train_set.columns}
            else:
                train_set = {train_set.name: np.asarray(list(train_set.values))}
            
            if val_set is not None:
                
                if isinstance(val_set, pd.DataFrame):
                    val_set = {c: np.asarray(list(val_set.loc[:, c].values)) for c in val_set.columns}
                else:
                    val_set = {val_set.name: np.asarray(list(val_set.values))}
            
            if isinstance(test_set, pd.DataFrame):
                test_set = {c: np.asarray(list(test_set.loc[:, c].values)) for c in test_set.columns}
            else:
                test_set = {test_set.name: np.asarray(list(test_set.values))}
            
        return train_set, val_set, test_set
    
    def feat_lab(self, features, labels, to_dict=False):
        '''
        Divide the set into features and labels.
        
        Arguments:
        
            features: the columns to use as features,
            labels:   the columns to use as labels,
            to_dict:  convert to dictionary before returning.
            
        Returns:
        
            features, labels
        '''
        
        X = self.__obj.loc[:, features]
        y = self.__obj.loc[:, labels]
        
        if to_dict:
            
            if isinstance(X, pd.DataFrame):
                X = {c: np.asarray(list(X.loc[:, c].values)) for c in X.columns}
            else:
                X = {X.name: np.asarray(list(X.values))}
                
            if isinstance(y, pd.DataFrame):
                y = {c: np.asarray(list(y.loc[:, c].values)) for c in y.columns}
            else:
                y = {y.name: np.asarray(list(y.values))}
            
        return X, y
    
    def standardise(self, mean=None, std=None, return_mean_std=False):
        '''
        Standardise the set.
        
        Arguments:
        
            mean: the mean to centre the data,
            std:  the standard deviation to normalise the dispersion.
            
        Return:
        
            the dataframe or ((mean, std), the dataframe)
        '''
        
        if mean is None:
            mean = self.__obj.mean()
        if std is None:
            std  = self.__obj.std()
        
        if return_mean_std:
            return (mean, std), (self.__obj - mean) / std
        else:
            return (self.__obj - mean) / std
            
    def heatmap(self,
                vmin=None,
                vmax=None,
                centre=None,
                cmap='RdBu_r',
                name=None,
                figsize=(6,5),
                save=False,
                ctx=None,
                **kwargs
               ):
        '''
        Plot a heatmap of a data matrix.
        
        Arguments:
        
            vmin:   the minimum value,
            vmax:   the maximum value,
            center: the centre value,
            cmap:   the colour map,
            name:     name of the figure,
            figsize:  size of the Matplotlib figure,
            save:     save the figure,
            ctx:      Context,
            **kwargs: addition arguments for Seaborn heatmap.
        '''
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.tight_layout()

        sns.heatmap(self.__obj,
                    vmin=vmin,
                    vmax=vmax,
                    center=centre,
                    cmap='RdBu_r',
                    ax=ax
                   )
                               
        # save the figure
        if save and ctx is not None:
            fig.savefig(os.path.join(ctx.img(), name + '.pdf'), dpi=72)
            
    def pad(self, column, shape):
        '''
        Pad a given column into a specific size.
        
        Arguments:
        
            column: the column to pad,
            size:   the target size.
        '''
        
        new_values = self.__obj.loc[:, column].apply(lambda x: np.pad(x, ((0, shape[0] - np.shape(x)[0]), (0, shape[1] - np.shape(x)[1]))))
        
        return self.replace_values(column, new_values)
    
    def to_dict(self):
        '''
        Transform the dataset into a dictionary {column -> values}.
        '''
        
        return {c: self.__obj.loc[:, c] for c in self.__obj.columns}
    
    def explode(self, column):
        '''
        Transform a column (made of lists) into a new dataframe.
        
        Arguments:
        
            column: the column to consider.
        '''
        
        lists = self.__obj.loc[:, column].apply(lambda x: list(np.reshape(x, (-1,)))).tolist()
        cols  = [column + '_' + str(n+1) for n in range(np.shape(lists)[-1])]
        
        return pd.DataFrame(lists, columns=cols)
