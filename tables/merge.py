import pandas as pd

class DFMerge:
    '''
    Merge dataframes.
    '''
    
    def __init__(self, dfs=[]):
        '''
        Arguments:
            
            dfs: the list of dataframes.
        '''
        
        self.__dfs = dfs
        
    def append(self, df):
        '''
        Append a dataframe to the list.
        
        Arguments:
        
            df: the dataframe to add.
        '''
        
        self.__dfs.append(df)
        
        return self
        
    def merge(self, on, suffixes):
        '''
        Merge a list of datasets on a column.
        
        Arguments:
        
            on:       the column on which to merge,
            suffixes: list of ordered suffixes (must be same length as the dataframes).
        '''
        
        # sanitise
        assert len(suffixes) == len(self.__dfs), 'Lists do not have the same length!'
        
        # merge the first two
        buff = pd.merge(self.__dfs[0], self.__dfs[1], on=on, suffixes=(suffixes[0], suffixes[1]))
        
        # keep merging
        for n in range(2, len(self.__dfs)):
            
            buff = pd.merge(buff, self.__dfs[n], on=on, suffixes=(suffixes[n-1], suffixes[n]))
            
        return buff