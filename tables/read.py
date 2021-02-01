import os
import sys
import pandas as pd
import numpy as np
import urllib

class Table:
    '''
    Read a table from file or URL using Pandas.
    '''
    
    def __init__(self,
                 table,
                 save=False,
                 loc=None,
                 log=None,
                 ctx=None
                ):
        '''
        Arguments:
        
            table: path or URL to the table,
            save:  save the file instead of reading remotely,
            loc:   saving location,
            log:   logger,
            ctx:   Context (if provided will override loc and log).
        '''
        
        self.__table  = table
        self.__remote = False
        self.__save   = save
        self.__loc    = loc
        self.__log    = log
        self.__df     = None
        self.__format = None
        self.__ctx    = ctx
        
        # check if context
        if self.__ctx is not None:
            self.__loc = self.__ctx.dat()
            self.__log = self.__ctx.logger()
        
        # check if path or URL
        if urllib.parse.urlparse(self.__table).scheme in ['http', 'https', 'ftp', 'sftp']:
            self.__remote = True
            
        # save the file
        if self.__save and self.__remote:
            self.__download()
            
    def __download(self):
        '''
        Download the table.
        '''
        # check the path
        if self.__loc.split(os.path.sep)[-1] != self.__table.split('/')[-1]:
            self.__loc = os.path.join(self.__loc, self.__table.split('/')[-1])
            
        urllib.request.urlretrieve(self.__table, self.__loc)
        
        if self.__log is not None:
            self.__log.info(f'Table downloaded in {self.__loc}.')
        
        return self
    
    def read(self, **kwargs):
        '''
        Read the table.
        
        Arguments:
        
            **kwargs: additional arguments to pass the Pandas readers.
        '''
        
        if self.__save and self.__remote:
            obj = self.__loc
        else:
            obj = self.__table
            
        # check the path
        if 'json' in obj.lower().split('.') and self.__df is None:
            
            self.__format = 'json'
            self.__df     = pd.read_json(obj, **kwargs)
            if self.__log is not None:
                self.__log.info(f'Table read from {self.__format} format.')
            
        elif 'csv' in obj.lower().split('.') and self.__df is None:
            
            self.__format = 'csv'
            self.__df     = pd.read_csv(obj, **kwargs)
            if self.__log is not None:
                self.__log.info(f'Table read from {self.__format} format.')
            
        elif ('h5' in obj.lower().split('.') or 'hdf' in obj.lower().split('.')) and self.__df is None:
            
            self.__format = 'hdf'
            self.__df     = pd.read_hdf(obj, **kwargs)
            if self.__log is not None:
                self.__log.info(f'Table read from {self.__format} format.')
            
        elif self.__df is not None:
            
            if self.__log is not None:
                self.__log.debug('Data already read from source.')
            
        else:
            
            sys.stderr.write('Method not yet implemented or format not supported.')
            if self.__log is not None:
                self.__log.error('Unable to open file. Format not supported or method not implemented.')
                
        return self
    
    def get_df(self):
        '''
        Return the dataset.
        '''
        
        return self.__df
    
    def data(self, verbose=True):
        '''
        Return the data and shape.
        '''
        
        if verbose:
            print(f'No. of rows:    {self.__df.shape[0]:d}.')
            print(f'No. of columns: {self.__df.shape[1]:d}.')
        
        return self.__df, self.__df.shape
    
    def update(self, df):
        '''
        Update the dataframe (usually before saving it to file).
        
        Arguments:
        
            df:      the dataframe.
        '''
        
        self.__df = df
        
        return self
    
    def write(self, name, **kwargs):
        '''
        Write the table to file.
        
        Arguments:
        
            name:     the name of the table table (with extension) into the Context,
            **kwargs: additional arguments to save the table.
        '''
        
        # form the path
        path = os.path.join(self.__ctx.dat(), name)
        
        # check format
        if 'json' in path.lower().split('.'):
            
            self.__df.to_json(path, **kwargs)
            if self.__log is not None:
                self.__log.info(f'Table saved to {path}.')
            
        elif 'csv' in path.lower().split('.'):
            
            self.__df.to_csv(path, **kwargs)
            if self.__log is not None:
                self.__log.info(f'Table saved to {path}.')
            
        elif 'h5' in path.lower().split('.') or 'hdf' in path.lower().split('.'):
            
            self.__df.to_hdf(path, **kwargs)
            if self.__log is not None:
                self.__log.info(f'Table saved to {path}.')
            
        else:
            
            sys.stderr.write('Method not yet implemented or format not supported.')
            if self.__log is not None:
                self.__log.error('Unable to open file. Format not supported or method not implemented.')