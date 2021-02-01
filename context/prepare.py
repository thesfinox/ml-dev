import os
import sys
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from datetime import datetime

class Context:
    '''
    Define the environment in the notebook/script.
    '''
    
    def __init__(self,
                 context='notebook',
                 style='darkgrid',
                 palette='tab10',
                 color_codes=True,
                 gpu_ready=False,
                 wd='.',
                 img_dir=None,
                 mod_dir=None,
                 dat_dir=None,
                 log_dir=None,
                 tensorboard=None,
                 subdir=True,
                 session='context'
                ):
        '''
        Arguments:
        
            context:     Matplotlib context,
            style:       Seaborn style,
            palette:     Seaborn palette name,
            color_codes: translate "r", "b", "g", etc. to the palette colours,
            gpu_ready:   setup GPU for Tensorflow,
            wd:          working directory,
            img_dir:     create directory for images (if not None),
            mod_dir:     create directory for models (if not None),
            dat_dir:     create directory for data (if not None),
            log_dir:     create directory for logs (if not None),
            tensorboard: create directory for tensorboard (if not None),
            subdir:      for each directory, create a subdir to identify the run,
            session:     name of the logging session.
        '''
        
        self.__context     = context
        self.__style       = style
        self.__palette     = palette
        self.__color_codes = color_codes
        self.__gpu_ready   = gpu_ready
        self.__wd          = wd
        self.__img_dir     = img_dir
        self.__mod_dir     = mod_dir
        self.__dat_dir     = dat_dir
        self.__log_dir     = log_dir
        self.__tensorboard = tensorboard
        self.__subdir      = subdir
        self.__sub_name    = None
        self.__session     = session
        self.__logger      = None
        
        # add subdirectory
        if self.__subdir:
            
            self.__sub_name = datetime.now().strftime(self.__session + '_%H%M%S_%Y%m%d')
        
        # check the paths
        if self.__img_dir is not None:
            self.__img_dir = self.__check_path(self.__img_dir, self.__wd)
            
        if self.__mod_dir is not None:
            self.__mod_dir = self.__check_path(self.__mod_dir, self.__wd)
            
        if self.__dat_dir is not None:
            self.__dat_dir = self.__check_path(self.__dat_dir, self.__wd)
            
        if self.__log_dir is not None:
            self.__log_dir = self.__check_path(self.__log_dir, self.__wd)
            
        if self.__tensorboard is not None:
            self.__tensorboard = self.__check_path(self.__tensorboard, self.__wd)
        
        # set the context
        self.__set_context(self.__context, self.__style, self.__palette, self.__color_codes)
        
        # create working directories            
        if self.__img_dir is not None:
            self.__create_dir(self.__img_dir, sub_name=self.__sub_name)
            
        if self.__mod_dir is not None:
            self.__create_dir(self.__mod_dir, sub_name=self.__sub_name)
            
        if self.__dat_dir is not None:
            self.__create_dir(self.__dat_dir)
            
        if self.__log_dir is not None:
            self.__create_dir(self.__log_dir)
            self.__logging(self.__log_dir, self.__session)
            
        if self.__tensorboard is not None:
            self.__create_dir(self.__tensorboard, sub_name=self.__sub_name)
            
        # check if GPU ready
        if self.__gpu_ready:
            
            gpus = tf.config.experimental.list_physical_devices('GPU')
            
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print('GPU is ready!')
                except RuntimeError as e:
                    sys.stderr.write(e)
            
    def __check_path(self, directory, wd):
        '''
        Check if the path is in the working directory.
        
        Arguments:
        
            directory: the directory to check,
            wd:        the working directory.
        '''
        
        p = directory.split(os.path.sep)
        
        if p[0] != wd:
            directory = os.path.join(wd, directory)
            
        return directory
        
    def __set_context(self, context, style, palette, color_codes):
        '''
        Set the context.
        
        Arguments:
        
            context:     Matplotlib context,
            style:       Seaborn style,
            palette:     Seaborn palette name,
            color_codes: translate "r", "b", "g", etc. to the palette colours,
        '''
        
        sns.set_theme(context=context, style=style, palette=palette, color_codes=color_codes)
        
        return self
    
    def __create_dir(self, directory, sub_name=None):
        '''
        Create directories.
        
        Arguments:
        
            directory: name of the directory,
            sub_name:  name of the subdirectory.
        '''
        
        if sub_name is not None:
            
            directory = os.path.join(directory, sub_name)
        
        os.makedirs(directory, exist_ok=True)
        
        return self
        
        
    def __logging(self, log_dir, session):
        '''
        Setup logging.
        
        Arguments:
        
            log_dir: log directory,
            session: name of the logging session.
        '''
        
        # set logger
        self.__logger = logging.getLogger(session)
        self.__logger.setLevel(logging.DEBUG)
        
        # file handler
        n = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M%S") + '_' + session + '.log')
        f = logging.FileHandler(n)
        f.setLevel(logging.DEBUG)
        
        # formatter
        form = logging.Formatter('%(asctime)s | %(levelname)s: %(message)s')
        f.setFormatter(form)
        
        # add handler
        self.__logger.addHandler(f)
        
        # signal creation of the log
        self.__logger.info('Log file created.')
        
        return self
        
    def pwd(self):
        '''
        Prints the working directory.
        '''
        
        return self.__wd
    
    def subdir(self):
        '''
        Prints the name of the subdir.
        '''
        
        return self.__sub_name
    
    def img(self):
        '''
        Prints the image directory.
        '''
        
        if self.__subdir:
            
            return os.path.join(self.__img_dir, self.__sub_name)
        
        else:
            
            return self.__img_dir
    
    def mod(self):
        '''
        Prints the model directory.
        '''
        
        if self.__subdir:
            
            return os.path.join(self.__mod_dir, self.__sub_name)
        
        else:
            
            return self.__mod_dir
    
    def dat(self):
        '''
        Prints the data directory.
        '''
        
        return self.__dat_dir
    
    def log(self):
        '''
        Prints the log directory.
        '''
        
        return self.__log_dir
    
    def tboard(self):
        '''
        Prints the tensorboard directory.
        '''
        
        if self.__subdir:
            
            return os.path.join(self.__tensorboard, self.__sub_name)
        
        else:
            
            return self.__tensorboard
    
    def log_sess(self):
        '''
        Returns the name of the logging session.
        '''
        
        return self.__session
    
    def logger(self):
        '''
        Returns the logger object.
        '''
        
        return self.__logger