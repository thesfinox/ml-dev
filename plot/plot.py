import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Plot:
    '''
    Plot data.
    '''
    
    def __init__(self,
                 data,
                 xlabel='',
                 ylabel=None,
                 logscale=(False,False),
                 rotate=None,
                 title=None,
                 name=None,
                 figsize=(6,5),
                 save=False,
                 ctx=None,
                ):
        '''
        Arguments:
        
            data:     the dataset,
            xlabel:   label of the x-axis,
            ylabel:   label of the y-axis,
            logscale: tuple containing whether to use log scale on the axes,
            rotate:   rotate the labels on the x-axis (0 < rotate <= 90),
            title:    the plot title,
            name:     name of the figure,
            figsize:  size of the Matplotlib figure,
            save:     save the figure,
            ctx:      Context.
        '''
        
        self.__data     = data
        self.__xlabel   = xlabel
        self.__ylabel   = ylabel
        self.__logscale = logscale
        self.__rotate   = rotate
        self.__title    = title
        self.__name     = name
        self.__figsize  = figsize
        self.__save     = save
        self.__ctx      = ctx
        
    def lineplot(self, y, x=None, series=False, hue=None, **kwargs):
        '''
        Draw a lineplot.
        
        Arguments:
        
            x:        the name of the x-axis series,
            y:        the name of the y-axis series,
            series:   plot as a "time" series with arbitrary x-axis,
            hue:      the series to distinguish colours,
            **kwargs: additional arguments to pass to Seaborn barplot.
        '''
        
        fig, ax = plt.subplots(1, 1, figsize=self.__figsize)
        fig.tight_layout()
        
        if series:
            if isinstance(self.__data[y], list):
                N = len(self.__data[y])
            else:
                N = self.__data[y].shape[0]
                
            x = np.arange(N)
        
        sns.lineplot(data=self.__data,
                     x=x,
                     y=y,
                     hue=hue,
                     ax=ax,
                     **kwargs
                    )
        
        # set parameters
        if self.__title is not None:
            ax.set_title(self.__title)
        if self.__xlabel is not None:
            ax.set_xlabel(self.__xlabel)
        if self.__ylabel is not None:
            ax.set_ylabel(self.__ylabel)
        if self.__logscale[0]:
            ax.set_xscale('log')
        if self.__logscale[1]:
            ax.set_yscale('log')
        if self.__rotate is not None:
            plt.xticks(rotation=self.__rotate, ha='right', va='top')
        
        # save the figure
        if self.__save and self.__ctx is not None:
            fig.savefig(os.path.join(self.__ctx.img(), self.__name + '_lineplot.pdf'), dpi=72)
            
    def barplot(self, x, y, hue=None, **kwargs):
        '''
        Draw a barplot.
        
        Arguments:
        
            x:        the name of the x-axis series,
            y:        the name of the y-axis series,
            hue:      the series to distinguish colours,
            **kwargs: additional arguments to pass to Seaborn barplot.
        '''
        
        fig, ax = plt.subplots(1, 1, figsize=self.__figsize)
        fig.tight_layout()
        
        sns.barplot(data=self.__data,
                    x=x,
                    y=y,
                    hue=hue,
                    ax=ax,
                    **kwargs
                   )
        
        # set parameters
        if self.__title is not None:
            ax.set_title(self.__title)
        if self.__xlabel is not None:
            ax.set_xlabel(self.__xlabel)
        if self.__ylabel is not None:
            ax.set_ylabel(self.__ylabel)
        if self.__logscale[0]:
            ax.set_xscale('log')
        if self.__logscale[1]:
            ax.set_yscale('log')
        if self.__rotate is not None:
            plt.xticks(rotation=self.__rotate, ha='right', va='top')
        
        # save the figure
        if self.__save and self.__ctx is not None:
            fig.savefig(os.path.join(self.__ctx.img(), self.__name + '_barplot.pdf'), dpi=72)