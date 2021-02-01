import os
import seaborn as sns
import matplotlib.pyplot as plt

class TrainTestPlot:
    '''
    Plot figures in the train:test:val splits.
    '''
    
    def __init__(self,
                 train,
                 validation=None,
                 test=None,
                 colours=None,
                 dashes=None,
                 labels=None
                ):
        '''
        Arguments:
        
            train:      training data,
            validation: validation data,
            test:       test data,
            colours:    list of 3 colours to use,
            dashes:     list of 3 dash codes to use,
            labels:     list of 3 labels to use.
        '''
        
        self.__train      = train
        self.__validation = validation
        self.__test       = test
        
        # combine the data
        self.data      = {}
        self.__dashes  = {}
        self.__colours = {}
        self.__labels  = {}
        
        # training data
        self.data['training'] = self.__train
        
        if dashes is not None:
            self.__dashes['training'] = dashes[0]
        else:
            self.__dashes['training'] = 'solid'
            
        if colours is not None:
            self.__colours['training'] = colours[0]
        else:
            self.__colours['training'] = 'tab:blue'
            
        if labels is not None:
            self.__labels['training'] = labels[0]
        else:
            self.__labels['training'] = 'training'
            
        # validation data
        if self.__validation is not None:
            
            self.data['validation'] = self.__validation
            
            if dashes is not None:
                self.__dashes['validation'] = dashes[1]
            else:
                self.__dashes['validation'] = 'dashed'

            if colours is not None:
                self.__colours['validation'] = colours[1]
            else:
                self.__colours['validation'] = 'tab:red'

            if labels is not None:
                self.__labels['validation'] = labels[1]
            else:
                self.__labels['validation'] = 'validation'
                
        # test data        
        if self.__test is not None:
            
            self.data['test'] = self.__test

            if dashes is not None:
                self.__dashes['test'] = dashes[2]
            else:
                self.__dashes['test'] = 'dashdot'

            if colours is not None:
                self.__colours['test'] = colours[2]
            else:
                self.__colours['test'] = 'tab:green'

            if labels is not None:
                self.__labels['test'] = labels[2]
            else:
                self.__labels['test'] = 'test'
                
    def lineplot(self,
                 xlabel='epochs',
                 ylabel=None,
                 title=None,
                 logscale=(False,False),
                 name=None,
                 figsize=(6,5),
                 save=False,
                 ctx=None,
                 **kwargs
                ):
        '''
        Plot a line plot of the training data.
        
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
            **kwargs: addition arguments for Seaborn lineplot.
        '''
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.tight_layout()
        
        for label, values in self.data.items():
            sns.lineplot(data=values,
                         linestyle=self.__dashes[label],
                         color=self.__colours[label],
                         label=self.__labels[label],
                         ax=ax
                        )
        # set attributes
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if logscale[0]:
            ax.set_xscale('log')
        if logscale[1]:
            ax.set_yscale('log')
        if len(list(self.__labels.keys())) > 1:
            ax.legend()
        
        # save the figure
        if save:
            fig.savefig(os.path.join(ctx.img(), name + '.pdf'), dpi=72)
                
    def histplot(self,
                 xlabel=None,
                 ylabel=None,
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
        Plot a histogram of the data.
        
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
        for label, values in self.data.items():
            sns.histplot(values,
                         binwidth=binwidth,
                         color=self.__colours[label],
                         label=self.__labels[label],
                         log_scale=logscale,
                         ax=ax,
                         **kwargs
                        )
        
        # set attributes
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if len(list(self.__labels.keys())) > 1:
            ax.legend()
        
        # save the figure
        if save:
            fig.savefig(os.path.join(ctx.img(), name + '.pdf'), dpi=72)
            
