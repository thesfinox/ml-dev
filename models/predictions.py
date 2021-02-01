import json
import os
import joblib
import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Predictions:
    '''
    Compute predictions from a model.
    '''
    
    def __init__(self,
                 model,
                 keras=True,
                 train_data=(None, None),
                 val_data=(None, None),
                 test_data=(None, None),
                 metric='mse',
                 func=None
                ):
        '''
        Arguments:
        
            model:      the path to the saved model or the model itself,
            keras:      if True loads the model with Keras, otherwise use joblib,
            train_data: tuple containing (X_train, y_train),
            val_data:   tuple containing (X_val, y_val),
            test_data:  tuple containing (X_test, y_test),
            metric:     one of ['mse', 'acc'],
            func:       function to apply to the predictions before computing the metric.
        '''
        
        self.model                          = model
        self.__keras                        = keras
        self.__metric                       = metric
        self.__func                         = func
        self.__X                            = {'training':   train_data[0],
                                               'validation': val_data[0],
                                               'test':       test_data[0]
                                              }
        self.__y                            = {'training':   train_data[1],
                                               'validation': val_data[1],
                                               'test':       test_data[1]
                                              }
        self.__predictions                  = {'training':   None,
                                               'validation': None,
                                               'test':       None
                                              }
        self.__residuals                    = {'training':   None,
                                               'validation': None,
                                               'test':       None
                                              }
        self.__metrics                      = {'training':   None,
                                               'validation': None,
                                               'test':       None
                                              }
        
        # check model
        if isinstance(self.model, str):
            
            if self.__keras:
                self.model = tf.keras.models.load_model(self.model)
                
            else:
                self.model = joblib.load(self.model)
            
    def predictions(self, split='test', force=False):
        '''
        Compute the predictions.
        
        Arguments:
        
            split: one of ['training', 'validation', 'test'],
            force: force computation, avoid using cache.
        '''
        
        assert split in ['training', 'validation', 'test'], 'Split must be in ["training", "validation", "test"]!'
        
        if self.__predictions[split] is None or force:
            self.__predictions[split] = self.model.predict(self.__X[split])
            
        if isinstance(self.__predictions[split], dict):
            
            if self.__func is not None:
                self.__predictions[split] = {key: self.__func(values).squeeze() for key, values in self.__predictions[split].items()}
            else:
                self.__predictions[split] = {key: values.squeeze() for key, values in self.__predictions[split].items()}
                
        if isinstance(self.__predictions[split], np.ndarray):
            
            if self.__func is not None:
                self.__predictions[split] = self.__func(self.__predictions[split]).squeeze()
            else:
                self.__predictions[split] = self.__predictions[split].squeeze()
                
        if isinstance(self.__predictions[split], pd.Series):
            
            if self.__func is not None:
                self.__predictions[split] = self.__predictions[split].apply(self.__func).values.squeeze()
            else:
                self.__predictions[split] = self.__predictions[split].values.squeeze()
                
        return self.__predictions[split]
        
    def residuals(self, split='test', force=False):
        '''
        Compute the residuals.
        
        Arguments:
        
            split: one of ['training', 'validation', 'test'],
            force: force computation, avoid using cache.
        '''
        
        y_pred = self.predictions(split=split)
        res    = None

        if isinstance(y_pred, dict):
            res = {key: self.__y[split][key].reshape(-1,) - values.reshape(-1,) for key, values in y_pred.items()}
        else:
            res = self.__y[split] - y_pred
            
        if isinstance(res, np.ndarray):
            return res
        else:
            return res.values
            
    def reshist(self,
                output=None,
                splits=['training', 'validation', 'test'],
                xlabel='residuals',
                ylabel='count',
                binwidth=1,
                alpha=0.35,
                title=None,
                logscale=(False,False),
                name=None,
                figsize=(6,5),
                save=False,
                ctx=None,
                **kwargs
               ):
        '''
        Plot a histogram of the residuals.
        
        Arguments:
        
            output:   in multi-output models, specify the name of the output,
            splits:   a list with the splits to consider,
            xlabel:   label of the x-axis,
            ylabel:   label of the y-axis,
            binwidth: width of the binning,
            alpha:    transparency factor,
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
        
        # get the data
        data    = {}
        colours = {}
        labels  = {}
        if 'training' in splits:
            data['training']    = self.residuals(split='training')
            colours['training'] = 'tab:blue'
            labels['training']  = 'training'
            
        if 'validation' in splits:
            data['validation']    = self.residuals(split='validation')
            colours['validation'] = 'tab:red'
            labels['validation']  = 'validation'
            
        if 'test' in splits:
            data['test']    = self.residuals(split='test')
            colours['test'] = 'tab:green'
            labels['test']  = 'test'
            
        
        # histplot
        for label, values in data.items():
            
            if output is not None:
                values = values[output]
            
            sns.histplot(values.reshape(-1,),
                         binwidth=binwidth,
                         alpha=alpha,
                         color=colours[label],
                         label=labels[label],
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
        if len(list(labels.keys())) > 1:
            ax.legend()
        
        # save the figure
        if save:
            fig.savefig(os.path.join(ctx.img(), name + '.pdf'), dpi=72)
            
    def resplot(self,
                output=None,
                splits=['training', 'validation', 'test'],
                xlabel='predictions',
                ylabel='residuals',
                alpha=0.35,
                title=None,
                logscale=(False,False),
                name=None,
                figsize=(6,5),
                save=False,
                ctx=None,
                **kwargs
               ):
        '''
        Plot a scatter plot of the residuals.
        
        Arguments:
        
            output:   in multi-output models, specify the name of the output,
            splits:   a list with the splits to consider,
            xlabel:   label of the x-axis,
            ylabel:   label of the y-axis,
            alpha:    transparency factor,
            title:    the plot title,
            logscale: tuple containing whether to use log scale on the axes,
            name:     name of the figure,
            figsize:  size of the Matplotlib figure,
            save:     save the figure,
            ctx:      Context,
            **kwargs: addition arguments for Seaborn scatterplot.
        '''
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.tight_layout()
        
        # get the data
        predictions = {}
        residuals   = {}
        colours     = {}
        labels      = {}
        if 'training' in splits:
            predictions['training'] = self.predictions(split='training')
            residuals['training']   = self.residuals(split='training')
            colours['training']     = 'tab:blue'
            labels['training']      = 'training'
            
        if 'validation' in splits:
            predictions['validation'] = self.predictions(split='validation')
            residuals['validation']   = self.residuals(split='validation')
            colours['validation']     = 'tab:red'
            labels['validation']      = 'validation'
            
        if 'test' in splits:
            predictions['test'] = self.predictions(split='test')
            residuals['test']   = self.residuals(split='test')
            colours['test']     = 'tab:green'
            labels['test']      = 'test'
            
        
        # histplot
        for label in predictions.keys():
            
            preds = predictions[label]
            res   = residuals[label]
            
            if output is not None:
                preds = predictions[label][output]
                res   = residuals[label][output]
                
            sns.scatterplot(x=preds,
                            y=res,
                            alpha=alpha,
                            color=colours[label],
                            label=labels[label],
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
        if len(list(labels.keys())) > 1:
            ax.legend()
        if logscale[0]:
            ax.set_xscale('log')
        if logscale[1]:
            ax.set_yscale('log')
        
        # save the figure
        if save:
            fig.savefig(os.path.join(ctx.img(), name + '.pdf'), dpi=72)
            
    def metrics(self, output=None, split='test', verbose=False, ctx=None):
        '''
        Compute the metrics.
        
        Arguments:
        
            output:  in multi-output models, specify the output,
            split:   one of ['training', 'validation', 'test'],
            verbose: verbose output,
            ctx:     Context.
        '''
        
        true = self.__y[split]
        pred = self.predictions(split=split)
        
        if output is not None:
            true = true[output]
            pred = pred[output]
        
        metric = None
        name   = None
        
        if self.__metric == 'mse':
            
            name   = 'mean squared error'
            metric = np.mean(np.square(np.substract(true, pred)))
            
        if self.__metric == 'acc':
            
            name   = 'accuracy'
            metric = np.mean(np.equal(true, pred))
        
        if verbose:
            
            if output is not None:
                
                string = f'{output} | {split.upper()} | {name} = {metric:.3f}.'
                
            else:
                
                string = f'{split.upper()} | {name} = {metric:.3f}.'
                
            print(string)
            
        if ctx is not None:
            
            ctx.logger().info(string)
            
        # save the metric
        if output is not None:
            
            if self.__metrics[split] is None:
                self.__metrics[split] = {}
                
            self.__metrics[split][output] = metric
            
        else:
            
            self.__metrics[split] = metric
            
        return metric
    
    def get_metrics(self, to_json=False, to_file=None):
        '''
        Get the metrics.
        
        Arguments:
        
            to_json: return a JSON string,
            to_file: print to JSON file.
        '''
        
        if to_file:
            with open(to_file, 'w') as f:
                json.dump(self.__metrics, f)
                
        if to_json:
            return json.dumps(self.__metrics)
        
        return self.__metrics

    def get_predictions(self, to_json=False, to_file=None):
        '''
        Get the predictions.
        
        Arguments:
        
            to_json: return a JSON string,
            to_file: print to JSON file.
        '''
        
        if to_file:
            with open(to_file, 'w') as f:
                json.dump(list(self.__predictions), f)
                
        if to_json:
            return json.dumps(list(self.__predictions))
        
        return self.__predictions
    
    def get_residuals(self, to_json=False, to_file=None):
        '''
        Get the residuals.
        
        Arguments:
        
            to_json: return a JSON string,
            to_file: print to JSON file.
        '''
        
        if to_file:
            with open(to_file, 'w') as f:
                json.dump(list(self.__residuals), f)
                
        if to_json:
            return json.dumps(list(self.__residuals))
        
        return self.__residuals