
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np


def plots(series, save_path=None, title=None):
    title = title.replace('.',',')
    
    n_lines = int(np.ceil(len(series)/2))
    fig, ax = plt.subplots(n_lines,2,figsize=(16,12))
    
    if title is not None:
        fig.suptitle(title, fontsize=16)

    for i, (key, value) in enumerate(series.items()):
        legend = False
        if n_lines==1:
            idx = i%2
        else:
            idx = (i//2,i%2)
        for v in value:
            if isinstance(v, dict):
                x = v['x']
                y = v['y']
                ax[idx].plot(x,y,**{k:v[k] for k in v.keys() if k!='x' and k!='y'}, lw=1, ms=9, mec='black',)
                if 'label' in v:
                    legend = True
            else:
                ax[idx].plot(v)
        if legend:
            ax[idx].legend()
        ax[idx].grid()
        ax[idx].set_title(key)
    
    if save_path is None:
        plt.plot()
    else:
        plt.savefig(save_path)
        plt.close()


def plots_std(series, series_x=None, save_path=None, title=None):
    title = title.replace('.',',')
    
    n_lines = int(np.ceil(len(series)/2))
    fig, ax = plt.subplots(n_lines,2,figsize=(16,12))
    
    if title is not None:
        fig.suptitle(title, fontsize=16)

    for i, (key, value) in enumerate(series.items()):
        legend = False
        if n_lines==1:
            idx = i%2
        else:
            idx = (i//2,i%2)
        x = np.arange(len(value[0]['mean'])) if series_x is None else series_x[i]
        for v in value:
            p = ax[idx].plot(x, v['mean'], lw=1, ms=9, mec='black',)
            ax[idx].fill_between(x=x, y1=v['mean']+v['std'], y2=v['mean']-v['std'], color=p[-1].get_color(), alpha=0.18)
            if 'label' in v:
                legend = True
        if legend:
            ax[idx].legend()
        ax[idx].grid()
        ax[idx].set_title(key)
    
    if save_path is None:
        plt.plot()
    else:
        plt.savefig(save_path)
        plt.close()