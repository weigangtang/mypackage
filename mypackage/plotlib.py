import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
import seaborn as sns

import sys, pathlib
sys.path.append(pathlib.Path(__file__).parents[0]) 

from hyclean import find_gaps

def plot_flow(flow, ax=None):
    
    if not isinstance(ax, matplotlib.axes._subplots.Axes): ax = plt.gca()

    x, y = flow.index, flow.values
    y_max = np.nanmax(y) * 1.05
    
    ax.plot(x, y)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, y_max)
    
    seindx = find_gaps(y)    
    for [sindx, eindx] in seindx:
        left_indx = max([sindx - 1, 0])
        left_edge = matplotlib.dates.date2num(x[left_indx])
        right_indx = min([eindx+1, len(x)-1])
        right_edge = matplotlib.dates.date2num(x[right_indx])
        width = right_edge - left_edge
        rect = plt.Rectangle([left_edge, 0], width, y_max, color='k', alpha=0.2, ec=None)
        ax.add_patch(rect)

    return ax



def plot_gts(hys, ax=None):

    if not isinstance(ax, matplotlib.axes._subplots.Axes): ax = plt.gca()

    avg_hy = np.mean(hys, axis=0)
    ax.plot(hys.transpose(), linewidth=0.5, color='#808080', alpha=0.2)
    ax.plot(avg_hy, linewidth=1.2, color='#0082c8')
    
    ax.set_xlim(0, hys.shape[1])
    ax.legend(ax.lines[-2:], ['Annual', 'Generalized'])

    return ax


def plot_corr_mat(df):
    nvar = df.shape[1]
    varname = df.columns
    
    fig = plt.figure(figsize=(3*nvar, 2.4*nvar))
    gs = plt.GridSpec(nvar, nvar)

    for i in range(nvar):
        for j in range(nvar): 
            ax = fig.add_subplot(gs[i, j])
            x, y = df.iloc[:, i].values, df.iloc[:, j].values
            indx = ~(np.isnan(x) | np.isinf(x) | np.isnan(y) | np.isinf(y))
            x, y = x[indx], y[indx]
            if i == j:
                sns.distplot(x, color='tab:green', ax=ax);
            elif i > j: 
                sns.regplot(y, x, ax=ax, line_kws={'lw': 1},
                            scatter_kws={'s': 10, 'color': 'gray', 'alpha': 0.5})
            elif i < j: 
                ax.hexbin(y, x, cmap=plt.cm.Reds)  
            if i == 0:
                ax.set_title(varname[j], color='black', fontsize=14, fontweight='bold')
            if j == 0:
                ax.set_ylabel(varname[i], color='black', fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_n_y_axis(df, ylabels=[], ylims=[], mk_prop={}): 
    
    tk_prop = {'size': 4, 'width': 1.5} 
    
    colnames = df.columns.tolist()
    ncol = len(colnames)
    colors = plt.cm.tab10(range(10))
    
    if len(ylabels) != ncol: 
        if len(ylabels) != 0: 
            print('ylabels mismatch dataframe!')
        ylabels = colnames
    
    fig, ax = plt.subplots()
    fig.subplots_adjust(right=1-0.11*(ncol-1))
    
    t = df.index.values
    y0 = df[colnames[0]].values
    
    varname = colnames[0]
    c = colors[0]
    ylab = ylabels[0]
    
    p0, = ax.plot(t, y0, color=c, label=varname, **mk_prop)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel('Time')
    ax.set_ylabel(ylab)
    ax.yaxis.label.set_color(c)
    ax.tick_params(axis='x', **tk_prop)
    ax.tick_params(axis='y', colors=c, **tk_prop)
    lines = [p0]
    axes = [ax]
    
    if ncol > 1: 
        
        offset = 0.18
        for i in range(1, ncol): 
            varname = colnames[i]
            c = colors[i]
            ylab = ylabels[i]
            par = ax.twinx()
            
            par.spines['right'].set_position(('axes', 1+offset*(i-1)))
            par.spines['right'].set_visible(True)
                
            yi = df[varname]
            pi, = par.plot(t, yi, color=c, label=varname, **mk_prop)
            par.set_ylabel(ylab)
            par.yaxis.label.set_color(c)
            par.tick_params(axis='y', colors=c, **tk_prop)
            
            lines.append(pi)
            axes.append(par)
    
    if len(ylims) == ncol: 
        for i in range(ncol): axes[i].set_ylim(ylims[i])
    elif len(ylims) > 0: 
        print('ylims mismatch dataframe!')
    
    ax.legend(lines, [l.get_label() for l in lines])
    return fig, ax
