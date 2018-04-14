#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains functions to plot "heatmaps" of a Confusion Matrix

Idea and some calculations inspired by other works, includinng that of
stack overflow user `amillerrhodes` https://stackoverflow.com/questions/5821125

Author: Reynaldo Vazquez
Created on Sun Apr 8, 2018
"""
import numpy as np
import matplotlib.pyplot as plt

def populate_plot(ax, cm, cm_normalized, cm_color_vals, cmap, 
                  annotation = "both"):
    """
    Fills figure and values in a confusion matrix subplot
    
    Args:
        ax: a matplotlib AxesSubplot 
        cm: cm: a confusion matrix array
        cm_color_vals: a matrix of shape cm.shapen calculated 
                       in heated_confusion_matrix()
        cmap: a color map
        annotation = a string in ("both", "normalized", "absolute") indicating
                     the metric to be annotated in each plot cell
    """
    ax.imshow(cm_color_vals, cmap=cmap, interpolation='none', vmin=0.00000001)
    width, height = cm.shape
    for x in range(width):
        for y in range(height):
            cell_count = (str(cm[x,y]))
            cell_normalized = str(format(cm_normalized[x,y], '.2f'))
            if annotation is "both":
                cell_text = cell_count + "\n \n(" + cell_normalized + ")"
            elif annotation is "absolute":
                cell_text = cell_count
            elif annotation is "normalized":
                cell_text = cell_normalized
            ax.annotate(cell_text, xy=(y, x), 
                    horizontalalignment='center',
                    fontweight = 'bold' if cm[x,y] > 0 else "normal", size=12,
                    verticalalignment='center', 
                    color="white" if cm[x,y] > 0 else "gray")
    return None

def ticks_labels_title(ax, labels):
    """
    Labels a subplot's axes
    
    Args:
        ax: a matplotlib AxesSubplot 
        labels: a list with class labels
    """
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontweight = 'bold', size=13)
    ax.set_yticklabels(labels, fontweight = 'bold', size=13)
    ax.set_ylabel('Actual', fontweight = 'bold', size=14)
    ax.set_xlabel('Predicted', fontweight = 'bold', size=14)
    for spine in ax.spines.values():
        spine.set_edgecolor('#f1f1f1')
    return None

def side_by_side(cm, cm_normalized, cm_color_vals, cmap, labels, plot_title, p_size):
    """
    Creates side by side plots of the regular confusion matrix and the 
    normalizedd confusion matrix. Used when by_side = True in 
    heated_confusion_matrix()
    
    Args:
        cm: cm: a confusion matrix array
        cm_normalized: a normalized confusion matrix calculated
                       in heated_confusion_matrix()
        cm_color_vals: a matix of shape cm.shapen calculated 
                       in heated_confusion_matrix()
        cmap: a color map
        annotation = a string in "both", "normalized", "absolute" indicating
                     the metric to be annotated in each plot cell
        labels: a list with class labels
        plot_title: a string, determined in heated_confusion_matrix()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    axs = [ax1,ax2]
    plot_type = ["absolute", "normalized"]
    fig.set_size_inches(1.7*p_size, p_size)
    for f, ax in enumerate(axs):
        populate_plot(ax, cm, cm_normalized, cm_color_vals, cmap, 
                      annotation = plot_type[f])
        ticks_labels_title(ax, labels)
    ax1.set_title("Counts", fontweight = 'bold', size=11)
    ax2.set_title("Normalized", fontweight = 'bold', size=11)
    title_size = min((p_size+10), 18)
    fig.suptitle(plot_title, fontweight = 'bold', size=title_size)  
    fig.tight_layout(rect=[0, 0, 1, 1])
    return None
        
def one_plot(cm, cm_normalized, cm_color_vals, cmap, annotation, 
             labels, plot_title, p_size):
    """
    Creates a single confusion matrix plot,  used when by_side = False in 
    heated_confusion_matrix()
    
    Args:
        cm: cm: a confusion matrix array
        cm_normalized: a normalized confusion matrix calculated
                       in heated_confusion_matrix()
        cm_color_vals: a matix of shape cm.shapen calculated 
                       in heated_confusion_matrix()
        cmap: a color map
        annotation = a string in ("both", "normalized", "absolute") indicating
                     the metric to be annotated in each plot cell
        labels: a list with class labels
        plot_title: a string, determined in heated_confusion_matrix()
    """
    fig = plt.figure()
    fig.set_size_inches(p_size, p_size)
    ax = fig.add_subplot(111)
    populate_plot(ax, cm, cm_normalized, cm_color_vals, cmap, annotation)
    ticks_labels_title(ax, labels)
    title_size = min((p_size+8.5), 18)
    plt.title(plot_title, fontweight = 'bold', size=title_size)  
    plt.tight_layout()
    return None


def heated_confusion_matrix(cm, labels, annotation = "both", 
                               p_size = 8,
                               cmap = plt.cm.Reds,
                               contrast = 4, 
                               model_name = None, 
                               by_side = False,
                               save_fig = False):
    """
    Plots a heatmap of the confusion matrix
    
    Args: 
        cm: a confusion matrix array. i.e. a sklearn.metrics.confusion_matrix()
        labels: a list with class labels
        annotation = a string in ("both", "normalized", "absolute") indicating
                     the metric to be annotated in each plot cell. Only applies 
                     when by_side = False
        cmap: a color map
        contrast: an integer or float, higher numbers will place more weight on
            wrong predictions in terms of the color intensity
        model_name: a string with the model name to print in the title
        by_side: if False, will return only one confusion matrix with specified
                     annotation. If True, will plot a "Count" Confusion Matrix 
                     and a Normalized confusion matrix side by side. 
    """
    if model_name is None:
        model_name = ""
        plot_title = "\nConfusion Matrix\n"
    else:
        plot_title = "\nConfusion Matrix\n" + "Model: " + model_name + "\n"
        
        
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_color_vals = np.power(cm_normalized, 1/contrast)
    cm_color_vals = np.ma.masked_where(cm_color_vals == 0, cm_color_vals)
    cmap.set_under(color= "white")
    
    if by_side is False:
        one_plot(cm, cm_normalized, cm_color_vals, cmap, annotation, labels,
                 plot_title, p_size)
    else:
        side_by_side(cm, cm_normalized, cm_color_vals, cmap, labels, plot_title, p_size)
        
    if save_fig is True:
        file_name = 'confusion_matrix' + model_name + '.png'
        plt.savefig(file_name, format='png', pad_inches=0.1, dpi = 480)
    plt.show()
    return None