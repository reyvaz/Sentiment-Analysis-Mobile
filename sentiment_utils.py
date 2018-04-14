"""
Contains functions to aid in NLP Analysis. Useful mostly for  Binary 
classification. 

Author: Reynaldo Vazquez
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import pandas as pd
import time

def influential_features(coefficients, vect, num_features = 10):
    '''
    Retrieves the most influential features in binary classification models that 
    support this feature i.e. Logistic regression
    
    Args: 
        coefficients: model coefficients, i.e. from coefficients = model.coef_[0]
        vect: a CountVectorizer() instance
        num_features: the number of most influential features in each direction requested

    Returns: 
        a list of 2 pandas series, each with influential features as index and their 
        respective coefficients as values
    '''
    feature_names = np.array(vect.get_feature_names())
    sorted_index   = coefficients.argsort()
    sm_index  = sorted_index[:num_features]
    lg_index  = sorted_index[:-(num_features+1):-1]
    sm_series = pd.Series(coefficients[sm_index], index = feature_names[sm_index])
    sm_series = sm_series.iloc[np.lexsort([sm_series.index, sm_series.values])]
    lg_series  = pd.Series(coefficients[lg_index], index = feature_names[lg_index])
    lg_series  = lg_series.iloc[np.lexsort([lg_series.index, -lg_series.values])]
    return lg_series, sm_series

def infl_features_non_tr(coefficients, feature_names, num_features = 10):
    '''
    NOTE: Use this when feature_names is non trivial
    Retrieves the most influential features in binary classification models that 
    support this feature i.e. Logistic regression
    
    Args: 
        coefficients: model coefficients, i.e. from coefficients = model.coef_[0]
        feature_names: feature_names can come from CountVectorizer()  .get_feature_names()
        num_features: the number of most influential features in each direction requested

    Returns: 
        a list of 2 pandas series, each with influential features as index and their 
        respective coefficients as values
    '''
    sorted_index   = coefficients.argsort()
    sm_index  = sorted_index[:num_features]
    lg_index  = sorted_index[:-(num_features+1):-1]
    sm_series = pd.Series(coefficients[sm_index], index = feature_names[sm_index])
    sm_series = sm_series.iloc[np.lexsort([sm_series.index, sm_series.values])]
    lg_series  = pd.Series(coefficients[lg_index], index = feature_names[lg_index])
    lg_series  = lg_series.iloc[np.lexsort([lg_series.index, -lg_series.values])]
    return lg_series, sm_series

def plot_features(features, title, classes, cmap = plt.cm.PuRd, 
                  label_rotation = 20, save_fig = False):
    """
    Plots features of a binary classification model
    
    Args:
        features: a list of 2 pandas series
        title: a list of 2 strings
        cmap: a color map
        label_rotation: float, to rotate x-axis labels 
        save_fig: True to save fig
    """
    print("\n")
    plot_colors = cmap(np.linspace(.9, .5, (features[0].shape[0])))
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.tight_layout(rect=[0, 0, .9, .9])
    axs = [ax1, ax2]
    fig.set_size_inches(10, 6)
    fig_title = 'Influential Features ' + "(" + title + ")"
    fig.suptitle(fig_title,
                 fontsize=18, color = '0.4', fontweight = 'bold')

    for n in range(len(axs)):
        feature = features[n]
        w = np.arange(len(feature))
        axs[n].bar(w, feature.values, align='center', alpha=0.8, color = plot_colors)
        axs[n].tick_params(length=0, width=0.5, colors='0.75', labelsize = 12)
        axs[n].set_xticks(np.arange(len(feature)))
        axs[n].set_xticklabels(feature.index, rotation = label_rotation, 
                               fontweight = 'bold', color = '#6f6f6f')
        axs[n].get_yaxis().set_ticks([])  
        xmin = min(w) - 0.4
        xmax = max(w) + 0.4
        axs[n].set_xlim(xmin, xmax)
        for edge in 'top', 'right', 'left', 'bottom':
            axs[n].spines[edge].set_visible(False)
        axs[n].set_ylabel(classes[1-n], fontweight = 'bold', size=16, color = '#6f6f6f')
    if save_fig is True:
        file_name = 'features' + title + '.png'
        plt.savefig(file_name, format='png', pad_inches=0.1, dpi = 480)
    plt.show()
    print("\n")
    return None

def predict_and_time(trained_model, X):
    """
    Calculates predictions and prints the time elapsed to make the predictions
    Args:
        trained_model: a trained (sklearn) model 
        X: a set of features
    Returns:
        predictions
    """
    model_name = str(trained_model).split("(")[0]
    start_time = time.time()
    predictions = trained_model.predict(X)
    elapsed_time = time.time() - start_time
    predictions = np.rint(predictions)   # Required by Random Forest
    print("Time to calculate", X.shape[0], "predictions with", model_name, "model:", 
          str(format(elapsed_time, '.4f')), "seconds")
    return predictions

def read_glove_vecs(glove_file):
    """
    Creates maps between words, index and GloVe vectors from the 
    glove.6B.50d.txt file
    
    Note: glove.6B.50d.txt from glove.6B.zip for more info
          check https://nlp.stanford.edu/projects/glove
    """
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

def sen2average(sentence, word_to_glove):
    """
    Encodes the meaning of a sentence by averaging representation vectors.
    
    Args:
        sentence: a sentence (string)
        word_to_glove: dictionary mapping words into their n-dimensional vector representation
    Returns:
        average: vector with the meaning of the sentence enconded
    """
    words = sentence.split()
    average = sum([word_to_glove[w] for w in words])/len(words)
    return average

def truncate_sentence(sentence, number_of_words = 100):
    """
    Truncates a sentence (str) to a number of words
    """
    return ' '.join(sentence.split()[:number_of_words])


def balanced_train_index(Y_train):
    """
    Creates an index to balance training X and Y sets of a binary classification
    variable. Retains all observations of the least representated class.
    
    Args: A pandas.Series or numpy array
    Returns: Index to balance X and Y training sets
    """
    np.random.seed(1)
    pos_ind = np.flatnonzero(Y_train == 1) 
    neg_ind = np.flatnonzero(Y_train == 0) 
    if pos_ind.shape[0] > neg_ind.shape[0]:
        print("1 is the dominant class")
        least = neg_ind
        most  = pos_ind 
    else:
        print("0 is the dominant class")
        most  = neg_ind
        least = pos_ind
    np.random.shuffle(most)
    balanced_index = np.concatenate((least, most[0:len(least)]))
    np.random.shuffle(balanced_index)
    return balanced_index

def preprocess_text(pandas_series):
    """
    Preprocesses text from a pandas series
    """
    processed_series = pandas_series.copy()
    p0 = '[hH][tT]{2}[pP]\S*|[wW]{3}\S*' # URL 
    p1 = "&amp"              # html entity for &, replace with 'and'
    p2 = "&\S{2,6}"          # rest of html entities, replace with ' '
    p3 = "[-_/,.]"           # hyphen, period, etc., replace with ' '
    p4 = "(?<=(.{2}))\1+"    # to trim repeated patterns i.e. gooooood
    p5 = "[!-'*-/;-@[-_{-~]" # finds most common punctuation etc, preserves :, ),  
                                   # and ( for :), and emoji. replace with ''   
    p6 = "\S{15,}"           # very large words replace with ' '
    p7 = "\s+"               # excessive white space, replace with ' '
    
    patterns = [p0, p1, p2, p3, p4, p5, p6, p7]
    substitutions = [' ', 'and', ' ', ' ', '', '', ' ', ' ' ]
    
    for i in range(len(patterns)):
        processed_series = processed_series.str.replace(patterns[i], substitutions[i])
    processed_series = processed_series.str.strip()
    return processed_series


def get_stop_words_list():
    from sklearn.feature_extraction import text 
    stop_words = text.ENGLISH_STOP_WORDS
    return stop_words
