import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
import seaborn as sns

## Inner helper functions
def grad_times_input_to_df(x, grad, alphabet='ACGT'):
    """generate pandas dataframe for saliency plot
     based on grad x inputs """

    x_index = np.argmax(np.squeeze(x), axis=1)
    grad = np.squeeze(grad)
    L, A = grad.shape

    seq = ''
    saliency = np.zeros((L))
    for i in range(L):
        seq += alphabet[x_index[i]]
        saliency[i] = grad[i,x_index[i]]

    # create saliency matrix
    saliency_df = logomaker.saliency_to_matrix(seq=seq, values=saliency)
    return saliency_df

def plot_attribution_map(saliency_df, ax=None, figsize=(20,2)):
    """plot an single attribution map using logomaker"""

    logomaker.Logo(saliency_df, figsize=figsize, ax=ax)
    if ax is None:
        ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    #plt.xticks([])
    #plt.yticks([])

## Function for plot genertaion
def plot_clean_saliency(X_sample,saliency_scores,window=200,titles=[],filename=None, ax=None):
    "Plot clean saliency with average channel saliency subtracted"
    axs = None
    N, L, A = X_sample.shape
    if ax == None:
        fig, axs = plt.subplots(N, 1, figsize=[20, 2 * N])
    average_saliency = np.average(saliency_scores,axis=-1)
    average_matrix = np.repeat(average_saliency[:,:,np.newaxis],4,axis=2)
    saliency_scores = saliency_scores - average_matrix
    for i in range(N):
        if N == 1 and axs is not None:
            ax = axs
        elif axs is not None:
            ax = axs[i]
        x_sample = np.expand_dims(X_sample[i], axis=0)
        scores = np.expand_dims(saliency_scores[i], axis=0)
        # find window to plot saliency maps (about max saliency value)
        index = np.argmax(np.max(np.abs(scores), axis=2), axis=1)[0]
        if index - window < 0:
            start = 0
            end = window * 2 + 1
        elif index + window > L:
            start = L - window * 2 - 1
            end = L
        else:
            start = index - window
            end = index + window

        saliency_df = grad_times_input_to_df(x_sample[:, start:end, :], scores[:, start:end, :])
        plot_attribution_map(saliency_df, ax, figsize=(20, 2))
        if len(titles):
            ax.set_title(titles[i])
    plt.tight_layout()
    if filename:
        assert not os.path.isfile(filename), 'File exists!'
        plt.savefig(filename, format='svg')

def plot_prob_matrix(X_sample,p_matrix,window=200,titles=[],filename=None):
    """ Plot Probability Matrix for predicted probability per nucleotide """
    N, L, A = X_sample.shape
    fig, axs = plt.subplots(N, 1, figsize=[20, 2 * N])
    for i in range(N):
        if N == 1:
            ax = axs
        else:
            ax = axs[i]
        x_sample =X_sample[i]
        scores = p_matrix[i]
        # find window to plot saliency maps (about max saliency value)
        index = np.argmax(np.max(np.abs(scores), axis=1), axis=0)
        if index - window < 0:
            start = 0
            end = window * 2 + 1
        elif index + window > L:
            start = L - window * 2 - 1
            end = L
        else:
            start = index - window
            end = index + window

        prob_df = grad_times_input_to_df(x_sample[start:end, :], scores[start:end, :])
        final_df = prob_df.rename(columns={0:'A',1:'C',2:'G',3:'T'})
        plot_attribution_map(final_df, ax, figsize=(20, 1))

def plot_info_matrix(X_sample,p_matrix,window=200,titles=[],filename=None):
    """Plot information matrix with all four nucleotide included"""
    N, L, A = X_sample.shape
    fig, axs = plt.subplots(N, 1, figsize=[20, 2 * N])
    for i in range(N):
        if N == 1:
            ax = axs
        else:
            ax = axs[i]
        x_sample = np.expand_dims(X_sample[i], axis=0)
        scores = p_matrix[i]
        # find window to plot saliency maps (about max saliency value)
        index = np.argmax(np.max(np.abs(scores), axis=1), axis=0)
        if index - window < 0:
            start = 0
            end = window * 2 + 1
        elif index + window > L:
            start = L - window * 2 - 1
            end = L
        else:
            start = index - window
            end = index + window

        saliency_df = pd.DataFrame(np.squeeze(scores))
        info_df = logomaker.transform_matrix(saliency_df,from_type = 'probability',to_type='information')
        final_df = info_df.rename(columns={0:'A',1:'C',2:'G',3:'T'})
        plot_attribution_map(final_df, ax, figsize=(20, 1))

def plot_value_per_loc(X_sample,entropy,window=200,titles=[],filename=None,ax = None):
    """Plot information matrix with all four nucleotide included"""
    #N, L, A = X_sample.shape
    axs = None
    N = len(X_sample)
    if ax == None:
        fig, axs = plt.subplots(N, 1, figsize=[20, 2 * N])
    for i in range(N):
        if N == 1 and axs is not None:
            ax = axs
        elif axs is not None:
            ax = axs[i]
        x_sample = X_sample[i]
        L = len(x_sample)
        scores = entropy[i]
        # find window to plot saliency maps (about max saliency value)
        index = np.argmax(np.abs(scores),axis=0)
        if index - window < 0:
            start = 0
            end = window * 2 + 1
        elif index + window > L:
            start = L - window * 2 - 1
            end = L
        else:
            start = index - window
            end = index + window

        entropy_matrix = x_sample*scores[:L,None]
        entropy_df = pd.DataFrame(entropy_matrix)
        final_df = entropy_df.rename(columns={0:'A',1:'C',2:'G',3:'T'})
        plot_attribution_map(final_df, ax, figsize=(20, 2))
    plt.tight_layout()