import numpy as np

REVERSE_VOCAB = np.array(['A','C','G','T','N'])

def onehot_to_seq(onehot):
    adj = np.sum(onehot, axis=-1) == 0
    x_index = np.argmax(onehot,axis = -1) - adj
    seq_onehot = REVERSE_VOCAB[x_index]
    seq_char = [''.join(single_seq) for single_seq in seq_onehot]
    return seq_char

def seq_to_onehot(seq):
    """
    Function to convert string DNA sequences into onehot
    :param seq: string DNA sequence
    :return: onehot sequence
    """
    seq_len = len(seq)
    seq_start = 0
    seq = seq.upper()

    # map nt's to a matrix len(seq)x4 of 0's and 1's.
    seq_code = np.zeros((seq_len, 4), dtype='float16')

    for i in range(seq_len):
        if i >= seq_start and i - seq_start < len(seq):
            nt = seq[i - seq_start]
            if nt == 'A':
                seq_code[i, 0] = 1
            elif nt == 'C':
                seq_code[i, 1] = 1
            elif nt == 'G':
                seq_code[i, 2] = 1
            elif nt == 'T':
                seq_code[i, 3] = 1
            else:
                seq_code[i, :] = 0.25

    return seq_code

def onehot_rc(seq,rc_range=None):
    if rc_range is None:
        rc_range = (0,seq.shape[1])
    reverse_seq = seq[:,rc_range[0]:rc_range[1]]
    reverse_seq = np.flip(reverse_seq,axis=0)
    rc_seq = np.concatenate((seq[:,0:rc_range[0]],reverse_seq,seq[:,rc_range[1]:]),axis = 1)
    return rc_seq