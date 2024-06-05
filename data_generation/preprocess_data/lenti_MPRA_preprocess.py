import h5py
import numpy as np
import sys
sys.path.append('../embedding_generation/')
import utils

cell_type = ['HepG2','K562']
for ct in cell_type:

    train_folds = [1,2,3,4,5,6,7,8]
    valid_folds = [9]
    test_folds  = [10]

    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    x_test = []
    y_test = []
    file_name = '../../data/lenti_MPRA/lentiMPRA_'+ct+'.h5'
    with h5py.File(file_name, "r") as file:
        for i in train_folds:
            x_train.append(file[str(i)+"_x"][:])
            y_train.append(file[str(i)+"_y"][:])

        for i in valid_folds:
            x_valid.append(file[str(i)+"_x"][:])
            y_valid.append(file[str(i)+"_y"][:])

        for i in test_folds:
            x_test.append(file[str(i)+"_x"][:])
            y_test.append(file[str(i)+"_y"][:])


    x_train = np.vstack(x_train).astype(np.float32)
    seq_train = utils.onehot_to_seq(x_train)
    x_valid = np.vstack(x_valid).astype(np.float32)
    seq_valid = utils.onehot_to_seq(x_valid)
    x_test = np.vstack(x_test).astype(np.float32)
    seq_test = utils.onehot_to_seq(x_test)

    y_train = np.expand_dims(np.concatenate(y_train), axis=1)
    y_valid = np.expand_dims(np.concatenate(y_valid), axis=1)
    y_test = np.expand_dims(np.concatenate(y_test), axis=1)


    N, L, A = x_train.shape
    print([N, L, A])

    out_file = h5py.File('../../data/lenti_MPRA/'+ct+'_data.h5','w')
    out_file.create_dataset('onehot_train',data=x_train)
    out_file.create_dataset('onehot_valid',data=x_valid)
    out_file.create_dataset('onehot_test',data=x_test)
    out_file.create_dataset('seq_train',data=seq_train)
    out_file.create_dataset('seq_valid',data=seq_valid)
    out_file.create_dataset('seq_test',data=seq_test)
    out_file.create_dataset('y_train',data=y_train)
    out_file.create_dataset('y_valid',data=y_valid)
    out_file.create_dataset('y_test',data=y_test)