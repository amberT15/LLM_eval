import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import scipy.stats
import os
import sys
sys.path.append('/home/ztang/multitask_RNA/data_generation/mtsplice/')
import mt_preprocess
import custom_model

if len(sys.argv) > 1:
    sub_ratio = float(sys.argv[1])
else:
    sub_ratio = 1


rep_list = {}
rep_list['Tissue'] = mt_preprocess.a_tissues

#Read dataset
file = h5py.File('/home/ztang/multitask_RNA/data/mtsplice/delta_logit.h5','r')
x_train = file['x_train'][()]
y_train = file['y_train'][()]
x_valid = file['x_valid'][()]
y_valid = file['y_valid'][()]
x_test = file['x_test'][()]
y_test = file['y_test'][()]

if sub_ratio != 1:
    train_idx = np.random.choice(range(len(x_train)),size = int(sub_ratio*len(x_train)),replace=False)
    
    x_train = x_train[train_idx]
    y_train = y_train[train_idx]

    valid_idx = np.random.choice(range(len(x_valid)),size = int(sub_ratio*len(x_valid)),replace=False)
    x_valid = x_train[valid_idx]
    y_valid = y_train[valid_idx]


#filter test set
filter_list = []
for i in range(len(y_test)):
    target = y_test[i,:,0]
    mean = y_test[i,:,1]
    target = target + mean
    nan_count = np.sum(~np.isnan(target))
    target_psi = tf.math.sigmoid(target)
    mean_psi = tf.math.sigmoid(mean)
    diff = np.abs(mean_psi - target_psi)
    if nan_count >=10 and np.nanmax(np.abs(diff)) >= 0.2:
        filter_list.append(i)
print(len(filter_list))

v_x = x_test[filter_list]
v_y = y_test[filter_list]

earlyStopping_callback = tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True
        )
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2,
            patience=5, min_lr=1e-6)


#replications
for i in range(5):
    model = custom_model.mtsplice_model(56)
    #loss = nan_KL()
    loss = custom_model.diff_KL()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    if sub_ratio == 1:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                            '/home/ztang/multitask_RNA/model/mtsplice/delta_model'+str(i)+'.h5',
                                            monitor='val_loss',
                                            save_best_only=True,
                                            mode = 'min',
                                            save_freq='epoch',)
    else:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                            '/home/ztang/multitask_RNA/model/mtsplice/delta_model'+str(i)+'_'+str(sub_ratio)+'.h5',
                                            monitor='val_loss',
                                            save_best_only=True,
                                            mode = 'min',
                                            save_freq='epoch',)
    model.compile(loss = loss,
                optimizer=optimizer)
    
    result = model.fit([x_train[:,:400,:],x_train[:,400:,:]], y_train,
                batch_size=256,
                validation_data=([x_valid[:,:400,:],x_valid[:,400:,:]], y_valid),
                epochs=100,
                verbose=0,
                callbacks=[earlyStopping_callback,reduce_lr,checkpoint]
                )
    #pred on selected test set
    p_y = model.predict([v_x[:,:400],v_x[:,400:]])

    corr_list = []
    for a in tqdm(range(0,56)):
        corr,_ = scipy.stats.spearmanr(v_y[:,a,0],p_y[:,a],nan_policy='omit')
        corr_list.append(corr)
        #print(mt_preprocess.a_tissues[i],' PR: ',corr)

    rep_list['rep' + str(i)] = corr_list

if sub_ratio != 1:
    path = '/home/ztang/multitask_RNA/replications/MTSplice/result/' + str(sub_ratio)+'/'
else:
    path = '/home/ztang/multitask_RNA/replications/MTSplice/result/'

if os.path.exists(path) == False:
    os.makedirs(path)

perf_df = pd.DataFrame.from_dict(rep_list)
perf_df.to_csv(path+'CNN_perf.csv')