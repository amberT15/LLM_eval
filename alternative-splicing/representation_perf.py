import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import scipy.stats
import sys
import os
sys.path.append('../data_generation/')
import mt_preprocess
import custom_model

data_file = sys.argv[1]
model_dir = sys.argv[2]
output_file = sys.argv[3]
sub_ratio = float(sys.argv[4])

rep_list = {}
rep_list['Tissue'] = mt_preprocess.a_tissues

#Read dataset
data = h5py.File(data_file,'r')
xl_test = data['xl_test'][()]
xr_test = data['xr_test'][()]
y_test = data['y_test'][()]

#Sub-select train/valid set
train_idx = np.random.choice(range(len(data['xl_train'])),
                            size = int(sub_ratio*len(data['xl_train'])),
                            replace = False)
train_idx = np.sort(train_idx)
valid_idx = np.random.choice(range(len(data['xl_valid'])),
                            size = int(sub_ratio*len(data['xl_valid'])),
                            replace = False)
valid_idx = np.sort(valid_idx)

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
v_xl = xl_test[filter_list]
v_xr = xr_test[filter_list]
v_y = y_test[filter_list]

#create TF dataset object
with tf.device("CPU"):
    trainset = tf.data.Dataset.from_tensor_slices(({"input_1": data['xl_train'][train_idx], 
                                                    "input_2": data['xr_train'][train_idx]},
                                                   data['y_train'][train_idx])).shuffle(256*4).batch(256)
    validset = tf.data.Dataset.from_tensor_slices(({"input_1": data['xl_valid'][valid_idx], 
                                                    "input_2": data['xr_valid'][valid_idx]},
                                                   data['y_valid'][valid_idx])).shuffle(256*4).batch(256)
    # testset = tf.data.Dataset.from_tensor_slices(({"input_1": v_xl, "input_2": v_xr},
    #                                               v_y)).shuffle(256*4).batch(256)

#model callbacks
earlyStopping_callback = tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True
        )
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2,
            patience=5, min_lr=1e-6)

#run replication experiments
rep_list = {}
rep_list['Tissue'] = mt_preprocess.a_tissues

for i in range(5):
    
    #build model
    model = custom_model.mtsplice_model(56,input_shape =xl_test[0].shape )
    loss = custom_model.diff_KL()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    if sub_ratio == 1:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                            model_dir + '/model'+str(i)+'.h5',
                                            monitor='val_loss',
                                            save_best_only=True,
                                            mode = 'min',
                                            save_freq='epoch',)
    else:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                            model_dir + '/model'+str(i)+'_'+str(sub_ratio)+'.h5',
                                            monitor='val_loss',
                                            save_best_only=True,
                                            mode = 'min',
                                            save_freq='epoch',)
    model.compile(loss = loss,
                optimizer=optimizer)
    print('Start training')
    result = model.fit(trainset,
                batch_size=256,
                validation_data=validset,
                epochs=100,
                verbose=0,
                callbacks=[earlyStopping_callback,reduce_lr,checkpoint]
                )
    
    p_y = model.predict([v_xl,v_xr])
    print(p_y.shape)

    #clear session
    tf.keras.backend.clear_session()
    print('evaluating')
    corr_list = []
    for a in tqdm(range(0,56)):
        corr,_ = scipy.stats.spearmanr(v_y[:,a,0],p_y[:,a],nan_policy='omit')
        corr_list.append(corr)
        #print(mt_preprocess.a_tissues[i],' PR: ',corr)

    rep_list['rep' + str(i)] = corr_list

if sub_ratio != 1:
    path = './result/' + str(sub_ratio)+'/'
else:
    path = './result/'

if os.path.exists(path) == False:
    os.makedirs(path)

perf_df = pd.DataFrame.from_dict(rep_list)
perf_df.to_csv(path + output_file)