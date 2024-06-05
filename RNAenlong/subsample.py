import h5py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import gc
from sklearn.metrics import mean_squared_error
import sys

TASK_N = 1
#dataset = sys.argv[1]
dataset = 'rbp_embed'

def residual_block(input_layer, filter_size, activation='relu', dilated=False):
    if dilated:
        factor = [2, 4]
    else:
        factor = [1]
    num_filters = input_layer.shape.as_list()[-1]  

    nn = keras.layers.Conv1D(filters=num_filters,
                                    kernel_size=filter_size,
                                    activation=None,
                                    use_bias=False,
                                    padding='same',
                                    dilation_rate=1,
                                    )(input_layer) 
    nn = keras.layers.BatchNormalization()(nn)
    for f in factor:
        nn = keras.layers.Activation('relu')(nn)
        nn = keras.layers.Dropout(0.1)(nn)
        nn = keras.layers.Conv1D(filters=num_filters,
                                        kernel_size=filter_size,
                                        strides=1,
                                        activation=None,
                                        use_bias=False, 
                                        padding='same',
                                        dilation_rate=f,
                                        )(nn) 
        nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.add([input_layer, nn])
    return keras.layers.Activation(activation)(nn)
            
def model_2CNN(input_shape,output_shape,norm = False):

  l2 = keras.regularizers.l2(1e-8)
  # input layer
  inputs = keras.layers.Input(shape=input_shape)

  #optional layer norm
  if norm == True:
    nn = keras.layers.LayerNormalization()(inputs)
  else:
    nn = inputs

  # layer 1 - convolution
  nn = keras.layers.Conv1D(filters = 48, kernel_size=1)(nn)
  nn = keras.layers.Conv1D(filters=96,
                              kernel_size=19,
                              strides=1,
                              activation=None,
                              use_bias=False,
                              padding='same',
                              kernel_regularizer=l2, 
                              )(nn)        
  nn = keras.layers.BatchNormalization()(nn)
  nn = keras.layers.Activation('exponential')(nn)
  nn = keras.layers.Dropout(0.1)(nn)
  
  nn = residual_block(nn, filter_size=3, dilated=True)

  #change parameter
  nn = keras.layers.MaxPool1D(pool_size=10)(nn)
  nn = keras.layers.Dropout(0.1)(nn)
  
  # layer 2 - convolution
  nn = keras.layers.Conv1D(filters=128,
                              kernel_size=7,
                              strides=1,
                              activation=None,
                              use_bias=False,
                              padding='same',
                              kernel_regularizer=l2, 
                              )(nn)        
  nn = keras.layers.BatchNormalization()(nn)
  nn = keras.layers.Activation('relu')(nn)
  nn = keras.layers.GlobalAveragePooling1D()(nn)
  nn = keras.layers.Dropout(0.1)(nn)

  # layer 3 - Fully-connected 
  nn = keras.layers.Flatten()(nn)
  nn = keras.layers.Dense(128,
                          activation='relu',
                          use_bias=True
                          )(nn)      
  nn = keras.layers.Dropout(0.5)(nn)

  # Output layer
  outputs = keras.layers.Dense(output_shape, activation='linear', use_bias=True)(nn)
  model = keras.Model(inputs=inputs, outputs=outputs)
  return model

es_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=10, 
                                            verbose=1, 
                                            mode='min', 
                                            restore_best_weights=True)

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                              factor=0.2,
                                              patience=5, 
                                              min_lr=1e-7,
                                              mode='min',
                                              verbose=1) 

ratio_list = []
pr_list = []
mse_list = []
dataset_list = []
for data in [dataset]:
#for data in ['insert_dataset','gpn_human_embed','2B5_1000G_embed','2B5_1000G_embed_l10','2B5_1000G_embed_l12']:
    dataset = '../data/RNAenlong/' + data + '.h5'
    f = h5py.File(dataset,'r')
    x_train = f['X_train'][()]
    y_train = f['Y_train'][()]
    x_test = f['X_test'][()]
    y_test = f['Y_test'][()]
    x_valid = f['X_valid'][()]
    y_valid = f['Y_valid'][()]

    # lr = 0.001
    if data in ['insert_dataset']:
        lr = 0.001
    else:
        lr = 0.0001

    for i in range(1,11):
        dataset_size = int(len(x_train)*i*0.1)
        sample_data = np.random.choice(len(x_train),dataset_size,replace=False)
        sub_x_train = x_train[sample_data]
        sub_y_train = y_train[sample_data]
        log_train = np.log(sub_y_train[:,:1]+1)
        log_valid = np.log(y_valid[:,:1]+1)
        
        for rep in range(0,50):
            print('build model')
            model = model_2CNN(sub_x_train.shape[1:],1)
            optimizer = keras.optimizers.Adam(learning_rate=lr)
            loss = keras.losses.MeanSquaredError()
            model.compile(optimizer=optimizer, loss=loss)

            # train model
            print('#################run ',str(i),' for ', data, '################')
            history = model.fit(sub_x_train, log_train, 
                                epochs=200,
                                batch_size=128,
                                verbose = 0, 
                                shuffle=True,
                                validation_data=(x_valid, log_valid), 
                                callbacks=[es_callback, reduce_lr])
        
            y_pred = model.predict(x_test)
            y_label = np.log(y_test+1)
            pearsonr = scipy.stats.pearsonr(y_label[:,0], y_pred[:,0])
            mse = mean_squared_error(y_label[:,0], y_pred[:,0])
            print('#################PR ',str(pearsonr[0]),'################')
            ratio_list.append(i*0.1)
            pr_list.append(pearsonr[0])
            dataset_list.append(data)
            mse_list.append(mse)
            print('clean memory')
            del model
            del optimizer
            del loss
            del history
            del y_pred
            del y_label
            tf.keras.backend.clear_session()
            _ = gc.collect()
        
        f.close()
        del sub_x_train
        del sub_y_train
        del log_train
        del log_valid
        

df = pd.DataFrame(list(zip(ratio_list, pr_list,mse_list,dataset_list)),
               columns =['Data Ratio','Prediction Pearson R','Prediction MSE','Input Data'])

save_name = './result/rerun_'+data+'.pkl'
df.to_pickle(save_name)