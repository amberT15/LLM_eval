import h5py 
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import mean_squared_error

f = h5py.File('../data/eclip/K562.h5','r')
x_train = f['X_train'][()]
y_train = f['Y_train'][()]
x_test = f['X_test'][()]
y_test = f['Y_test'][()]
x_valid = f['X_valid'][()]
y_valid = f['Y_valid'][()]

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
            
def model_2CNN(input_shape,output_shape):

  l2 = keras.regularizers.l2(1e-8)
  # input layer
  inputs = keras.layers.Input(shape=input_shape)

  # layer 1 - convolution
  nn = keras.layers.Conv1D(filters=96,
                              kernel_size=19,
                              strides=1,
                              activation=None,
                              use_bias=False,
                              padding='same',
                              kernel_regularizer=l2, 
                              )(inputs)        
  nn = keras.layers.BatchNormalization()(nn)
  nn = keras.layers.Activation('exponential')(nn)
  nn = keras.layers.Dropout(0.1)(nn)
  
  nn = residual_block(nn, filter_size=3, dilated=True)
  nn = keras.layers.Activation('relu')(nn)

  #change parameter
  nn = keras.layers.MaxPool1D(pool_size=10)(nn)
  nn = keras.layers.Dropout(0.1)(nn)
  
  # layer 2 - convolution
  nn = keras.layers.Conv1D(filters=192,
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
  nn = keras.layers.Dense(512,
                          activation=None,
                          use_bias=False,
                          kernel_regularizer=l2, 
                          )(nn)      
  nn = keras.layers.BatchNormalization()(nn)
  nn = keras.layers.Activation('relu')(nn)
  nn = keras.layers.Dropout(0.5)(nn)

  # Output layer
  logits = keras.layers.Dense(output_shape, activation='linear', use_bias=True)(nn)
  outputs = keras.layers.Activation('sigmoid')(logits)
  model = keras.Model(inputs=inputs, outputs=outputs)
  return model


model = model_2CNN(x_train.shape[1:],y_train.shape[1])

auroc = keras.metrics.AUC(curve='ROC', name='auroc')
aupr = keras.metrics.AUC(curve='PR', name='aupr')
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0)
model.compile(optimizer=optimizer, loss=loss, metrics=[auroc, aupr])

es_callback = keras.callbacks.EarlyStopping(monitor='val_aupr',
                                            patience=12, 
                                            verbose=1, 
                                            mode='max', 
                                            restore_best_weights=True)

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_aupr', 
                                              factor=0.2,
                                              patience=4, 
                                              min_lr=1e-7,
                                              mode='max',
                                              verbose=1) 


# train model
history = model.fit(x_train, y_train, 
                     epochs=100,
                     batch_size=100, 
                     shuffle=True,
                     validation_data=(x_valid, y_valid), 
                     callbacks=[es_callback, reduce_lr])
    
results = model.evaluate(x_test, y_test, batch_size=512)  
model.save('../model/RBP_pretrain_1k.h5')
