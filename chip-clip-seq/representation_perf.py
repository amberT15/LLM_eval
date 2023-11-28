import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import glob
import os
import sys

f_dir = sys.argv[1]
output_dir = sys.argv[2]
model_name = sys.argv[3]
if not os.path.exists(output_dir):
    print('making output directory')
    os.makedirs(output_dir)


file_list = glob.glob(f_dir)
print('Number of tasks detected : %d'%(len(file_list)))
print(file_list)
test_aupr = []
test_auroc = []
test_accuracy = []
tf_list = []
model_list = []

#model setup
def chip_cnn(input_shape,output_shape):
    initializer = tf.keras.initializers.HeUniform()
    input = keras.Input(shape=input_shape)
    
    #add batchnorm and dimension reduction
    if model_name!= 'GPN' and model_name != 'CNN':
        nn = keras.layers.BatchNormalization()(input)
    else:
        nn = input
    nn = keras.layers.Conv1D(filters=512,kernel_size=1,
                             kernel_initializer = initializer)(nn)
    #first conv layer
    nn = keras.layers.Conv1D(filters=64,
                             kernel_size=7,
                             padding = 'same',
                             kernel_initializer=initializer)(input)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    #Second conv layer
    nn = keras.layers.Conv1D(filters=96,
                             kernel_size=5,
                             padding = 'same',
                             kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    #third conv layer
    nn = keras.layers.Conv1D(filters=128,
                             kernel_size=5,
                             padding = 'same',
                             kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(2)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    #dense layer
    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(256)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    #Output layer
    logits = keras.layers.Dense(output_shape)(nn)
    output = keras.layers.Activation('sigmoid')(logits)
    
    model = keras.Model(inputs=input, outputs=output)
    return model
earlyStopping_callback = tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True
        )
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2,
            patience=5, min_lr=1e-6)
auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0)

#loop through TFs
for file in file_list:
    tf_name = file.split('/')[-1][:-7]

    #load dataset into TF dataset form
    data = h5py.File(file,'r')   
    with tf.device("CPU"):
        trainset = tf.data.Dataset.from_tensor_slices((data['x_train'][()],data['y_train'][()])).shuffle(256*4).batch(256)
        validset = tf.data.Dataset.from_tensor_slices((data['x_valid'][()],data['y_valid'][()])).shuffle(256*4).batch(256)
        testset = tf.data.Dataset.from_tensor_slices((data['x_test'][()],data['y_test'][()])).shuffle(256*4).batch(256)
    
    #model compile and training
    print('########Training ',tf_name, '########')
    for i in range(5):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                        output_dir+tf_name+'.h5',
                                        monitor='val_loss',
                                        save_best_only=True,
                                        mode = 'min',
                                        save_freq='epoch',)
        model = chip_cnn((data['x_train'][0].shape),1)
        model.compile(loss = loss,
                    metrics=['accuracy',auroc,aupr],
                    optimizer=optimizer)
        
        result = model.fit(trainset,
            batch_size=256,
            validation_data=validset,
            epochs=100,
            verbose=2,
            callbacks=[earlyStopping_callback,reduce_lr,checkpoint]
        )
        _, acc, roc, pr = model.evaluate(testset)
        tf_list.append(tf_name)
        test_accuracy.append(acc)
        test_auroc.append(roc)
        test_aupr.append(pr)
        model_list.append(model_name)

### collect result into csv
df = pd.DataFrame(list(zip(tf_list, test_accuracy, test_auroc, test_aupr,model_list)),
               columns =['TF','Accuracy','AUROC','AUPR','Model'])

df.to_csv('./result/eclip_result/'+model_name+'_perf.csv')







