import h5py
import sys
import csv
import numpy as np
import tensorflow as tf
import mpra_model
from sklearn import model_selection
import scipy.stats

cnn_config = {
    'activation':'exponential',
    'reduce_dim': 196,
    'conv1_filter':196,
    'conv1_kernel':7,
    'dropout1':0.2,
    'res_pool':5,
    'res_dropout':0.2,
    'conv2_filter':256,
    'conv2_kernel':7,
    'pool2_size':4,
    'dropout2':0.2,
    'dense':512,
    'dense2':256
}

for celltype in ['HepG2','K562']:
    file = '../data/lenti_MPRA/'+celltype+'_data.h5'

    f = h5py.File(file, 'r')
    x_train = f['onehot_train'][:]
    x_valid = f['onehot_valid'][:]
    x_test = f['onehot_test'][:]
    y_train = f['y_train'][:]
    y_valid = f['y_valid'][:]
    y_test = f['y_test'][:]
    
    for model_name in ['rep_cnn','ResNet','MPRAnn']:
        save_model  = '../model/lenti_MPRA/%s_%s.h5'%(model_name,celltype)
        
        model_func = getattr(mpra_model,model_name)
        model = model_func((230,4),config = cnn_config)
        
        earlyStopping_callback = tf.keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True
            )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.2,
                patience=5, min_lr=1e-6)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(save_model,
                                        monitor='val_loss',
                                        save_best_only=True,
                                        mode = 'min',
                                        save_freq='epoch',)
        model.compile(
                    loss="mean_squared_error",
                    metrics=["mse", "mae"],
                    optimizer=optimizer,
                )
        
        result = model.fit(x_train,y_train,
            batch_size=128,
            validation_data=(x_valid,y_valid),
            epochs=100,
            shuffle=True,
            verbose=2,
            callbacks=[earlyStopping_callback,checkpoint,reduce_lr],
        )

        y_pred = model.predict(x_test)
        pr = scipy.stats.pearsonr(np.squeeze(y_pred),np.squeeze(y_test))[0]

        print(pr)
  
    fields=[model_name,'CNN',pr,celltype]
    with open(r'./results/MPRA.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    del model
