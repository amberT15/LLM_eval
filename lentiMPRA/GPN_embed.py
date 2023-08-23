# %%
import importlib
import tensorflow as tf
import pandas as pd
import mpra_model
import h5py
importlib.reload(mpra_model)
import numpy as np
from tqdm import tqdm
import sklearn
from sklearn import model_selection
import scipy.stats
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
cell_type = 'K562'

# %%
cnn_config = {
    'input_shape': (41,2560),
    'activation':'exponential',
    'reduce_dim': 128,
    'conv1_filter':196,
    'conv1_kernel':7,
    'dropout1':0.2,
    'res_filter':5,
    'res_layers':3,
    'res_pool':5,
    'res_dropout':0.2,
    'conv2_filter':256,
    'conv2_kernel':7,
    'pool2_size':4,
    'dropout2':0.2,
    'dense':512,
    'dense2':256,
    'l_rate':0.0001
}

# %%
tf.keras.backend.clear_session()

data_dir = '/home/ztang/multitask_RNA/data/lenti_MPRA_embed/HepG2_seq_2B5_1000G/'

trainset = mpra_model.make_dataset(data_dir, 'train', mpra_model.load_stats(data_dir),
                            batch_size=128,seqs = False)
validset = mpra_model.make_dataset(data_dir, 'valid', mpra_model.load_stats(data_dir),
                            batch_size=128,seqs = False)
testset = mpra_model.make_dataset(data_dir, 'test', mpra_model.load_stats(data_dir),
                            batch_size=128,seqs = False)

# %%
file = h5py.File('/home/ztang/multitask_RNA/data/lenti_MPRA_embed/gpn_'+cell_type+'.h5','r')
seq = file['seq'][()]
target = file['mean'][()]
x_train,x_test,y_train,y_test=model_selection.train_test_split(seq,target,random_state=42,test_size=0.1)
x_train,x_valid,y_train,y_valid = model_selection.train_test_split(x_train,y_train,random_state=42,test_size=0.1)
with tf.device("CPU"):
        trainset = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(256*4).batch(256)
        validset = tf.data.Dataset.from_tensor_slices((x_valid,y_valid)).shuffle(256*4).batch(256)
        testset = tf.data.Dataset.from_tensor_slices((x_test,y_test)).shuffle(256*4).batch(256)

# %%
model = mpra_model.rep_cnn((230,512),cnn_config)

# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=cnn_config['l_rate'])
loss = tf.keras.losses.MeanSquaredError()
model.compile(optimizer=optimizer,
                loss=loss,
                metrics=['mse'])
earlyStopping_callback = tf.keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True
    )
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2,
        patience=5, min_lr=1e-8)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                    '/home/ztang/multitask_RNA/model/lenti_MPRA_embed/'+cell_type+'/gpn.h5',
                                    monitor='val_loss',
                                    save_best_only=True,
                                    mode = 'min',
                                    save_freq='epoch',)
model.fit(
        trainset,
        epochs=100,
        batch_size=512,
        shuffle=True,
        validation_data = validset,
        callbacks=[earlyStopping_callback,reduce_lr
                   ,checkpoint
                    #,TuneReportCallback({"loss": "loss","val_loss":'val_loss'})
                    ]
    )

# %%
pred_y = model.predict(x_test)

# %%
pred_y = []
y_test = []
for i,(x,y) in enumerate(testset):
    pred_y.extend(model.predict(x))
    y_test.extend(y)

# %%
scipy.stats.pearsonr(np.squeeze(pred_y),np.squeeze(y_test))


