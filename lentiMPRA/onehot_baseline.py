# %% [markdown]
# ## Train Model

# %%
import h5py
import sys
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import sys
sys.path.append('/home/ztang/multitask_RNA/evaluation/rep_learning/')
import mpra_model
from sklearn import model_selection
import scipy.stats
celltype = 'HepG2'
file = '/home/ztang/multitask_RNA/data/lenti_MPRA/'+celltype+'_onehot.h5'
save_model = '/home/ztang/multitask_RNA/model/lenti_MPRA_onehot/'+celltype+'/base_CNN_model.h5'

# %%
f = h5py.File(file, 'r')
x = f['onehot'][()]
y = f['target'][()]
x = np.swapaxes(x,1,2)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1,random_state=42)

# %%
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
    'dense2':256,
    'l_rate':0.0001
}
#Select which baseline model to use
model=mpra_model.rep_cnn((230,4),cnn_config)
#model = mpra_model.ResNet((230,4),1)

# %%

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

# %%
result = model.fit(x_train,y_train,
        batch_size=128,
        validation_split=0.1,
        epochs=100,
        shuffle=True,
        verbose=2,
        callbacks=[earlyStopping_callback,checkpoint,reduce_lr],
    )


# %%
import matplotlib.pyplot as plt
plt.plot(result.history['val_loss'])

# %%
y_pred = model.predict(x_test)
print(scipy.stats.pearsonr(np.squeeze(y_pred),y_test))
