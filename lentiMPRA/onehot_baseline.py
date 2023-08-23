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
model = mpra_model.MPRAnn((None,230,4),(None,1))

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
scipy.stats.pearsonr(np.squeeze(y_pred),y_test)

# %%


# %% [markdown]
# ## Test model on corresponding CAGI

# %%
import tensorflow as tf
from tensorflow import keras
import h5py
import scipy.stats
import numpy as np
import os
import pandas as pd
from sklearn import model_selection
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
celltype = 'HepG2'

# %%
model = keras.models.load_model('/home/ztang/multitask_RNA/model/lenti_MPRA_onehot/'+celltype+'/ResNet.h5')

# %%
f = h5py.File('/home/ztang/multitask_RNA/data/lenti_MPRA/'+celltype+'_onehot.h5', 'r')
x = f['onehot'][()]
y = f['target'][()]
x = np.swapaxes(x,1,2)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1,random_state=42)
model.evaluate(x_test,y_test)

# %%
y_pred = model.predict(x_test)
scipy.stats.pearsonr(np.squeeze(y_pred),y_test)

# %%
file = h5py.File("/home/ztang/multitask_RNA/data/CAGI/"+celltype+"/onehot.h5", "r")
alt = file['alt']
ref = file['ref']
alt_pred = model.predict(alt)
ref_pred = model.predict(ref)
pred = alt_pred - ref_pred
pred.shape

# %%
exp_df = pd.read_csv('/home/ztang/multitask_RNA/data/CAGI/'+celltype+'/metadata.csv')
target = exp_df['6'].values.tolist()

# %%
start_idx=0
perf = []
for exp in exp_df['8'].unique():
    sub_df = exp_df[exp_df['8'] == exp]
    exp_target = np.array(target)[sub_df.index.to_list()]
    exp_pred = np.squeeze(pred)[sub_df.index.to_list()]
    print(exp)
    perf.append(scipy.stats.pearsonr(exp_pred,exp_target)[0])
    print(scipy.stats.pearsonr(exp_pred,exp_target)[0])

# %%
np.mean(perf)

# %%


# %% [markdown]
# ## Test Model on all CAGI

# %%
import tensorflow as tf
from tensorflow import keras
import h5py
import scipy.stats
import numpy as np
import os
import pandas as pd
from sklearn import model_selection
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
celltype = 'HepG2'

# %%
model = keras.models.load_model('/home/ztang/multitask_RNA/model/lenti_MPRA_onehot/'+celltype+'/model.h5')

# %%
file = h5py.File("/home/ztang/multitask_RNA/data/CAGI/230/CAGI_onehot.h5", "r")
alt = file['alt']
ref = file['ref']
alt_pred = model.predict(alt)
ref_pred = model.predict(ref)
pred = alt_pred - ref_pred
pred.shape

# %%
cagi_df = pd.read_csv('../../data/CAGI/230/final_cagi_metadata.csv',
                      index_col=0).reset_index()
exp_list = cagi_df['8'].unique()
target = cagi_df['6'].values.tolist()

# %%
import scipy.stats as stats
perf = []
sanity_check = 0
for exp in cagi_df['8'].unique():
    sub_df = cagi_df[cagi_df['8'] == exp]
    sanity_check += len(sub_df)
    exp_target = np.array(target)[sub_df.index.to_list()]
    exp_pred = np.squeeze(pred)[sub_df.index.to_list()]
    print(exp)
    perf.append(stats.pearsonr(exp_pred,exp_target)[0])
    print(stats.pearsonr(exp_pred,exp_target)[0])

# %%
np.mean(perf)

# %%



