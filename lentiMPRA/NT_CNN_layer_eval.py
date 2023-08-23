# %%
import pandas as pd
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from nucleotide_transformer.pretrained import get_pretrained_model
from tqdm import tqdm
import math
import h5py
import mpra_model
import tensorflow as tf
from sklearn.linear_model import Ridge
import scipy.stats
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

datalen = 230
cell_name = 'HepG2'
model_name = '2B5_1000G'

if '2B5' in model_name:
    print('2B5_model')
    max_layer = 32
else:
    print('500M model')
    max_layer = 24

cnn_config = {
    'input_shape': (41,2560),
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

# %% [markdown]
# ## Linear regression over lenti-MPRA

# %%
data_file = h5py.File('/home/ztang/multitask_RNA/data/lenti_MPRA/'+cell_name+'_data.h5', 'r')
sequence = data_file['seq'][()]
target = data_file['mean'][()]


# %%
max_len = math.ceil(datalen/6)+2
parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name=model_name,
        mixed_precision=False,
        embeddings_layers_to_save=(),
        attention_maps_to_save=(),
        max_positions=max_len,
    )

# %%
N,  = sequence.shape
seq_pair = []
for i in range(N):
    seq = sequence[i].decode()
    token_out = tokenizer.batch_tokenize([seq])
    token_id = [b[1] for b in token_out]
    seq_pair.append(np.squeeze(token_id))
#seq_pair = jnp.asarray(seq_pair,dtype=jnp.int32)

# %%
batch_size = 1024
alpha = [1e-3]
cnn_test_score = []
mlp_cls = []
mlp_embed = []
rr_cls = {}
rr_embed = {}
for a in alpha:
    rr_cls[a] = []
    rr_embed[a] = []

max_len = math.ceil(datalen/6)+2
random_key = jax.random.PRNGKey(0)
for layer_i in range(1,max_layer+1):    
#for layer_i in range(1,2):
    parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name=model_name,
        mixed_precision=False,
        embeddings_layers_to_save=(layer_i,),
        attention_maps_to_save=(),
        max_positions=max_len,
    )
    forward_fn = hk.transform(forward_fn)
    cls = []
    mean_embed = []
    total_embed = []

    for i in tqdm(range(0, N, batch_size)):
        seq_batch = jnp.asarray(seq_pair[i:i+batch_size],dtype=jnp.int32)
        outs = forward_fn.apply(parameters, random_key, seq_batch)
        #embed = np.array(outs['embeddings_'+str(layer_i)])
        total_embed.extend(np.array(outs['embeddings_'+str(layer_i)]))
        cls.extend(np.squeeze(outs['embeddings_'+str(layer_i)][:,:1, :]))
        #embed = embed[:, 1:, :]  # removing CLS token
        mean_embed.extend(np.sum(outs['embeddings_'+str(layer_i)][:,1:,:], axis=1) / (outs['embeddings_'+str(layer_i)].shape[1]-1))
    
    print('clear memory')
    del(outs)
    del(parameters)
    del(seq_batch)

    #CLS/mean embedding/complete embedding train-test split
    print('Parsing data')
    cls = np.array(cls)
    mean_embed = np.array(mean_embed)
    total_embed = np.array(total_embed)
    index = np.random.permutation(len(target))
    train_index = index[:int(0.9*len(target))]
    test_index = index[int(0.9*len(target)):]
    
    #CNN model and report test set performance
    #print('CNN training ...')
    model = mpra_model.rep_cnn(cnn_config['input_shape'],cnn_config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=cnn_config['l_rate'])
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                    '/home/ztang/multitask_RNA/model/lenti_MPRA_embed/'+cell_name+'/layer_'+str(layer_i)+'.h5',
                                    monitor='val_loss',
                                    save_best_only=True,
                                    mode = 'min',
                                    save_freq='epoch',)
    model.compile(optimizer=optimizer,
                    loss='mean_squared_error',
                    metrics=['mse'])
    earlyStopping_callback = tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True
        )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2,
            patience=5, min_lr=1e-8)
    ##TF dataset instead of numpy arrays? Does this help memory?
    cnn_train_index = train_index[:int(0.9*len(train_index))]
    cnn_valid_index = train_index[int(0.9*len(train_index)):]

    print('data transfer to TF Datatset')
    with tf.device("CPU"):
        trainset = tf.data.Dataset.from_tensor_slices((total_embed[cnn_train_index],target[cnn_train_index])).shuffle(256*4).batch(256)
        validset = tf.data.Dataset.from_tensor_slices((total_embed[cnn_valid_index],target[cnn_valid_index])).shuffle(256*4).batch(256)
   
    print('CNN training ...')
    model.fit(
            #total_embed[train_index],target[train_index],
            trainset,
            epochs=100,
            batch_size=256,
            shuffle=True,
            validation_data=validset,
            callbacks=[earlyStopping_callback,reduce_lr,checkpoint],
            verbose=2,)
    y_pred = model.predict(total_embed[test_index])
    cnn_score = scipy.stats.pearsonr(np.squeeze(y_pred),target[test_index])[0]
    cnn_test_score.append(cnn_score)
    del(total_embed) 
    del(model)
    tf.keras.backend.clear_session()

    #ridge regression for CLS/mean embedding with different factor
    for a in alpha:
        print('regression running with alpha = %f' % a)
        cls_model = Ridge(a).fit(cls[train_index], target[train_index])
        rr_cls[a].append(cls_model.score(cls[test_index], target[test_index]))
        embed_model = Ridge(a).fit(mean_embed[train_index], target[train_index])
        rr_embed[a].append(embed_model.score(mean_embed[test_index], target[test_index]) )

    #MLP for CLS/mean embedding
    ##CLS
    print('MLP for CLS training...')
    model = mpra_model.rep_mlp(cls.shape[1])
    optimizer = tf.keras.optimizers.Adam(learning_rate=cnn_config['l_rate'])
    earlyStopping_callback = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=1e-8)
    model.compile(optimizer=optimizer,
                    loss='mean_squared_error',
                    metrics=['mse'])
    model.fit(
            cls[train_index],target[train_index],
            epochs=100,
            batch_size=512,
            shuffle=True,
            validation_split=0.1,
            callbacks=[earlyStopping_callback,reduce_lr],
            verbose=2,)
    y_pred = model.predict(cls[test_index])
    cls_score = scipy.stats.pearsonr(np.squeeze(y_pred),target[test_index])[0]
    mlp_cls.append(cls_score)
    del(model)
    tf.keras.backend.clear_session()

    ##embed
    print('MLP for mean embedding training...')
    model = mpra_model.rep_mlp(mean_embed.shape[1])
    optimizer = tf.keras.optimizers.Adam(learning_rate=cnn_config['l_rate'])
    earlyStopping_callback = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=1e-8)
    model.compile(optimizer=optimizer,
                    loss='mean_squared_error',
                    metrics=['mse'])
    model.fit(
            mean_embed[train_index],target[train_index],
            epochs=100,
            batch_size=512,
            shuffle=True,
            validation_split=0.1,
            callbacks=[earlyStopping_callback,reduce_lr],
            verbose=2,)
    y_pred = model.predict(mean_embed[test_index])
    cls_score = scipy.stats.pearsonr(np.squeeze(y_pred),target[test_index])[0]
    mlp_embed.append(cls_score)
    del(model)
    tf.keras.backend.clear_session()

## Save results to csv
df = pd.DataFrame(list(zip(cnn_test_score, mlp_cls,mlp_embed)),
               columns =['CNN', 'MLP for CLS','MLP for mean embed' ])
for a in alpha:
    df['CLS RR'+str(a)] = np.array(rr_cls[a])
    df['Mean Embedding RR'+str(a)] = np.array(rr_embed[a])

# %%
df.to_csv('/home/ztang/multitask_RNA/evaluation/rep_learning/'+cell_name+'_'+model_name+'_layersearch.csv')

# %%



