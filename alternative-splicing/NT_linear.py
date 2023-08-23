import pandas as pd
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from nucleotide_transformer.pretrained import get_pretrained_model
from tqdm import tqdm
import scipy.stats
import sys
sys.path.append('/home/ztang/multitask_RNA/data_generation/mtsplice/')
sys.path.append('/home/ztang/multitask_RNA/data_generation')
import utils
import h5py
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import Ridge
import glob
import os
import mt_preprocess
import custom_model
from sklearn.linear_model import LinearRegression, Ridge
#memory settings
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

model_name = '2B5_1000G'
batch_size = 128
datalen = 200


#Read dataset
file = h5py.File('/home/ztang/multitask_RNA/data/mtsplice/delta_logit.h5','r')
x_test = file['x_test'][()]
y_test = file['y_test'][()]

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

#model callbacks
earlyStopping_callback = tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True
        )
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2,
            patience=5, min_lr=1e-6)

for layer in (32,10):
  #initialize perf df
  cls_list = {}
  cls_list['Tissue'] = mt_preprocess.a_tissues
  mean_list = {}
  mean_list['Tissue'] = mt_preprocess.a_tissues

  #NT model
  max_len = 400//6 + 400%6 + 1
  random_key = jax.random.PRNGKey(0)
  parameters, forward_fn, tokenizer, config = get_pretrained_model(
                  model_name=model_name,
                  mixed_precision=False,
                  embeddings_layers_to_save=(layer,),
                  attention_maps_to_save=(),
                  max_positions=max_len
              )
  forward_fn = hk.transform(forward_fn)

#dataset creation
  dataset = {}
  for label in ['test','valid','train']:
    #load onehot data
    sequence = file['x_'+label][()]
    target = file['y_'+label][()]
    if label == 'test':
      sequence = v_x
      target = v_y
    sequence = utils.onehot_to_seq(sequence)
    sequence = np.array(sequence)
    seq_l = []
    seq_r = []
    for seq in sequence:
       seq_l.append(seq[:400])
       seq_r.append(seq[400:])

    #tokenization
    l_token_out = tokenizer.batch_tokenize(seq_l)
    r_token_out = tokenizer.batch_tokenize(seq_r)
    l_token_id = [b[1] for b in l_token_out]
    r_token_id = [b[1] for b in r_token_out]
    l_seq_pair= np.squeeze(l_token_id)
    r_seq_pair = np.squeeze(r_token_id)

    #Embedding extraction
    cls = []
    mean = []
    for i in tqdm(range(0, len(l_seq_pair), batch_size)):
      l_seq_batch = jnp.asarray(l_seq_pair[i:i+batch_size,:],dtype=jnp.int32)
      r_seq_batch = jnp.asarray(r_seq_pair[i:i+batch_size,:],dtype=jnp.int32)
      l_outs = forward_fn.apply(parameters, random_key, l_seq_batch)
      r_outs = forward_fn.apply(parameters, random_key, r_seq_batch)
      l_outs= np.array(l_outs['embeddings_'+str(layer)])
      r_outs= np.array(r_outs['embeddings_'+str(layer)])
      
      #batch_size *2560
      temp_cls = np.concatenate([l_outs[:,1,:],r_outs[:,1,:]],axis=-1)
      l_outs = l_outs[:, 1:, :]  # removing CLS token
      r_outs = r_outs[:, 1:, :]
      temp_mean = np.concatenate([(np.sum(l_outs, axis=1) / l_outs.shape[1]),
                                  (np.sum(r_outs, axis=1) / r_outs.shape[1])],axis=-1)
      
      #append to dataset list
      cls.extend(temp_cls)
      mean.extend(temp_mean)
    #save train/test set
    dataset[label] = [np.array(cls),np.array(mean),target]

  #model training
  print('start training')
  for i in range(5):

  #CLS MLP
    model = custom_model.linear_model(dataset['train'][0][0].shape,56)
    loss = loss = custom_model.diff_KL()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                            '/home/ztang/multitask_RNA/model/mtsplice_NT/'+str(layer)+'_Mean_linear'+str(i)+'.h5',
                                            monitor='val_loss',
                                            save_best_only=True,
                                            mode = 'min',
                                            save_freq='epoch',)
    model.compile(optimizer=optimizer,loss=loss)
    result = model.fit(dataset['train'][0],dataset['train'][-1],
                batch_size=256,
                validation_data=(dataset['valid'][0],dataset['valid'][-1]),
                epochs=100,
                verbose=0,
                callbacks=[earlyStopping_callback,reduce_lr,checkpoint]
                )
    #evaluate
    p_y = model.predict(dataset['test'][0])
    tf.keras.backend.clear_session()
    cls_list['rep'+ str(i)] = custom_model.mt_evaluate(dataset['test'][-1],p_y)
  
  #Mean MLP
    model = custom_model.linear_model(dataset['train'][1][0].shape,56)
    loss = loss = custom_model.diff_KL()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                            '/home/ztang/multitask_RNA/model/mtsplice_NT/'+str(layer)+'_CLS_linear'+str(i)+'.h5',
                                            monitor='val_loss',
                                            save_best_only=True,
                                            mode = 'min',
                                            save_freq='epoch',)
    model.compile(optimizer=optimizer,loss=loss)
    result = model.fit(dataset['train'][1],dataset['train'][-1],
                batch_size=256,
                validation_data=(dataset['valid'][0],dataset['valid'][-1]),
                epochs=100,
                verbose=0,
                callbacks=[earlyStopping_callback,reduce_lr,checkpoint]
                )
    p_y = model.predict(dataset['test'][1])
    mean_list['rep'+ str(i)] = custom_model.mt_evaluate(dataset['test'][-1],p_y)
    tf.keras.backend.clear_session()

  perf_df = pd.DataFrame.from_dict(cls_list)
  perf_df.to_csv('/home/ztang/multitask_RNA/replications/MTSplice/result/NT'+str(layer)+'_CLS_Linear_perf.csv')

  perf_df = pd.DataFrame.from_dict(mean_list)
  perf_df.to_csv('/home/ztang/multitask_RNA/replications/MTSplice/result/NT'+str(layer)+'_Mean_Linear_perf.csv')
