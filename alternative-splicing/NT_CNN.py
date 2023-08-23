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

if len(sys.argv) > 1:
    sub_ratio = float(sys.argv[1])
else:
    sub_ratio = 1

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
  rep_list = {}
  rep_list['Tissue'] = mt_preprocess.a_tissues

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
  for label in ['test','valid', 'train']:
    #load onehot data
    sequence = file['x_'+label][()]
    target = file['y_'+label][()]
    if sub_ratio != 1:
      idx = np.random.choice(range(len(sequence)),size = int(sub_ratio*len(sequence)),replace = False)
      sequence = sequence[idx]
      target = target[idx]
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
    l_embed = []
    r_embed = []
    for i in tqdm(range(0, len(l_seq_pair), batch_size)):
      l_seq_batch = jnp.asarray(l_seq_pair[i:i+batch_size,:],dtype=jnp.int32)
      r_seq_batch = jnp.asarray(r_seq_pair[i:i+batch_size,:],dtype=jnp.int32)
      l_outs = forward_fn.apply(parameters, random_key, l_seq_batch)
      l_embed.extend(np.array(l_outs['embeddings_'+str(layer)]))
      r_outs = forward_fn.apply(parameters, random_key, r_seq_batch)
      r_embed.extend(np.array(r_outs['embeddings_'+str(layer)]))
    #create TF dataset from embeddings          
    with tf.device("CPU"):
        if label == 'test':
          test_xl = l_embed
          test_xr = r_embed
          test_y = v_y
        else:
          dataset[label] = tf.data.Dataset.from_tensor_slices(
                            ({"input_1": l_embed, "input_2": r_embed},target)).shuffle(256*4).batch(256)

#model training
  print('start training')
  for i in range(5):
    model = custom_model.mtsplice_embed_model((71,2560),56)
    loss = custom_model.diff_KL()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    if sub_ratio == 1:
      checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                              '/home/ztang/multitask_RNA/model/mtsplice_NT/'+str(layer)+'_model'+str(i)+'.h5',
                                              monitor='val_loss',
                                              save_best_only=True,
                                              mode = 'min',
                                              save_freq='epoch',)
    else:
      checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                              '/home/ztang/multitask_RNA/model/mtsplice_NT/'+str(layer)+'_model'+str(i)+'_'+str(sub_ratio)+'.h5',
                                              monitor='val_loss',
                                              save_best_only=True,
                                              mode = 'min',
                                              save_freq='epoch',)
    model.compile(loss = loss,
                optimizer=optimizer)
    result = model.fit(dataset['train'],
                batch_size=256,
                validation_data=dataset['valid'],
                epochs=100,
                verbose=0,
                callbacks=[earlyStopping_callback,reduce_lr,checkpoint]
                )
    
    p_y = model.predict([np.array(test_xl),np.array(test_xr)])
    print(p_y.shape)

    #clear session
    tf.keras.backend.clear_session()

    corr_list = []
    for a in tqdm(range(0,56)):
        corr,_ = scipy.stats.spearmanr(v_y[:,a,0],p_y[:,a],nan_policy='omit')
        corr_list.append(corr)
        #print(mt_preprocess.a_tissues[i],' PR: ',corr)

    rep_list['rep' + str(i)] = corr_list

  if sub_ratio != 1:
    path = '/home/ztang/multitask_RNA/replications/MTSplice/result/' + str(sub_ratio)+'/'
  else:
    path = '/home/ztang/multitask_RNA/replications/MTSplice/result/'
  if os.path.exists(path) == False:
    os.makedirs(path)

  perf_df = pd.DataFrame.from_dict(rep_list)
  perf_df.to_csv(path+'NT'+str(layer)+'_CNN_perf.csv')
    