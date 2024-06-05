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
from sklearn.linear_model import Ridge, LinearRegression
import scipy.stats
import sys
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES']='2'

def seq_to_token(sequence,tokenizer):
    N,  = sequence.shape
    seq_pair = []
    for i in range(N):
        seq = sequence[i].decode()
        token_out = tokenizer.batch_tokenize([seq])
        token_id = [b[1] for b in token_out]
        seq_pair.append(np.squeeze(token_id))
    return seq_pair

## GPU Setting 
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

#Dataset and LM model choice
datalen = 230
cell_name = sys.argv[1]
model_name = '2B5_1000G'

if '2B5' in model_name:
    print('2B5_model')
    max_layer = 32
else:
    print('500M model')
    max_layer = 24

#Hyperparameter for the downstream CNN model
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

#Load lentiMPRA sequences
data_file = h5py.File('../data/lenti_MPRA/'+cell_name+'_data.h5','r')
train_seq = data_file['seq_train']
y_train = data_file['y_train'][:]
valid_seq = data_file['seq_valid']
y_valid = data_file['y_valid'][:]
test_seq = data_file['seq_test']
y_test = np.squeeze(data_file['y_test'][:])
#Load Tokenizer
max_len = math.ceil(datalen/6)+2
parameters, forward_fn, tokenizer, config = get_pretrained_model(
    model_name=model_name,
    embeddings_layers_to_save=(),
    attention_maps_to_save=(),
    max_positions=max_len,
)

#Convert sequences
train_tk = seq_to_token(train_seq,tokenizer)
train_size = len(train_tk)
valid_tk = seq_to_token(valid_seq,tokenizer)
valid_size = len(valid_tk)
test_tk = seq_to_token(test_seq,tokenizer)
test_size = len(test_tk)

#Model training for representation of each layer
random_key = jax.random.PRNGKey(0)
batch_size = 3072
alpha = 1e-3
#store test set performance for all models
cnn_test_score = []
mlp_cls = []
mlp_embed = []
rr_cls = []
rr_embed = []
lr_cls = []
lr_embed = []

seq = np.concatenate((train_tk,valid_tk,test_tk))

for layer_i in range(1,max_layer+1):
    print('processing layer %d' % layer_i)
    parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name=model_name,
        embeddings_layers_to_save=(layer_i,),
        attention_maps_to_save=(),
        max_positions=max_len,
    )
    forward_fn = hk.transform(forward_fn)
    cls_embed = []
    mean_embed = []

    temp_embed_file = './temp_nt_embedding.h5'
    output_f = h5py.File(temp_embed_file,'a')
    for dataset in ['train','valid','test']:
        seq = eval(dataset+'_tk')
        output_f.create_dataset(name='x_'+dataset,shape=(len(seq),max_len,2560))
        for i in tqdm(range(0, len(seq), batch_size)):
            seq_batch = jnp.asarray(seq[i:i+batch_size],dtype=jnp.int32)
            outs = forward_fn.apply(parameters, random_key, seq_batch)
            total_embed = np.array(outs['embeddings_'+str(layer_i)])
            output_f['x_'+dataset][i:i+len(total_embed)] = total_embed
            cls_embed.extend(np.squeeze(outs['embeddings_'+str(layer_i)][:,:1, :]))
            mean_embed.extend(np.sum(outs['embeddings_'+str(layer_i)][:,1:,:], axis=1) / (outs['embeddings_'+str(layer_i)].shape[1]-1))

    print('clear memory')
    del outs,parameters,seq_batch
    cls_embed = np.array(cls_embed)
    mean_embed = np.array(mean_embed)
    
    #CNN model and report test set performance
    model = mpra_model.rep_cnn(cnn_config['input_shape'],cnn_config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=cnn_config['l_rate'])
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                    '../model/lenti_MPRA/nt_'+cell_name+'_layer'+str(layer_i)+'.h5',
                                    monitor='val_loss',
                                    save_best_only=True,
                                    mode = 'min',
                                    save_freq='epoch',)
    model.compile(optimizer=optimizer,
                    loss='mean_squared_error',
                    metrics=['mse'])
    earlyStopping_callback = tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2,
            patience=5, min_lr=1e-8)
  
    class h5_generator:
        def __init__(self,file,dataset,y):
            self.file = file[dataset]
            self.y = y
        def __call__(self):
            i=0
            for seq in self.file:
                yield seq,self.y[i]
                i+=1

    trainset = tf.data.Dataset.from_generator(
                            h5_generator(output_f,'x_train',y_train),
                            output_types=(tf.float32,tf.float32),
                            output_shapes = ((41,2560),(1,)),
                            ).shuffle(256*4).batch(256)
    validset = tf.data.Dataset.from_generator(
                            h5_generator(output_f,'x_valid',y_valid),
                            output_types=(tf.float32,tf.float32),
                            output_shapes = ((41,2560),(1,)),
                            ).shuffle(256*4).batch(256)
    x_test = output_f['x_test']

    
    print('CNN training ...')
    model.fit(
            trainset,
            epochs=100,
            batch_size=256,
            shuffle=True,
            validation_data=validset,
            callbacks=[earlyStopping_callback,reduce_lr,checkpoint],
            verbose=2,)
    y_pred = model.predict(x_test)
    cnn_score = scipy.stats.pearsonr(np.squeeze(y_pred),y_test)[0]
    print('##################CNN score: %f###########'%cnn_score)
    cnn_test_score.append(cnn_score)
    del(total_embed) 
    del(model)
    output_f.close()
    os.remove(temp_embed_file)
    tf.keras.backend.clear_session()

    #Linear regression for CLS/mean embedding
    print('linear regression running for CLS/mean embedding...')
    cls_model = LinearRegression().fit(cls_embed[:train_size], y_train)
    lr_cls.append(scipy.stats.pearsonr(np.squeeze(cls_model.predict(cls_embed[train_size+valid_size:])), y_test)[0])
    embed_model = LinearRegression().fit(mean_embed[:train_size],y_train)
    lr_embed.append(scipy.stats.pearsonr(np.squeeze(embed_model.predict(mean_embed[train_size+valid_size:])),y_test)[0])

    #Ridge regression for CLS/mean embedding
    print('ridge regression running with alpha = %f...' % alpha)
    cls_model = Ridge(alpha).fit(cls_embed[:train_size],y_train)
    rr_cls.append(scipy.stats.pearsonr(np.squeeze(cls_model.predict(cls_embed[train_size+valid_size:])),y_test)[0])
    embed_model = Ridge(alpha).fit(mean_embed[:train_size], y_train)
    rr_embed.append(scipy.stats.pearsonr(np.squeeze(embed_model.predict(mean_embed[train_size+valid_size:])),y_test)[0])

    #MLP for CLS/mean embedding
    print('MLP for CLS training...')
    model = mpra_model.rep_mlp(cls_embed[0].shape[0])
    optimizer = tf.keras.optimizers.Adam(learning_rate=cnn_config['l_rate'])
    earlyStopping_callback = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=1e-8)
    model.compile(optimizer=optimizer,
                    loss='mean_squared_error',
                    metrics=['mse'])
    model.fit(
            cls_embed[:train_size],y_train,
            epochs=100,
            batch_size=512,
            shuffle=True,
            validation_data=(cls_embed[train_size:train_size+valid_size],
                            y_valid),
            callbacks=[earlyStopping_callback,reduce_lr],
            verbose=2,)
    y_pred = model.predict(cls_embed[train_size+valid_size:])
    cls_score = scipy.stats.pearsonr(np.squeeze(y_pred),y_test)[0]
    mlp_cls.append(cls_score)
    del(model)
    tf.keras.backend.clear_session()
    
    print('MLP for mean embedding training...')
    model = mpra_model.rep_mlp(mean_embed[0].shape[0])
    optimizer = tf.keras.optimizers.Adam(learning_rate=cnn_config['l_rate'])
    earlyStopping_callback = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=1e-8)
    model.compile(optimizer=optimizer,
                    loss='mean_squared_error',
                    metrics=['mse'])
    model.fit(
            mean_embed[:train_size],y_train,
            epochs=100,
            batch_size=512,
            shuffle=True,
            validation_data = (mean_embed[train_size:train_size+valid_size],
                                y_valid),
            callbacks=[earlyStopping_callback,reduce_lr],
            verbose=2,)
    y_pred = model.predict(mean_embed[train_size+valid_size:])
    cls_score = scipy.stats.pearsonr(np.squeeze(y_pred),y_test)[0]
    mlp_embed.append(cls_score)
    del(model)
    tf.keras.backend.clear_session()

## Save results to csv
df = pd.DataFrame(list(zip(cnn_test_score, mlp_cls,mlp_embed,rr_cls,rr_embed,lr_cls,lr_embed)),
            columns =['CNN', 'CLS-MLP','Mean-embed-MLP',
                        'CLS-Ridge','Mean-embed-Ridge',
                        'CLS-Linear','Mean-embed-Linear'])

df.to_csv('./results/NT_layer_prob_%s.csv'%cell_name)


