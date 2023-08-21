import pandas as pd
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from nucleotide_transformer.pretrained import get_pretrained_model
from tqdm import tqdm
import sys
sys.path.append('/home/ztang/multitask_RNA/data_generation')
import utils
import h5py
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import LogisticRegression
import glob
import os
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
batch_size = 32
datalen = 200
file_list = glob.glob('/home/ztang/multitask_RNA/data/eclip/*.h5')

test_aupr = []
test_auroc = []
test_accuracy = []
tf_list = []
model_list = []

#define model and callbacks
def rep_mlp(input_shape,output_shape = 1):
     #initializer
    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005)
    #input layer
    inputs = keras.Input(shape=input_shape, name='sequence')
    nn = keras.layers.Dense(512,kernel_initializer=initializer)(inputs)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    nn = keras.layers.Dense(256,kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    outputs = keras.layers.Dense(output_shape,activation = 'linear',kernel_initializer=initializer)(nn)

    model =  keras.Model(inputs=inputs, outputs=outputs)
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

#last layer and best layer
for layer in (32,10):
    test_aupr = []
    test_auroc = []
    test_accuracy = []
    tf_list = []
    model = []
    #load model with interested embedding layers
    print('load NT model')
    max_len = datalen//6 + datalen%6 + 1
    random_key = jax.random.PRNGKey(0)
    parameters, forward_fn, tokenizer, config = get_pretrained_model(
                    model_name=model_name,
                    mixed_precision=False,
                    embeddings_layers_to_save=(layer,),
                    attention_maps_to_save=(),
                    max_positions=max_len
                )
    forward_fn = hk.transform(forward_fn)
    # per TF,load onehot data and generate embedding for train/valid/test set.
    for file in file_list:
        tf_name = file.split('/')[-1][:-12]
        data = h5py.File(file,'r')
        print('tokenization for '+tf_name)
        cls_dataset = {}
        mean_dataset = {}

        #create dataset for train,valid,test.
        for label in ['test', 'valid', 'train']:
            #tokenization
            sequence = data['X_'+label][:,:4,:]
            sequence = np.transpose(sequence,(0,2,1))
            sequence = utils.onehot_to_seq(sequence)
            target = data['Y_'+label][()]
            token_out = tokenizer.batch_tokenize(sequence)
            token_id = [b[1] for b in token_out]
            seq_pair= np.squeeze(token_id)

            #pass through model for embeddings
            cls = []
            mean_embed = []
            for i in tqdm(range(0, len(seq_pair), batch_size)):
                seq_batch = jnp.asarray(seq_pair[i:i+batch_size,:],dtype=jnp.int32)
                outs = forward_fn.apply(parameters, random_key, seq_batch)
                embed = np.array(outs['embeddings_'+str(layer)])
                cls.extend(embed[:,:1, :])
                embed = embed[:, 1:, :]  # removing CLS token
                mean_embed.extend(np.sum(embed, axis=1) / embed.shape[1])

            #create TF dataset from embeddings          
            with tf.device("CPU"):
                cls_dataset[label] = tf.data.Dataset.from_tensor_slices(
                                    (np.squeeze(cls),data['Y_'+label][()])).shuffle(256*4).batch(256)
                mean_dataset[label] = tf.data.Dataset.from_tensor_slices(
                                    (np.squeeze(mean_embed),data['Y_'+label][()])).shuffle(256*4).batch(256)
    
        print('########Training ',tf_name, '########')
        for i in range(5):
            
            ### CLS as input
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                            '/home/ztang/multitask_RNA/model/eclip_NT/'+'/'+str(layer)+'_'+tf_name+'_CLS.h5',
                                            monitor='val_loss',
                                            save_best_only=True,
                                            mode = 'min',
                                            save_freq='epoch',)
            model = rep_mlp((2560),1)
            model.compile(loss = loss,
                        metrics=['accuracy',auroc,aupr],
                        optimizer=optimizer)
            
            result = model.fit(cls_dataset['train'],
                batch_size=256,
                validation_data=cls_dataset['valid'],
                epochs=100,
                verbose=2,
                callbacks=[earlyStopping_callback,reduce_lr,checkpoint]
            )
            _, acc, roc, pr = model.evaluate(cls_dataset['test'])
            tf_list.append(tf_name)
            model_list.append('NT'+str(layer)+'CLS MLP')
            test_accuracy.append(acc)
            test_auroc.append(roc)
            test_aupr.append(pr)

            ### Mean embedding as input
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                            '/home/ztang/multitask_RNA/model/eclip_NT/'+'/'+str(layer)+'_'+tf_name+'_mean.h5',
                                            monitor='val_loss',
                                            save_best_only=True,
                                            mode = 'min',
                                            save_freq='epoch',)
            model = rep_mlp((2560),1)
            model.compile(loss = loss,
                        metrics=['accuracy',auroc,aupr],
                        optimizer=optimizer)
            
            result = model.fit(mean_dataset['train'],
                batch_size=256,
                validation_data=mean_dataset['valid'],
                epochs=100,
                verbose=2,
                callbacks=[earlyStopping_callback,reduce_lr,checkpoint]
            )
            _, acc, roc, pr = model.evaluate(mean_dataset['test'])
            tf_list.append(tf_name)
            model_list.append('NT'+str(layer)+'Mean embed MLP')
            test_accuracy.append(acc)
            test_auroc.append(roc)
            test_aupr.append(pr)

    
    df = pd.DataFrame(list(zip(tf_list, test_accuracy, test_auroc, test_aupr,model_list)),
               columns =['TF','Accuracy','AUROC','AUPR','Model'])
    df.to_csv('/home/ztang/multitask_RNA/evaluation/chip/result/eclip_result/NT'+str(layer)+'_MLP_perf.csv')


