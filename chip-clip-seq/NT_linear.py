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
from sklearn import metrics
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
file_list = glob.glob('/home/ztang/multitask_RNA/data/eclip/*_200.h5')


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
    model_list = []
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
        for label in ['test', 'valid','train']:
            #tokenization
            sequence = data['X_'+label][()]
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

            cls_dataset[label] = np.squeeze(cls)
            mean_dataset[label] = np.squeeze(mean_embed)
    
        cls_dataset['train'] = np.concatenate((cls_dataset['train'],cls_dataset['valid']))
        target = np.concatenate((data['Y_train'][()],data['Y_valid'][()]))
        test_target = np.squeeze(data['Y_test'][()])
        mean_dataset['train'] = np.concatenate((mean_dataset['train'],mean_dataset['valid']))

        print('########Training ',tf_name, '########')
        ## Logisitic regression
        print(cls_dataset['train'].shape)
        print( np.squeeze(target).shape)
        cls_model = LogisticRegression(random_state=0).fit(cls_dataset['train'], np.squeeze(target))
        mean_model = LogisticRegression(random_state=0).fit(mean_dataset['train'],np.squeeze(target))
        ## Pred on test set
        cls_predict = cls_model.predict(cls_dataset['test'])
        mean_predict = mean_model.predict(mean_dataset['test'])
        ## Append CLS model performance
        tf_list.append(tf_name)
        model_list.append('NT'+str(layer)+' CLS logistic regression')
        test_accuracy.append(metrics.accuracy_score(test_target,cls_predict))
        test_auroc.append(metrics.roc_auc_score(test_target,cls_predict))
        test_aupr.append(metrics.average_precision_score(test_target,cls_predict))
        ## Append Mean embed model performance
        tf_list.append(tf_name)
        model_list.append('NT'+str(layer)+'Mean embed logistic regression')
        test_accuracy.append(metrics.accuracy_score(test_target,mean_predict))
        test_auroc.append(metrics.roc_auc_score(test_target,mean_predict))
        test_aupr.append(metrics.average_precision_score(test_target,mean_predict))

    
    df = pd.DataFrame(list(zip(tf_list, test_accuracy, test_auroc, test_aupr,model_list)),
               columns =['TF','Accuracy','AUROC','AUPR','Model'])
    df.to_csv('/home/ztang/multitask_RNA/evaluation/chip/result/eclip_result/NT'+str(layer)+'_logistic_perf.csv')


