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
from sklearn.linear_model import LinearRegression

model_name = '2B5_1000G'
if '2B5' in model_name:
    print('2B5_model')
    max_layer = 32
else:
    print('500M model')
    max_layer = 24

datalen = 230
cell_name = 'HepG2'

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
seq_pair = jnp.asarray(seq_pair,dtype=jnp.int32)

# %%
batch_size = 3072
cls_test_score = []
embed_test_score = []

max_len = math.ceil(datalen/6)+2
random_key = jax.random.PRNGKey(0)
for layer_i in range(1,max_layer+1):    
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

    for i in range(0, N, batch_size):
        seq_batch = seq_pair[i:i+batch_size]
        outs = forward_fn.apply(parameters, random_key, seq_batch)
        embed = np.array(outs['embeddings_'+str(layer_i)])
        cls.extend(np.squeeze(embed[:,:1, :]))
        embed = embed[:, 1:, :]  # removing CLS token
        mean_embed.extend(np.sum(embed, axis=1) / embed.shape[1])

    cls = np.array(cls)
    mean_embed = np.array(mean_embed)
    index = np.random.permutation(len(target))
    train_index = index[:int(0.9*len(target))]
    test_index = index[int(0.9*len(target)):]
    cls_model = LinearRegression().fit(cls[train_index], target[train_index])
    cls_test_score.append(scipy.stats.pearsonr(cls_model.predict(cls[test_index]), target[test_index])[0])
    embed_model = LinearRegression().fit(mean_embed[train_index], target[train_index])
    embed_test_score.append(scipy.stats.pearsonr(embed_model.predict(mean_embed[test_index]), target[test_index])[0])

    print('Performance on layer %d:' % (layer_i))
    print(cls_test_score[-1])
    print(embed_test_score[-1])

# %%
df = pd.DataFrame(list(zip(cls_test_score, embed_test_score)),
               columns =['CLS regression score', 'Mean Embedding regression score'])

# %%
df.to_csv('/home/ztang/multitask_RNA/evaluation/rep_learning/LR_'+cell_name+'_2B5_1000G.csv')

# %%



