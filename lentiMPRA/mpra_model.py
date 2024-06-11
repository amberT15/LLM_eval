import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import tensorflow.keras.layers as tfkl
import glob
from natsort import natsorted
import os
import json
def ResNet(input_shape,output_shape=1,**kwargs):
    initializer = keras.initializers.GlorotUniform()

    inputs = keras.Input(input_shape)

    nn = keras.layers.Conv1D(filters=196, kernel_size=19, padding='same', use_bias=True, kernel_initializer=initializer)(inputs)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('silu')(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    #residual block
    nn = residual_block(nn, filter_size=3, dilated=5,activation='silu',kernel_initializer = initializer)
    nn = keras.layers.Dropout(0.2)(nn)
    nn = keras.layers.MaxPooling1D(5)(nn)

    nn = keras.layers.Conv1D(filters=256, kernel_size=7, padding='same', use_bias=True, kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('silu')(nn)
    nn = keras.layers.Dropout(0.2)(nn)
    nn = keras.layers.MaxPooling1D(5)(nn)
    
    nn = keras.layers.Dense(256, use_bias=True, kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('silu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    nn = keras.layers.AveragePooling1D(2)(nn)
    nn = keras.layers.Flatten()(nn)

    nn = keras.layers.Dense(256, use_bias=True, kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('silu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    outputs = keras.layers.Dense(output_shape, activation='linear', kernel_initializer=initializer)(nn)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

def residual_block(input_layer, filter_size, activation='relu', dilated=5, kernel_initializer=None):
    factor = []
    base = 2
    for i in range(dilated):
        factor.append(base**i)
        
    num_filters = input_layer.shape.as_list()[-1]  

    nn = keras.layers.Conv1D(filters=num_filters,
                                    kernel_size=filter_size,
                                    activation=None,
                                    use_bias=False,
                                    padding='same',
                                    dilation_rate=1, kernel_initializer=kernel_initializer,
                                    )(input_layer) 
    nn = keras.layers.BatchNormalization()(nn)
    for f in factor:
        nn = keras.layers.Activation(activation)(nn)
        nn = keras.layers.Dropout(0.1)(nn)
        nn = keras.layers.Conv1D(filters=num_filters,
                                        kernel_size=filter_size,
                                        strides=1,
                                        activation=None,
                                        use_bias=False, 
                                        padding='same',
                                        dilation_rate=f, kernel_initializer=kernel_initializer,
                                        )(nn) 
        nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.add([input_layer, nn])
    return keras.layers.Activation(activation)(nn)
##### tensorflow model MPRAnn #####
def MPRAnn(input_shape,output_shape=1,**kwargs):
    inputs = keras.Input(shape=(input_shape[0], input_shape[1]), name="input")
    layer = keras.layers.Conv1D(250, kernel_size=7, strides=1, activation='relu', name="conv1")(inputs)  # 250 7 relu
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.Conv1D(250, 8, strides=1, activation='softmax', name="conv2")(layer)  # 250 8 softmax
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.MaxPooling1D(pool_size=2, strides=None, name="maxpool1")(layer)
    layer = keras.layers.Dropout(0.1)(layer)
    layer = keras.layers.Conv1D(250, 3, strides=1, activation='softmax', name="conv3")(layer)  # 250 3 softmax
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.Conv1D(100, 2, strides=1, activation='softmax', name="conv4")(layer)  # 100 3 softmax
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.MaxPooling1D(pool_size=1, strides=None, name="maxpool2")(layer)
    layer = keras.layers.Dropout(0.1)(layer)
    layer = keras.layers.Flatten()(layer)
    layer = keras.layers.Dense(300, activation='sigmoid')(layer)  # 300
    layer = keras.layers.Dropout(0.3)(layer)
    layer = keras.layers.Dense(200, activation='sigmoid')(layer)  # 300
    predictions = keras.layers.Dense(output_shape, activation='linear')(layer)
    model = keras.Model(inputs=inputs, outputs=predictions)
    return model

##### tensorflow model representation learning #####
def rep_cnn(input_shape,config={},factor=1):
    #initializer
    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005)
    #input layer
    inputs = keras.Input(shape=input_shape, name='sequence')

    #position wise dense layer for dimension reduction
    nn = keras.layers.BatchNormalization()(inputs)
    nn = keras.layers.Conv1D(filters=config['reduce_dim'],kernel_size=1,
                             kernel_initializer = initializer)(nn)

    #first conv block
    nn = keras.layers.Conv1D(filters=config['conv1_filter']*factor,
                             kernel_size=config['conv1_kernel'],
                             padding='same',
                             kernel_initializer = initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(config['activation'], name='filter_activation')(nn)
    nn = keras.layers.Dropout(config['dropout1'])(nn)
    nn = keras.layers.MaxPooling1D(pool_size=config['res_pool'])(nn)

    #residual block
    # nn = residual_block(nn, filter_size=config['res_filter'], 
    #                     dilated=config['res_layers'],
    #                     kernel_initializer = initializer)
    # nn = keras.layers.MaxPooling1D(pool_size=config['res_pool'])(nn)
    # nn = keras.layers.Dropout(config['res_dropout'])(nn)


    nn = keras.layers.Conv1D(filters=config['conv2_filter']*factor,
                             kernel_size=config['conv2_kernel'],
                             padding='same',
                             kernel_initializer = initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPool1D(pool_size=config['pool2_size'])(nn)
    nn = keras.layers.Dropout(config['dropout2'])(nn)

    #output block
    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(config['dense']*factor,kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    nn = keras.layers.Dense(config['dense2']*factor,kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    outputs = keras.layers.Dense(1,activation = 'linear',kernel_initializer=initializer)(nn)

    model =  keras.Model(inputs=inputs, outputs=outputs)
    return model   

def rep_onehot(input_shape,config={}):
    #initializer
    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005)

    #input layers
    seq_inputs = keras.Input(shape=input_shape[0], name='sequence')
    embed_inputs = keras.Input(shape=input_shape[1], name='embedding')

    #sequence input head
    nn = keras.layers.Conv1D(filters=config['conv_seq_filter'],
                             kernel_size=config['conv_seq_kernel'],
                             padding='same',
                             kernel_initializer = initializer)(seq_inputs)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(config['seq_activation'], name='filter_activation')(nn)
    nn = keras.layers.Dropout(config['seq_dropout'])(nn)   
    nn = residual_block(nn, filter_size=config['res_filters'], 
                        dilated=config['res_layers'],
                        kernel_initializer = initializer)
    nn = keras.layers.MaxPool1D(pool_size=6)(nn)
    nn = keras.layers.Dropout(config['res_dropout'])(nn)   
    nn = keras.layers.ZeroPadding1D(padding=(1,2))(nn)    #pad to same length as embedding

    #embedding input head
    e_nn = keras.layers.BatchNormalization()(embed_inputs)
    e_nn = keras.layers.Conv1D(filters=config['reduce_dim'],kernel_size=1,
                             kernel_initializer = initializer)(e_nn) #dimension reduction
    # e_nn = keras.layers.Conv1D(filters=config['conv_rep_filter'],
    #                          kernel_size=config['conv_rep_kernel'],
    #                          padding='same',
    #                          kernel_initializer = initializer)(e_nn)
    # e_nn = keras.layers.BatchNormalization()(e_nn)
    # e_nn = keras.layers.Activation(config['rep_activation'])(e_nn)
    # e_nn = keras.layers.Dropout(config['rep_dropout'])(e_nn)

    #concatenation
    nn = keras.layers.Concatenate()([nn,e_nn])

    #conv block
    nn = keras.layers.Conv1D(filters=config['conv2_filter'],
                             kernel_size=config['conv2_kernel'],
                             padding='same',
                             kernel_initializer = initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPool1D(pool_size=config['pool2_size'])(nn)
    nn = keras.layers.Dropout(config['dropout2'])(nn)

    #output block
    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(config['dense'],kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    nn = keras.layers.Dense(config['dense2'],kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    outputs = keras.layers.Dense(1,activation = 'linear',kernel_initializer=initializer)(nn)

    model =  keras.Model(inputs=[seq_inputs,embed_inputs], outputs=outputs)
    return model

def rep_mlp(input_shape,output_shape = 1,factor = 1):
     #initializer
    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005)
    #input layer
    
    
    inputs = keras.Input(shape=input_shape, name='sequence')
    if isinstance(input_shape,int) == False:
        # f_input = tf.keras.layers.AveragePooling1D(pool_size=input_shape[0],strides=None,
        #                                             padding='valid',data_format='channels_last',
        #                                            )(inputs)
        f_input = keras.layers.Flatten()(inputs)
    else:
        f_input = inputs
    nn = keras.layers.Dense(512*factor,kernel_initializer=initializer)(f_input)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    nn = keras.layers.Dense(256*factor,kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    outputs = keras.layers.Dense(output_shape,activation = 'linear',kernel_initializer=initializer)(nn)

    model =  keras.Model(inputs=inputs, outputs=outputs)
    return model   

##### Dataset function #####
def make_dataset(data_dir, split_label, data_stats, batch_size=64, seed=None,
                 shuffle=True, seqs=False, drop_remainder=False):
    """
    create tfr dataset from tfr files
    :param data_dir: dir with tfr files
    :param split_label: fold name to choose files
    :param data_stats: summary dictionary of dataset
    :param batch_size: batch size for dataset to be created
    :param seed: seed for shuffling
    :param shuffle: shuffle dataset
    :param coords: return onehot sequences of the dataset
    :param drop_remainder: drop last batch that might have smaller size then rest
    :return: dataset object
    """
    onehot_length = data_stats['onehot_length']
    embed_length = data_stats['embed_length']
    embed_dim = data_stats['embed_dim']
    num_targets = data_stats['num_targets']
    tfr_path = '%s/tfrecords/%s-*.tfr' % (data_dir, split_label)
    tfr_files = natsorted(glob.glob(tfr_path))
    dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)

    # train
    # if split_label == 'train':
    if (split_label == 'train'):
        # repeat
        # dataset = dataset.repeat()

        # interleave files
        dataset = dataset.interleave(map_func=file_to_records,
                                     cycle_length=4,
                                     num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # shuffle
        dataset = dataset.shuffle(buffer_size=32,
                                  reshuffle_each_iteration=True)

    # valid/test
    else:
        # flat mix files
        dataset = dataset.flat_map(file_to_records)

    dataset = dataset.map(generate_parser(onehot_length, embed_length, embed_dim, num_targets, seqs))
    if shuffle:
        if seed:
            dataset = dataset.shuffle(32, seed=seed)
        else:
            dataset = dataset.shuffle(32)
    # dataset = dataset.batch(64)
    # batch
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    # prefetch
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def generate_parser(onehot_length, embed_length, embed_dim, num_targets, seqs):
    def parse_proto(example_protos):
        """
        Parse TFRecord protobuf.
        :param example_protos: example from tfr
        :return: parse tfr to dataset
        """
        # TFRecord constants
        TFR_ONEHOT = 'onehot'
        TFR_INPUT = 'sequence'
        TFR_OUTPUT = 'target'

        # define features
        if seqs:
            features = {
                TFR_ONEHOT: tf.io.FixedLenFeature([], tf.string),
                TFR_INPUT: tf.io.FixedLenFeature([], tf.string),
                TFR_OUTPUT: tf.io.FixedLenFeature([], tf.string)
            }
        else:
            features = {
                TFR_INPUT: tf.io.FixedLenFeature([], tf.string),
                TFR_OUTPUT: tf.io.FixedLenFeature([], tf.string)
            }

        # parse example into features
        parsed_features = tf.io.parse_single_example(example_protos, features=features)

        if seqs:
            # decode coords
            onehot = tf.io.decode_raw(parsed_features[TFR_ONEHOT], tf.float16)
            onehot = tf.reshape(onehot, [onehot_length, 4])
            onehot = tf.cast(onehot, tf.float32)

        # decode sequence
        # sequence = tf.io.decode_raw(parsed_features[TFR_INPUT], tf.uint8)
        sequence = tf.io.decode_raw(parsed_features[TFR_INPUT], tf.float16)
        sequence = tf.reshape(sequence, [embed_length, embed_dim])
        sequence = tf.cast(sequence, tf.float32)

        # decode targets
        targets = tf.io.decode_raw(parsed_features[TFR_OUTPUT], tf.float16)
        targets = tf.reshape(targets, [num_targets])
        targets = tf.cast(targets, tf.float32)
        if seqs:
            return (onehot, sequence), targets
        else:
            return sequence, targets

    return parse_proto

def load_stats(data_dir):
    """
    :param data_dir: dir of a dataset created using the preprocessing pipeline
    :return: a dictionary of summary statistics about the dataset
    """
    data_stats_file = '%s/statistics.json' % data_dir
    assert os.path.isfile(data_stats_file), 'File not found!'
    with open(data_stats_file) as data_stats_open:
        data_stats = json.load(data_stats_open)
    return data_stats

def file_to_records(filename):
    """
    :param filename: tfr filename
    :return: tfr record dataset
    """
    return tf.data.TFRecordDataset(filename, compression_type='ZLIB')

##### Metrics function #####
