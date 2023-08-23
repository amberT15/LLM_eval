import tensorflow as tf
from tensorflow import keras
import scipy.stats

def mtsplice_model(task_num,input_shape=(400,4)):
    initializer = tf.keras.initializers.HeUniform()
    
    l_input = tf.keras.layers.Input(input_shape)
    r_input = tf.keras.layers.Input(input_shape)
    # left input head
    l_nn = tf.keras.layers.Conv1D(filters=128,
                             kernel_size=7,
                             padding = 'same',
                             kernel_initializer=initializer)(l_input)
    l_nn = keras.layers.BatchNormalization()(l_nn)
    l_nn = keras.layers.Activation('relu')(l_nn)
    l_nn = keras.layers.MaxPooling1D(4)(l_nn)
    l_nn = keras.layers.Dropout(0.2)(l_nn)

    #right input head
    r_nn = tf.keras.layers.Conv1D(filters=128,
                             kernel_size=7,
                             padding = 'same',
                             kernel_initializer=initializer)(r_input)
    r_nn = keras.layers.BatchNormalization()(r_nn)
    r_nn = keras.layers.Activation('relu')(r_nn)
    r_nn = keras.layers.MaxPooling1D(4)(r_nn)
    r_nn = keras.layers.Dropout(0.2)(r_nn)

    # Concatenate outputs from l/r input head
    nn = tf.keras.layers.Concatenate(axis=1)([l_nn,r_nn])

    #Second conv layer
    nn = keras.layers.Conv1D(filters=164,
                             kernel_size=5,
                             padding = 'same',
                             kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    #third conv layer
    nn = keras.layers.Conv1D(filters=256,
                             kernel_size=5,
                             padding = 'same',
                             kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(2)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    #dense layer
    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(256)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    nn = keras.layers.Dense(256)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    #Output layer
    logits = keras.layers.Dense(task_num)(nn)
    output = keras.layers.Activation('linear')(logits)
    
    model = keras.Model(inputs=[l_input,r_input], outputs=output)
    return model

def mtsplice_embed_model(input_shape,task_num):
    initializer = tf.keras.initializers.HeUniform()
    

    l_input = tf.keras.layers.Input(input_shape)
    r_input = tf.keras.layers.Input(input_shape)
    # left input head
    #add batchnorm and dimension reduction
    l_nn = keras.layers.BatchNormalization()(l_input)
    l_nn = keras.layers.Conv1D(filters=512,kernel_size=1,
                             kernel_initializer = initializer)(l_nn)
    l_nn = tf.keras.layers.Conv1D(filters=128,
                             kernel_size=7,
                             padding = 'same',
                             kernel_initializer=initializer)(l_nn)
    l_nn = keras.layers.BatchNormalization()(l_nn)
    l_nn = keras.layers.Activation('relu')(l_nn)
    l_nn = keras.layers.MaxPooling1D(4)(l_nn)
    l_nn = keras.layers.Dropout(0.2)(l_nn)

    #right input head
    r_nn = keras.layers.BatchNormalization()(r_input)
    r_nn = keras.layers.Conv1D(filters=512,kernel_size=1,
                             kernel_initializer = initializer)(r_nn)
    r_nn = tf.keras.layers.Conv1D(filters=128,
                             kernel_size=7,
                             padding = 'same',
                             kernel_initializer=initializer)(r_nn)
    r_nn = keras.layers.BatchNormalization()(r_nn)
    r_nn = keras.layers.Activation('relu')(r_nn)
    r_nn = keras.layers.MaxPooling1D(4)(r_nn)
    r_nn = keras.layers.Dropout(0.2)(r_nn)

    # Concatenate outputs from l/r input head
    nn = tf.keras.layers.Concatenate(axis=1)([l_nn,r_nn])

    #Second conv layer
    nn = keras.layers.Conv1D(filters=164,
                             kernel_size=5,
                             padding = 'same',
                             kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    #third conv layer
    nn = keras.layers.Conv1D(filters=256,
                             kernel_size=5,
                             padding = 'same',
                             kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(2)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    #dense layer
    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(256)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    nn = keras.layers.Dense(256)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    #Output layer
    logits = keras.layers.Dense(task_num)(nn)
    output = keras.layers.Activation('linear')(logits)
    
    model = keras.Model(inputs=[l_input,r_input], outputs=output)
    return model

def embed_MLP(input_shape,task_num):
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

    outputs = keras.layers.Dense(task_num,activation = 'linear',kernel_initializer=initializer)(nn)

    model =  keras.Model(inputs=inputs, outputs=outputs)
    return model

def linear_model(input_shape,task_num):
    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005)

    inputs = keras.Input(shape=input_shape, name='sequence')
    outputs = keras.layers.Dense(task_num,kernel_initializer=initializer)(inputs)

    model =  keras.Model(inputs=inputs, outputs=outputs)
    return model

class diff_KL (tf.keras.losses.Loss):
    def __init__(self, name="mask_kl", **kwargs):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        mean_logit = y_true[:,:,-1]
        psi_true = tf.math.sigmoid(y_true[:,:,0]+mean_logit)

        #find no observation mask
        mask = tf.cast(tf.where(tf.math.is_nan(psi_true),0,1),tf.bool)

        #use mean logit and predition to get psi_pred
        pred_logit = tf.math.add(y_pred,mean_logit)
        psi_pred = tf.math.sigmoid(pred_logit)

        #clip psi value
        clip_true = tf.clip_by_value(psi_true, clip_value_min=1e-5, clip_value_max=1-1e-5)
        clip_pred = tf.clip_by_value(psi_pred, clip_value_min=1e-5, clip_value_max=1-1e-5)

        clip_true = tf.where(mask,clip_true,clip_pred)

        #KL divergence
        kl1 = tf.math.log(tf.math.divide(clip_true,clip_pred))
        kl1 = tf.math.multiply(clip_true,kl1)
        kl2 = tf.math.log(tf.math.divide((1-clip_true),(1-clip_pred)))
        kl2 = tf.math.multiply((1-clip_true),kl2)
        kl = tf.math.add(kl1,kl2)
        clean_kl = tf.multiply(kl,tf.cast(mask,tf.float32))

        return tf.reduce_mean(clean_kl)

def mt_evaluate(true,pred):
    corr_list = []
    for a in range(0,56):
            corr,_ = scipy.stats.spearmanr(true[:,a,0],pred[:,a],nan_policy='omit')
            corr_list.append(corr)
    return corr_list