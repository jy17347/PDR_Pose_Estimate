import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

def Attention_3d_block(inputs):
    
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    
    a = tf.keras.layers.Permute((2, 1))(inputs) # same transpose
    #a = tf.keras.layers.Reshape((input_dim, TIME_STEPS))(a) 
    # this line is not useful. It's just to know which dimension is what.
    a = tf.keras.layers.Dense(20, activation='softmax')(a)
    
    a_probs = tf.keras.layers.Permute((2, 1), name='attention_vec')(a)
    
    output_attention_mul  = tf.keras.layers.multiply([inputs, a_probs])
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


def LSTM_model(num_degree, num_pose):

    Seq_input = Input(shape=(20,6), dtype='float32', name='seq_input')
    Seq_data = LSTM(64, dropout=0.2, activation='relu', return_sequences=True, input_shape=(20,6))(Seq_input)
    Att_seq = Attention_3d_block(Seq_data)
    Att_seq = LSTM(128, dropout=0.2, activation='relu', return_sequences=True)(Att_seq)
    Att_seq = LSTM(64, dropout=0.2, activation='relu')(Att_seq)
    Att_seq = Dense(32, activation='relu')(Att_seq)
    Att_seq = Dense(16, activation='relu')(Att_seq)
    prediction = Dense(num_pose, activation='softmax')(Att_seq)
    model = Model(inputs=[Seq_input], outputs=prediction)

    return model
