import numpy as np
import ipdb

from keras.layers import Embedding, Input, LSTM, Dense, GRU, Dropout
from keras.layers import SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import concatenate, Flatten
from keras.models import Model, load_model
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint

import keras.backend.tensorflow_backend as ktf
import tensorflow as tf

def get_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

ktf.set_session(get_session())


datapath = "./Numpydata/"
embedding_matrix = np.load(datapath+"embedding_matrix.npy")
padded_texts_train = np.load(datapath+"padded_texts_train.npy")
padded_texts_test_1 = np.load(datapath+"padded_texts_test_1.npy")
padded_texts_test_2 = np.load(datapath+"padded_texts_test_2.npy")
poss1_onehot_test_1 = np.load(datapath+"poss1_onehot_test_1.npy")
poss1_onehot_test_2 = np.load(datapath+"poss1_onehot_test_2.npy")
poss2_onehot_test_1 = np.load(datapath+"poss2_onehot_test_1.npy")
poss2_onehot_test_2 = np.load(datapath+"poss2_onehot_test_2.npy")
poss1_onehot_train = np.load(datapath+"poss1_onehot_train.npy")
poss2_onehot_train = np.load(datapath+"poss2_onehot_train.npy")
relations_onehot_train = np.load(datapath+"relations_onehot_train.npy")
relations_onehot_test_1 = np.load(datapath+"relations_onehot_test_1.npy")
relations_onehot_test_2 = np.load(datapath+"relations_onehot_test_2.npy")

poss_sequence_train = np.load(datapath+"poss_sequence.npy")
poss_sequence_test_1 = np.load(datapath+"poss_sequence_test_1.npy")
poss_sequence_test_2 = np.load(datapath+"poss_sequence_test_2.npy")
relations_sequence_train = np.load(datapath+"relations_sequences.npy")
relations_sequence_test_1 = np.load(datapath+"relations_sequences_test_1.npy")

relations_sequence_test_2 = np.load(datapath+"relations_sequences_test_2.npy")

max_length = 13
vocab_size = 8818

y_train = np.load(datapath+"y_train.npy")
'''
x_train = np.concatenate((poss1_onehot_train, poss2_onehot_train, relations_onehot_train), axis=1)
x_test_1 = np.concatenate((poss1_onehot_test_1, poss2_onehot_test_1, relations_onehot_test_1), axis=1)
x_test_2 = np.concatenate((poss1_onehot_test_2, poss2_onehot_test_2, relations_onehot_test_2), axis=1)
'''

poss_onehot_train = np.concatenate((poss1_onehot_train, poss2_onehot_train), axis=1)
poss_onehot_test_1 = np.concatenate((poss1_onehot_test_1, poss2_onehot_test_1), axis=1)
poss_onehot_test_2 = np.concatenate((poss1_onehot_test_2, poss2_onehot_test_2), axis=1)

'''
input_poss = Input(shape=(poss_sequence_train.shape[1],))
poss_embedded = Embedding(poss_onehot_train.shape[1]+1, 32, trainable=True)(input_poss)
lstm_poss = GRU(16, return_sequences=True)(poss_embedded)

input_relations = Input(shape=(relations_sequence_train.shape[1],))
relations_embedded = Embedding(relations_onehot_train.shape[1]+1, 32, trainable=True)(input_relations)
lstm_relations = GRU(32, return_sequences=True)(relations_embedded)
'''

inputs = Input(shape=(max_length,))
x_train_embedded = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_length, trainable=False)(inputs)
x_train_embedded = SpatialDropout1D(0.5314)(x_train_embedded)
lstm_hidden = GRU(512, return_sequences=True)(x_train_embedded)

avg_pool = GlobalAveragePooling1D()(lstm_hidden)
max_pool = GlobalMaxPooling1D()(lstm_hidden)

inputs2 = Input(shape=(poss_onehot_train.shape[1],))
inputs3 = Input(shape=(relations_onehot_train.shape[1],))

#x = concatenate([Flatten()(lstm_hidden), Flatten()(lstm_poss), Flatten()(lstm_relations)])
#x = concatenate([lstm_hidden, Flatten()(lstm_poss), Flatten()(lstm_relations)])
x = concatenate([avg_pool, max_pool, inputs2, inputs3])
#x = concatenate([avg_pool, max_pool])

x = Dense(2048, activation='selu')(x)
x = Dropout(0.3318)(x)
x = Dense(1024, activation='selu')(x)
predictions = Dense(19, activation='softmax')(x)

#model = Model(inputs=[inputs, input_poss, input_relations],outputs=predictions)
model = Model(inputs=[inputs, inputs2, inputs3], outputs=predictions)

#model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

#filepath = "model_poolgru.hdf5"
filepath = "../Model/model_poolgru.hdf5"
mycallback = ModelCheckpoint(filepath,monitor='val_acc', verbose=0, save_best_only=False, mode='auto', period=1)


#model.fit([padded_texts_train, poss_sequence_train, relations_sequence_train] , y_train, validation_split = 0.1 , batch_size=64, epochs=15, shuffle = True, callbacks=[mycallback])
#model.fit([padded_texts_train, poss_onehot_train, relations_onehot_train] , y_train, validation_split = 0.1 , batch_size=128, epochs=20, shuffle = True, callbacks=[mycallback])

model = load_model(filepath)

#pd_1 = model.predict([padded_texts_test_1, poss_sequence_test_1, relations_sequence_test_1])
#pd_2 = model.predict([padded_texts_test_2, poss_sequence_test_2, relations_sequence_test_2]) # dir
pd_1 = model.predict([padded_texts_test_1, poss_onehot_test_1, relations_onehot_test_1])
pd_2 = model.predict([padded_texts_test_2, poss_onehot_test_2, relations_onehot_test_2]) # dir
pd_sum = []

for index in range(len(pd_1)):
    sumtmp = np.zeros(20)
    for pd_index in range(len(pd_1[index])):
        if pd_index+1 == 10:
            sumtmp[10] = pd_1[index,10]+pd_2[index,10]
        elif pd_index+1 < 10:
            sumtmp[pd_index+1] = pd_1[index,pd_index]+pd_2[index,pd_index+10]
        else:
            sumtmp[pd_index+1] = pd_1[index,pd_index]+pd_2[index,pd_index-10]
    pd_sum.append(sumtmp)

feature_types = ["Other","Cause-Effect","Message-Topic","Instrument-Agency","Product-Producer","Entity-Destination","Entity-Origin","Member-Collection","Component-Whole","Content-Container"]



with open("output_poolgru_other.txt",'w') as f:
    for index in range(len(pd_sum)):
        f.write(str(index+8001)+'\t')
        #ans
        ans_tmp = np.argmax(pd_sum[index])
        ans_max = np.max(pd_sum[index])
        if ans_max < 0.99:
            ans_tmp = 10
        #dir
        if ans_tmp > 10 :
            ans_dir = "(e2,e1)"
            f.write(str(feature_types[ans_tmp-10])+ans_dir+'\n')
        elif ans_tmp == 10:
            f.write(str(feature_types[0])+'\n')
        else:
            ans_dir = "(e1,e2)"
            f.write(str(feature_types[ans_tmp])+ans_dir+'\n')









