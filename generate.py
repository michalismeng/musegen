from music21 import *
import numpy as np
import os
import os.path

from helper import loadChorales, createNoteVocabularies, loadModelAndWeights
from config import sequence_length, note_embedding_dim, latent_dim, note_embedding_dir, generator_dir

# disable GPU processing as the network doesn't fit in my card's memory
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# ----------------------------------------------

from keras.utils import to_categorical
from keras import backend as K
from keras.losses import mse, binary_crossentropy
from keras.models import model_from_json
from keras.optimizers import RMSprop
from keras import objectives
from keras import initializers
from keras.callbacks import ModelCheckpoint

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Dropout, Lambda

# gaussian sampling
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# VAE network loss (reconstruction + KL-divergence)
def vae_loss(x, x_decoded_mean):
    xent_loss = objectives.mse(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
    return xent_loss + kl_loss   

# create the vocabulary
note_vocab, note_names_vocab, note_vocab_categorical = createNoteVocabularies()

# note to integer and reversal dictinaries used to make categorical data
note_to_int = dict((note, number) for number, note in enumerate(note_vocab))
int_to_note = dict((number, note) for number, note in enumerate(note_vocab))

# load Bach chorales
print('loading chorales...')
notes = loadChorales()
only_notes = [chord[0] for (chord, _) in notes]         # discard durations

# load note embedding encoder
note_encoder = loadModelAndWeights(os.path.join(note_embedding_dir, 'encoder-model.json'), os.path.join(note_embedding_dir, 'encoder-weights.h5'))

# preapre data for the network

network_input = []
network_output = []

# prepare series of sequences and for each one the target output
for i in range(0, len(notes) - sequence_length):
    sequence_in = only_notes[i:i + sequence_length]
    sequence_out = only_notes[i + sequence_length]
    
    categorical_in = np.reshape([note_vocab_categorical[note_to_int[x]] for x in sequence_in], (sequence_length, -1))
    categorical_out = np.reshape([note_vocab_categorical[note_to_int[sequence_out]]], (1, -1))
    
    # append the embedding of each note
    network_input.append(note_encoder.predict(categorical_in))
    network_output.append([note_encoder.predict(categorical_out)])
    
network_input = np.array(network_input)
network_output = np.reshape(np.array(network_output), (-1,note_embedding_dim))

# define the music generator network

# at first we have a series of LSTM layers for progression of notes
x = Input(shape=(sequence_length, note_embedding_dim,), name='generator_input')
h = LSTM(512, return_sequences=True, name='h_lstm_1')(x)
h = LSTM(256, return_sequences=True, name='h_lstm_2')(h)
h = Dropout(0.3, name='h_dropout')(h)
h = LSTM(128, name='h_lstm_3')(h)

# then use a VAE for non-deterministic generation of notes
z_mean = Dense(latent_dim, kernel_initializer='uniform', name='z_mean')(h)
z_log_var = Dense(latent_dim, kernel_initializer='uniform', name='z_log_var')(h)

z = Lambda(sampling, output_shape=(latent_dim, ), name='z')([z_mean, z_log_var])
decoded_z = Dense(note_embedding_dim, activation='linear', name='generator_output')(z)

# end-to-end generator
generator = Model(x, decoded_z)

# compile and print generator summary
optimizer = RMSprop(lr=0.001)
generator.compile(optimizer=optimizer, loss=vae_loss)
generator.summary()

# save the model first as training may be interrupt due to it taking much time
os.makedirs(generator_dir, exist_ok=True)
with open(os.path.join(generator_dir, "model.json"), "w") as json_file:
    json_file.write(generator.to_json())

# wegihts will be saved every epoch
filepath = os.path.join(generator_dir, "weights-{epoch:02d}.h5")   

reportWeightsCheck = ModelCheckpoint(
    filepath, monitor='loss', 
    verbose=0,        
    save_best_only=True,        
    mode='min'
)

print('training...')
# train the generator network
generator.fit(x=network_input, y=network_output, shuffle=False, epochs=100, batch_size=64, callbacks=[reportWeightsCheck])

print('SUCCESS')