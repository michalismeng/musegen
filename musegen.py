from config import sequence_length, generator_dir, output_dir
from helper import loadModelAndWeights, createNoteVocabularies
from music21 import note, instrument, stream, duration
import numpy as np
import os

# disable GPU processing as the network doesn't fit in my card's memory
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# ----------------------------------------------

# select the epoch to use when loading the weights of the model generator
generator_epoch = 155

# how many notes to generate ('end' marks are created along the way and the result is split into pieces)
number_of_notes = 500

# how many notes from the beginning to discard
numer_of_discard_notes = 200

# create the vocabulary
note_vocab, note_names_vocab, note_vocab_categorical = createNoteVocabularies()

note_categorical_size = note_vocab_categorical.shape[0]

# note to integer and reversal dictionaries used to make categorical data
note_to_int = dict((note, number) for number, note in enumerate(note_vocab))
int_to_note = dict((number, note) for number, note in enumerate(note_vocab))

print('loading networks...')
generator = loadModelAndWeights(os.path.join(generator_dir, 'model.json'), os.path.join(generator_dir, 'weights-{:02d}.h5'.format(generator_epoch)))

# make a melody!!!
pattern = np.eye(note_categorical_size)[np.random.choice(note_categorical_size, size=sequence_length)]

print('generating output...')

# generate notes
generator_output = []

for _ in range(number_of_notes):
    generator_input = np.reshape(pattern, (1, sequence_length, note_categorical_size))

    prediction = generator.predict(generator_input, verbose=0)
    generator_output.append(prediction)

    pattern = np.vstack([pattern, prediction])
    pattern = pattern[1:len(pattern)]

generator_output = generator_output[numer_of_discard_notes:]
output_notes = [int_to_note[np.argmax(pattern)] for pattern in generator_output]
output_notes = np.array(output_notes)

# output_notes contains: pitch values in midi format (integers), 'rest' marks, 'end' marks

# split the generated notes into pieces based on 'end' marks
indices = np.where(output_notes == 'end')
indices = np.reshape(indices, (-1))             # reshape to 1D
indices = np.insert(indices, 0, 0)              # insert edge value for looping
    
pieces = [output_notes]
if len(indices) > 1:
    pieces = ([ output_notes[(indices[j] + 1):indices[j + 1] ] for j in range(len(indices) - 1)])

print('writing output to disk...')
os.makedirs(output_dir, exist_ok=True)

# output pieces to midi files
for index, piece in enumerate(pieces):
    midi_notes = []
    offset = 0
    for n in piece:
        if n == 'rest':
            new_note = note.Rest()
            new_note.duration = duration.Duration(0.5)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            midi_notes.append(new_note)
        else:
            new_note = note.Note(int(n))
            new_note.duration = duration.Duration(0.5)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            midi_notes.append(new_note)
        offset += 0.5
        
    midi_stream = stream.Stream(midi_notes)
    midi_stream.write('midi', fp=os.path.join(output_dir, 'sample-{}.mid'.format(index)))