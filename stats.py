from music21 import corpus, note, instrument, stream
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path

from helper import parseToFlatArray, transpose_to_C_A, loadChorales

# load Bach chorales
print('loading chorales...')
notes = loadChorales()

# calculate statistics 

note_stats = dict()
duration_stats = dict()

for (chord, duration) in notes:
    for _note in chord:
        if _note not in note_stats:
            note_stats[_note] = 0
        note_stats[_note] += 1
        
    if duration not in duration_stats:
        duration_stats[duration] = 0
    duration_stats[duration] += 1

print("Note statistics:")

for key, value in sorted(note_stats.items()):
    k = key
    if(key != 'end' and key != 'rest'):
        k = note.Note(int(key)).pitch.nameWithOctave
    print ("%s: %s" % (k, value))

print("Duration statistics:")                       # the 'end' mark has 0.0 duration!
for key, value in sorted(duration_stats.items()):
    print ("%s: %s" % (key, value))

# calculate chorale length stats
notes_per_chorale = sum(v for k, v in note_stats.items() if k != 'end') / note_stats['end']
print('notes per chorale: ', notes_per_chorale)

average_chorale_duration = sum(v for k, v in duration_stats.items() if k != 0.0) / duration_stats[0.0]
print('average chorale length: ', average_chorale_duration)

print('SUCCESS')