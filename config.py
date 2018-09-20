# configuration file 

# max chorales to be used for statistics, training etc...
maxChorales = 2

# dimension of notes embedding
note_embedding_dim = 16

# sequence length for generator LSTM
sequence_length = 128

# latent dimension of VAE
latent_dim = 512

# directory for saving the note embedding network model
note_embedding_dir = "models/note-embedding"

# directory for saving the generator network model
generator_dir = 'models/generator'

# directory for saving generated music samples
output_dir = 'samples'