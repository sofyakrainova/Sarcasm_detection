import json
import tensorflow as tf
import numpy as np

# Load the JSON file
with open("../sarcasm_data/all_data.json", 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []

# Collect sentences and labels into the lists
for item in datastore["data"]:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

print("Number of sentences:", len(labels))

# Constants
TRAINING_SIZE = 40000
VOCAB_SIZE = 10000
MAX_LENGTH = 32
EMBEDDING_DIM = 100

# Split the sentences
train_sentences = sentences[0:TRAINING_SIZE]
test_sentences = sentences[TRAINING_SIZE:]

# Split the labels
train_labels = labels[0:TRAINING_SIZE]
test_labels = labels[TRAINING_SIZE:]

# Instantiate the vectorization layer
vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE, output_sequence_length=MAX_LENGTH)

# Generate the vocabulary based on the training inputs
vectorize_layer.adapt(train_sentences)


# Define path to file containing the embeddings
glove_file = "glove.6B.100d.txt"

# Initialize an empty embeddings index dictionary
glove_embeddings = {}

# Read file and fill glove_embeddings with its contents
with open(glove_file, encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        glove_embeddings[word] = coefs

# Create a word index dictionary
word_index = {x:i for i,x in enumerate(vectorize_layer.get_vocabulary())}

# Initialize an empty numpy array with the appropriate size
embeddings_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))

# Iterate all of the words in the vocabulary and if the vector representation for
# each word exists within GloVe's representations, save it in the embeddings_matrix array
for word, i in word_index.items():
    embedding_vector = glove_embeddings.get(word)
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector


# Apply the vectorization layer on the train and test inputs
train_sequences = vectorize_layer(train_sentences)
test_sequences = vectorize_layer(test_sentences)

# Combine input-output pairs for training
train_dataset_vectorized = tf.data.Dataset.from_tensor_slices((train_sequences,train_labels))
test_dataset_vectorized = tf.data.Dataset.from_tensor_slices((test_sequences,test_labels))

# Save vectorized datasets and embedding matrix
train_dataset_vectorized.save("train_data")
test_dataset_vectorized.save("test_data")
with open('emb_matrix.npy', 'wb') as f:
    np.save(f, embeddings_matrix)