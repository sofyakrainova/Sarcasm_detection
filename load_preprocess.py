import json
import tensorflow as tf

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
EMBEDDING_DIM = 16

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


# Apply the vectorization layer on the train and test inputs
train_sequences = vectorize_layer(train_sentences)
test_sequences = vectorize_layer(test_sentences)

# Combine input-output pairs for training
train_dataset_vectorized = tf.data.Dataset.from_tensor_slices((train_sequences,train_labels))
test_dataset_vectorized = tf.data.Dataset.from_tensor_slices((test_sequences,test_labels))

# Save vectorized datasets
train_dataset_vectorized.save("train_data")
test_dataset_vectorized.save("test_data")