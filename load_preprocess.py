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

SIZE = len(labels)
print("Number of sentences:", len(labels))


# Constants
SPLIT_RATES = [0.6, 0.8]
VOCAB_SIZE = 10000
MAX_LENGTH = 32


# Split the sentences
train_sentences = sentences[0:int(SIZE*SPLIT_RATES[0])]
valid_sentences = sentences[int(SIZE*SPLIT_RATES[0]):int(SIZE*(SPLIT_RATES[1]))]
test_sentences = sentences[int(SIZE*(SPLIT_RATES[1])):]

# Split the labels
train_labels = labels[0:int(SIZE*SPLIT_RATES[0])]
valid_labels = labels[int(SIZE*SPLIT_RATES[0]):int(SIZE*SPLIT_RATES[1])]
test_labels = labels[int(SIZE*SPLIT_RATES[1]):]

# Instantiate the vectorization layer
vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE, output_sequence_length=MAX_LENGTH)

# Generate the vocabulary based on the training inputs
vectorize_layer.adapt(train_sentences)

# Apply the vectorization layer on the train and test inputs
train_sequences = vectorize_layer(train_sentences)
valid_sequences = vectorize_layer(valid_sentences)
test_sequences = vectorize_layer(test_sentences)

# Combine input-output pairs for training
train_dataset_vectorized = tf.data.Dataset.from_tensor_slices((train_sequences,train_labels))
valid_dataset_vectorized = tf.data.Dataset.from_tensor_slices((valid_sequences,valid_labels))
test_dataset_vectorized = tf.data.Dataset.from_tensor_slices((test_sequences,test_labels))

# Save vectorized datasets
train_dataset_vectorized.save("train_data")
valid_dataset_vectorized.save("valid_data")
test_dataset_vectorized.save("test_data")