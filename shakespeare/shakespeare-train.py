import tensorflow as tf
import numpy as np
import os
import time
import argparse

def split_input_target(chunk) -> tuple:
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  return tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
        batch_input_shape=[batch_size, None]
    ),
    tf.keras.layers.GRU(rnn_units,
        return_sequences=True,
        stateful=True,
        recurrent_initializer='glorot_uniform'
    ),
    tf.keras.layers.Dense(vocab_size)
  ])

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def train_model():

    # download dataset
    path_to_file = tf.keras.utils.get_file('shakespeare.txt', 
        'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    print('Length of text: {} characters'.format(len(text)))

    # pre-process dataset
    vocab = sorted(set(text))
    print ('Number of unique characters: {}'.format(len(vocab)))
    char_to_idx = {u:i for i, u in enumerate(vocab)}
    idx_to_char = np.array(vocab)
    text_as_int = np.array([char_to_idx[c] for c in text])

    # divide training and target data
    seq_length = 100
    examples_per_epoch = len(text) // (seq_length + 1)
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
    dataset = sequences.map(split_input_target)

    # prepare training data
    batch_size, buffer_size = 64, 10000
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

    # build the model
    vocab_size = len(vocab)
    embedding_dim = 256
    rnn_units = 1024
    model = build_model(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=batch_size
    )
    model.compile(optimizer='adam', loss=loss)
    model.summary()

    # specify checkpoint callbacks
    checkpoint_dir = 'shakespeare\\training-checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True
    )

    # train the model
    epochs = 10
    history = model.fit(dataset,
        epochs=epochs,
        callbacks=[checkpoint_callback]
    )

def generate_model(start_string):
    embedding_dim = 256
    rnn_units = 1024

    # download dataset
    path_to_file = tf.keras.utils.get_file('shakespeare.txt', 
        'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    print('Length of text: {} characters'.format(len(text)))

    # pre-process dataset
    vocab = sorted(set(text))
    print ('Number of unique characters: {}'.format(len(vocab)))
    char_to_idx = {u:i for i, u in enumerate(vocab)}
    idx_to_char = np.array(vocab)
    text_as_int = np.array([char_to_idx[c] for c in text])
    vocab_size = len(vocab)

    # rebuild model
    checkpoint_dir = 'shakespeare\\training-checkpoints'
    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))
    model.summary()

    # tune generation parameters
    temperature = 1.0
    num_generate = 1000
    text_generated = []
    input_eval = [char_to_idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # run prediction loop
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx_to_char[predicted_id])
    print(start_string + ''.join(text_generated))

if __name__ == "__main__":

    # parse CL arguments
    parser = argparse.ArgumentParser(description="Build, train, and generate GRU RNN over Shakespearean text.")
    parser.add_argument('-t', '--task', action="store", dest="task", help="desired RNN task [train/generate]", required=True)
    parser.add_argument('-s', '--string_string', action="store", dest="start_string", help="start string for generation task", default=u"ROMEO: ")
    args = parser.parse_args()

    # run correct pipeline
    if args.task == "train": train_model()
    elif args.task == "generate": generate_model(args.start_string)
    else: parser.error("invalid task - see help for instructions") 
