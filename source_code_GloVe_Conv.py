'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 10 News dataset
(classification of newsgroup messages into 10 different categories).
'''

from __future__ import print_function

import os
import codecs
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.callbacks import CSVLogger

csv_logger = CSVLogger('glove_conv_log.csv', append=True, separator=';')

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
all_words = set()
with codecs.open("GLOVE_vectors.txt", "r", encoding="utf-8") as f:
#with codecs.open("MORPHEME_GLOVE_vectors.txt", "r", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        all_words.add(word)
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
S = u'абвгдежзийклмнопрстуфхцчшщыъьэюя'
dir_inx = 0
INPUT_DIR = "TEXTS"
#INPUT_DIR = "MORHEME_TEXTS"
for dir in os.listdir(INPUT_DIR):
    labels_index[os.path.dirname(dir)] = dir_inx
    for file in os.listdir(INPUT_DIR + "\\" + dir):
        train_text = ""
        with codecs.open(INPUT_DIR + "\\" + dir + "\\" + file, "r", encoding="utf-8") as input_file:
            for line in input_file:
                line = line.replace("-", " ")
                line = line.replace(u"—", " ")
                line = line.replace(u"–", " ")
                word_list = line.split(" ")
                for word in word_list:
                    new_word = ''
                    letters_list = list(word.lower())
                    for letter in letters_list:
                        if letter in S:
                            new_word += letter
                    new_word = new_word.strip()
                    if (len(new_word) and (new_word in all_words)):
                        train_text = train_text + new_word + " "

        train_text = train_text.strip()
        if len(train_text):
            texts.append(train_text)
            labels.append(labels_index[os.path.dirname(dir)])
    dir_inx += 1

print('Found %s texts.' % len(texts))

MAX_SEQUENCE_LENGTH = 0
text_to_tokenizer = texts
to_categorical_labels = labels
for inx in range(5):
    MAX_SEQUENCE_LENGTH += 500
    MAX_NUM_WORDS = 20000
    EMBEDDING_DIM = 300
    VALIDATION_SPLIT = 0.2
    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(text_to_tokenizer)
    sequences = tokenizer.texts_to_sequences(text_to_tokenizer)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    categorical_labels = to_categorical(np.asarray(to_categorical_labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', categorical_labels.shape)

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    categorical_labels = categorical_labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-num_validation_samples]
    y_train = categorical_labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = categorical_labels[-num_validation_samples:]

    print('Preparing embedding matrix.')

    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    print('Training model.')

    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(dir_inx, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    model.fit(x_train, y_train,
              batch_size=128,
              epochs=5,
              validation_data=(x_val, y_val))

    scores = model.evaluate(x_val, y_val)

    log_file = open('glove_conv_log.csv', 'a')
    log_file.write("MAX_NUM_WORDS = %d \n" % MAX_SEQUENCE_LENGTH)
    log_file.write("Точность на тестовых данных: %.2f%% \n" % (scores[1] * 100))
    log_file.close()