import pandas as pd
import numpy as np

df = pd.read_excel('shopee_comments.xlsx')
df.drop(columns='Unnamed: 0',inplace=True)
df.loc[1730,'Label'] = 2
df.loc[1264,'Label'] = 2
df.loc[3948,'Label'] = 2

df=df.dropna()
df.shape

df = df.drop_duplicates()
df.shape

sentences = df['Comments'].values
labels = df['Label'].values

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
max_len = 128

tokenizer = Tokenizer(num_words=vocab_size, filters='', oov_token='<oov>')
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
sequences_padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

import json
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    json.dump(tokenizer.to_json(), f)
    
X_train = sequences_padded
y_train = labels


x_train = np.asarray(X_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, GlobalAveragePooling1D
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
        Bidirectional(LSTM(units=64)),
        Dense(units=64, activation='relu'),
        Dense(units=3, activation='softmax')
    ])

loss_fn = SparseCategoricalCrossentropy(from_logits=True)
optimizer = Adam(learning_rate=0.01, epsilon=1e-6)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'],run_eagerly=True)

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='loss', patience=6, verbose=1)

model.fit(X_train, y_train, epochs=1000, callbacks=[early_stop])

import pickle
with open('sentiment_lstm.pk1', 'wb') as f:
    pickle.dump(model, f)


