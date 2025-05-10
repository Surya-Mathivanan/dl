import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Sample data
texts = [
    "I love this movie",         # Positive
    "This film was amazing",     # Positive
    "I hate this movie",         # Negative
    "This film was terrible",    # Negative
    "It was a great experience", # Positive
    "Worst movie ever",          # Negative
]

labels = [1, 1, 0, 0, 1, 0]  # 1 = positive, 0 = negative

# Tokenize the text
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences
padded = pad_sequences(sequences, padding='post', maxlen=6)

# Build model
model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=16, input_length=6))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(padded, np.array(labels), epochs=50, verbose=1)

# Predict
test_text = ["I  enjoyed the movie"]
test_seq = tokenizer.texts_to_sequences(test_text)
test_pad = pad_sequences(test_seq, maxlen=6, padding='post')
prediction = model.predict(test_pad)

print("Sentiment:", "Positive 😊" if prediction[0][0] < 0.5 else "Negative 😞")
