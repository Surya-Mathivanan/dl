import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Generate dummy sequence data
# 1000 samples, each with 10 time steps and 8 features
X = np.random.rand(1000, 10, 8)
y = np.random.randint(0, 2, 1000)  # Binary labels

# Build a simple RNN model
model = Sequential()
model.add(SimpleRNN(32, activation='relu', input_shape=(10, 8)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=25, batch_size=32)

# Evaluate on the same data (just for example)
loss, acc = model.evaluate(X, y)
print(f"Accuracy: {acc:.4f}")
