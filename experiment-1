import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# XOR input and output
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Build model
model = Sequential([
    Dense(2, input_dim=2, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Quick training (just enough to get correct XOR output)
model.fit(X, y, epochs=500, verbose=0)

# Predict
pred = model.predict(X)
print("Predicted XOR outputs:")
print(np.round(pred))
