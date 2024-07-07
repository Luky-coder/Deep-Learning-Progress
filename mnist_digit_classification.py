# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 14:50:56 2024

@author: Lucas
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Loading MNIST data
(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()

print('trainset:', X_train.shape) # 60,000 images
print('testset:', X_test.shape) # 10,000 images

# Data normalization
X_train = X_train / 255
X_test = X_test / 255

# Visualization of some images
fig, ax = plt.subplots(nrows=1, ncols=10, figsize=(20, 4))
for i in range(10):
  ax[i].imshow(X_train[i], cmap='gray')

plt.tight_layout()
plt.show()

# Network layer configuration
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# Model compilation
model.compile(optimizer='adam',
              loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Model training
model.fit(X_train, y_train, epochs=10)

# Model evaluation
test_loss, test_acc = model.evaluate(X_test,  y_test)
print('Test accuracy:', test_acc)

# Predictive model (softmax)
prediction_model = keras.Sequential([model, keras.layers.Softmax()])
predict_proba = prediction_model.predict(X_test)
predictions = np.argmax(predict_proba, axis=1)

print(predictions[:10])
print(y_test[:10])