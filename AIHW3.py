import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
# Training/testing split
(train_images, train_labels), (test_images,
                               test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# A VGG16 model - 16 layers, using same number of layers, but took out one of the pooling layers for a conv2d
# for the reduced image size compared to the original, and reduced number of filters on later levels to compensate
# for increased complexity from one less pooling step
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu',
          input_shape=(32, 32, 3), padding="same"))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding="same"))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding="same"))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding="same"))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding="same"))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()
# compile model with SGD optimizer, and categorical crossentropy loss
model.compile(optimizer='SGD', loss=tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True), metrics=['accuracy'])
# fit the model, while logging in 'history'
history = model.fit(train_images, train_labels, epochs=20,
                    validation_data=(test_images, test_labels))
# show required metrics - loss and accuracy
fig, axs = plt.subplots(2)

plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.tight_layout()
plt.savefig('TrainLoss.png')
# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)
