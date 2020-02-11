import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Load data

train_img = []
train_label = []
test_img = []
test_label = []

# Preprocess

# Reshape onto same size? 128*128 etc?
# Reshape RGB value to [0,1] by divide 255

#Create CNN model
# Current : 3 conv layers, 2 pooling, 1 flatten, 2 dense.

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train
history = model.fit(train_img, train_label, epochs=20,
                    validation_data=(test_img, test_label))

# test
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

#output result