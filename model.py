import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


img_height = 100
img_width = 100
batch_size = 32

# set up training model
model = keras.Sequential(
    [
        layers.Input((100, 100, 3)),
        layers.Conv2D(16, 3, padding="same"),
        layers.Conv2D(32, 3, padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(24),
    ]
)

# set up training data
dataset_train = tf.keras.preprocessing.image_dataset_from_directory(
    'data/',
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=69,
    validation_split=0.1,
    subset='training'
)

dataset_validation = tf.keras.preprocessing.image_dataset_from_directory(
    'data/',
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=69,
    validation_split=0.1,
    subset='validation'
)

# augment the image's brightness
def augment(x, y):
    image = tf.image.random_brightness(x, max_delta=0.1)
    return image, y

dataset_train = dataset_train.map(augment)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
    metrics=['accuracy']
)

# training model
model.fit(dataset_train, epochs=10,verbose=2)

tf.saved_model.save(model, './model_data')