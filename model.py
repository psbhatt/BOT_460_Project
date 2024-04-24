import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

data = pd.read_csv("22_23.csv", converters={'WL': lambda x: int(x == 'W'), 'GAME_ID': lambda x: str(x)[2:]})
data['GAME_ID'] = data['GAME_ID'].astype(int)
print(data['GAME_ID'])
numeric = data[data.columns.difference(['WL', 'GAME_ID'])].select_dtypes(include='number')
norm_data = (numeric - numeric.mean()) / numeric.std()

# print(graphs.columns, graphs.head)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print("norm graphs shape ", norm_data.shape)
# print(norm_data['GAME_ID'])
# print(" norm graphs first element ", norm_data.loc[0])

ids = data['GAME_ID']
target = data['WL']
# unnamed is the actual index of the dataset. dont know what 'index' is
# WL was already dropped at this point
columns_to_drop = ['MIN', 'MIN_OPP', 'index']
features = norm_data.drop(columns=columns_to_drop)
# print(features['GAME_ID'])
# features = norm_data[norm_data.columns.difference(['WL'])]
# print("features shape ", features.shape)

# print(" features first elemn ", features.loc[0])
# features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state=0)
features_train = features[ids < 22300061]
features_test = features[ids >= 22300061]
target_train = target[ids < 22300061]
target_test = target[ids >= 22300061]
print(features_train.shape)
# chosen arbitrarily
initializer = tf.keras.initializers.GlorotUniform
# also chosen arbitrarily
regularizer = tf.keras.regularizers.L2(0.01)


# removed kernel_regularizer from the arguments of first keras layer
def create_model(initializer, regularizer):
    model = tf.keras.models.Sequential([
        # tf.keras.layers.Flatten(),  # Optionally add a Flatten layer if needed
        tf.keras.layers.Dense(128, activation=tf.nn.relu,
                              kernel_initializer=initializer, kernel_regularizer=regularizer),
        # You may add a dropout to this layer by adding a dropout layer as below
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(64, activation=tf.nn.relu, kernel_initializer=initializer),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(32, activation=tf.nn.relu, kernel_initializer=initializer),
        # tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer)
        # This output is z without softmax. It's called logits.
        # you could also use an output layer with softmax activation as in below
        # In that case, your definition of loss will include from_logits=False
        # tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=initializer)
    ])
    return model


# model = create_model(initializer, regularizer)
batch_size = 8  # what should this be changed to
epochs = 40
learning_rates = [0.1, 0.01, 0.001]
optimizers = []
for lr in learning_rates:
    optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.0, weight_decay=0.0)
    optimizers.append(optimizer)
    optimizer = keras.optimizers.Adagrad(learning_rate=lr)
    optimizers.append(optimizer)


# v_accs = {}
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
for i, opt in enumerate(optimizers):
    model = create_model(initializer, regularizer)
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=["accuracy"],
    )
    optimizer_type = opt.get_config()['name']
    learning_rate = opt.get_config()['learning_rate']
    #   history=model.fit(features, target, batch_size=batch_size, epochs=epochs)
    history = model.fit(features_train, target_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(features_test, target_test))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train (' + optimizer_type + ', LR=' + str(learning_rate) + ')', 'validation'], loc='upper left')
    plt.savefig(f'graphs/{optimizer_type} {learning_rate:.2} plot{i}.png')
    plt.show()
    plt.clf()
