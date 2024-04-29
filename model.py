import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

data = pd.read_csv("19_23_updated.csv", converters={'WL': lambda x: int(x == 'W'), 'GAME_ID': lambda x: str(x)[2:]})
data['GAME_ID'] = data['GAME_ID'].astype(int)
numeric = data[data.columns.difference(['WL', 'GAME_ID'])].select_dtypes(include='number')
norm_data = (numeric - numeric.mean()) / numeric.std()
#data = pd.read_csv("Dataset1.csv", converters={'WL': lambda x: int(x == 'W')})
#numeric = data[data.columns.difference(['WL'])].select_dtypes(include='number')
#norm_data=(numeric-numeric.mean())/numeric.std()



# print(data.columns, data.head)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print("norm data shape ", norm_data.shape)
# print(" norm data first element ", norm_data.loc[0])

ids = data['GAME_ID']
target = data['WL']
#unnamed is the actual index of the dataset. dont know what 'index' is
# WL was already dropped at this point
columns_to_drop = ['MIN', 'MIN_OPP','index', 'TEAM_ID', 'TEAM_ID_OPP']
features = norm_data.drop(columns=columns_to_drop)
# features = norm_data[norm_data.columns.difference(['WL'])]
print("features shape ", features.shape)

print(" features first elemn ", features.loc[0])
#features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state=0)
features_train = features[ids < 22300000]
features_test = features[ids >= 22300000]
target_train = target[ids < 22300000]
target_test = target[ids >= 22300000]
print(features_train.shape)


class PrintFinalAccuracy(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch == (epochs - 1):
            print(f"Final Training Accuracy: {logs['accuracy']:.4f}")
            print(f"Final Validation Accuracy: {logs['val_accuracy']:.4f}")

print_final_accuracy_callback = PrintFinalAccuracy()

# chosen arbitrarily
initializer = tf.keras.initializers.GlorotUniform
# also chosen arbitrarily
regularizer=tf.keras.regularizers.L2(0.01)
# removed kernel_regularizer from the arguments of first keras layer
def create_model(initializer, regularizer):
    model = tf.keras.models.Sequential([
        # tf.keras.layers.Flatten(),  # Optionally add a Flatten layer if needed
        tf.keras.layers.Dense(128, activation=tf.nn.relu,
                              kernel_initializer=initializer, kernel_regularizer=regularizer),
        # You may add a dropout to this layer by adding a dropout layer as below
        # tf.keras.layers.Dropout(.2, input_shape=(2,)),
        tf.keras.layers.Dropout(.3), # dropout layer with rate of .3
        tf.keras.layers.Dense(64, activation=tf.nn.relu, kernel_initializer=initializer),
        tf.keras.layers.Dropout(.3),
        tf.keras.layers.Dense(32, activation=tf.nn.relu, kernel_initializer=initializer),
          tf.keras.layers.Dropout(.3),
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer)
        # This output is z without softmax. It's called logits.
        # you could also use an output layer with softmax activation as in below
        # In that case, your definition of loss will include from_logits=False
        # tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=initializer)
    ])
    return model

batch_size = 64 # what should this be changed to
epochs = 120
# learning_rates = [0.25, 0.1, 0.01]
# optimizers = []
# loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# for lr in learning_rates:
#   optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.0, weight_decay=0.0)
#   optimizers.append(optimizer)
#   optimizer = keras.optimizers.Adam(learning_rate=lr)
#   optimizers.append(optimizer)

# for i, opt in enumerate(optimizers):
#   model = create_model(initializer, regularizer)
#   model.compile(
#       optimizer=opt,
#       loss=loss,
#       metrics=["accuracy"],
#   )
#   optimizer_type = opt.get_config()['name']
#   learning_rate = opt.get_config()['learning_rate']
# #   history=model.fit(features, target, batch_size=batch_size, epochs=epochs)
#   #history=model.fit(features_train, target_train, batch_size=batch_size, epochs=epochs, validation_data=(features_test, target_test))
#   history=model.fit(features_train, target_train, batch_size=batch_size, epochs=epochs, validation_data=(features_test, target_test), verbose=0,callbacks=[print_final_accuracy_callback])
#   plt.plot(history.history['accuracy'])
#   plt.plot(history.history['val_accuracy'])
#   plt.title('model accuracy')
#   plt.ylabel('accuracy')
#   plt.xlabel('epoch')
#   plt.legend(['train (' + optimizer_type + ', LR=' + str(learning_rate) + ')', 'validation'], loc='upper left')
#   plt.savefig(f'graphs/{optimizer_type} {learning_rate:.2} plot{i}.png')
#   plt.clf()
  

opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, weight_decay=0.0)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
model = create_model(initializer, regularizer)
model.compile(
    optimizer=opt,
    loss=loss,
    metrics=["accuracy"],
)
history = model.fit(features_train, target_train, batch_size=batch_size, epochs=epochs, validation_data=(features_test, target_test), verbose=0,callbacks=[print_final_accuracy_callback])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train (' + 'SGD' + ', LR=' + str(0.01) + ')', 'validation'], loc='upper left')
# plt.savefig(f'graphs/{optimizer_type} {learning_rate:.2} plot{i}.png')
plt.clf()
plt.show()
test_pred = model.predict(features_test)
output = data[ids >= 22300000]
output["Model Win Prob"] = test_pred
output.to_csv("23_results.csv")


# print(output["GAME_DATE"])
# print(output[["GAME_ID", "GAME_DATE", "TEAM_NAME"]].shape)