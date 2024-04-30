import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


# MLP model
def create_model(init, reg):
    # removed kernel_regularizer from the arguments of first keras layer
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation=tf.nn.relu,
                              kernel_initializer=init, kernel_regularizer=reg),
        tf.keras.layers.Dropout(.3),  # dropout layer with rate of .3
        tf.keras.layers.Dense(64, activation=tf.nn.relu, kernel_initializer=init),
        tf.keras.layers.Dropout(.3),
        tf.keras.layers.Dense(32, activation=tf.nn.relu, kernel_initializer=init),
        tf.keras.layers.Dropout(.3),
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=init)
    ])
    return model


def main():
    matplotlib.use('Agg')
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    data = pd.read_csv("data/19_23_updated.csv", converters={'WL': lambda x: int(x == 'W'), 'GAME_ID': lambda x: str(x)[2:]})
    data['GAME_ID'] = data['GAME_ID'].astype(int)
    numeric = data[data.columns.difference(['WL', 'GAME_ID'])].select_dtypes(include='number')
    norm_data = (numeric - numeric.mean()) / numeric.std()

    ids = data['GAME_ID']
    target = data['WL']
    # unnamed is the actual index of the dataset. dont know what 'index' is
    columns_to_drop = ['MIN', 'MIN_OPP', 'index', 'TEAM_ID', 'TEAM_ID_OPP']
    features = norm_data.drop(columns=columns_to_drop)

    features_train = features[ids < 22300000]
    features_test = features[ids >= 22300000]
    target_train = target[ids < 22300000]
    target_test = target[ids >= 22300000]

    class PrintFinalAccuracy(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch == (epochs - 1):
                print(f"Final Training Accuracy: {logs['accuracy']:.4f}")
                print(f"Final Validation Accuracy: {logs['val_accuracy']:.4f}")

    print_final_accuracy_callback = PrintFinalAccuracy()

    # chosen arbitrarily
    initializer = tf.keras.initializers.GlorotUniform
    regularizer = tf.keras.regularizers.L2(0.01)

    # tested different batch sizes and epochs
    batch_size = 64
    epochs = 120

    # tuning_params(initializer, regularizer, features_train, features_test, target_train, target_test, print_final_accuracy_callback, batch_size, epochs)
    best_model(data, ids, initializer, regularizer, features_train, features_test, target_train, target_test, print_final_accuracy_callback, batch_size, epochs)


# code used while testing which optimizers and learning rates to use
def tuning_params(init, reg, features_train, features_test, target_train, target_test, print_final, batch_size, epochs):

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    # learning rates to test
    learning_rates = [0.25, 0.1, 0.01]

    # optimizers with each learning rate
    optimizers = []
    for lr in learning_rates:
        optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.0, weight_decay=0.0)
        optimizers.append(optimizer)
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        optimizers.append(optimizer)

    # train model with each hyperparameters combination
    for i, opt in enumerate(optimizers):
        model = create_model(init, reg)
        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=["accuracy"],
        )
        optimizer_type = opt.get_config()['name']
        learning_rate = opt.get_config()['learning_rate']
        history = model.fit(features_train, target_train, batch_size=batch_size, epochs=epochs,
                            validation_data=(features_test, target_test), verbose=0,
                            callbacks=[print_final])

        # generate plot
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train (' + optimizer_type + ', LR=' + str(learning_rate) + ')', 'validation'], loc='upper left')
        plt.savefig(f'graphs/{optimizer_type} {learning_rate:.2} plot{i}.png')
        plt.clf()


# final output using best model
def best_model(data, ids, init, reg, features_train, features_test, target_train, target_test, print_final, batch_size, epochs):
    opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, weight_decay=0.0)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    model = create_model(init, reg)
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=["accuracy"],
    )
    history = model.fit(features_train, target_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(features_test, target_test), verbose=0, callbacks=[print_final])

    # generate plot
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train (' + 'SGD' + ', LR=' + str(0.01) + ')', 'validation'], loc='upper left')
    # plt.savefig(f'graphs/{optimizer_type} {learning_rate:.2} plot{i}.png')
    plt.clf()

    # get predictions
    test_pred = model.predict(features_test)

    # append predictions to 23-24 season data and output to csv
    output = data[ids >= 22300000]
    output["Model Win Prob"] = test_pred
    output.to_csv("data/23_results.csv")


if __name__ == '__main__':
    main()
