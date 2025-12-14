import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


# MLP model
def create_model(init, reg):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation=tf.nn.relu,
                             kernel_initializer=init, kernel_regularizer=reg),
        tf.keras.layers.Dropout(.3),
        tf.keras.layers.Dense(64, activation=tf.nn.relu, kernel_initializer=init),
        tf.keras.layers.Dropout(.3),
        tf.keras.layers.Dense(32, activation=tf.nn.relu, kernel_initializer=init),
        tf.keras.layers.Dropout(.3),
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=init)
    ])
    return model


class PrintFinalAccuracy(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch == (epochs - 1):
            print(f"Final Training Accuracy: {logs['accuracy']:.4f}")
            print(f"Final Validation Accuracy: {logs['val_accuracy']:.4f}")


def normalize_matchup_probs(data, id_col='MATCHUP', prob_col='Model Win Prob'):
    # Normalize the win probabilities within each game/matchup so the sum equals 1
    def normalize(group):
        total_prob = group[prob_col].sum()
        if total_prob > 0:
            group[prob_col] = group[prob_col] / total_prob
        else:
            group[prob_col] = 0.5  # fallback equal probability if sum is 0
        return group

    normalized_data = data.groupby(id_col).apply(normalize)
    return normalized_data


def tuning_params(init, reg, features_train, features_test, target_train, target_test, print_final, batch_size, epochs):
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    learning_rates = [0.25, 0.1, 0.01]

    optimizers = []
    for lr in learning_rates:
        optimizers.append(keras.optimizers.SGD(learning_rate=lr, momentum=0.0, weight_decay=0.0))
        optimizers.append(keras.optimizers.Adam(learning_rate=lr))

    for i, opt in enumerate(optimizers):
        model = create_model(init, reg)
        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=["accuracy"],
        )
        optimizer_type = opt.get_config()['name']
        learning_rate = opt.get_config()['learning_rate']
        print(optimizer_type, learning_rate)
        history = model.fit(features_train, target_train, batch_size=batch_size, epochs=epochs,
                            validation_data=(features_test, target_test), verbose=0,
                            callbacks=[print_final])

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train (' + optimizer_type + ', LR=' + str(learning_rate) + ')', 'validation'], loc='upper left')
        plt.savefig(f'graphs/tuning/{optimizer_type} {learning_rate:.2} plot{i}.png')
        plt.clf()


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

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train (SGD, LR=0.01)', 'validation'], loc='upper left')
    plt.clf()

    test_pred = model.predict(features_test)

    output = data.loc[(ids >= 22500001) & (data['WL'].notnull()) & (data['HOME'] == 1)].copy()
    output["Model Win Prob"] = test_pred.flatten()
    # output = normalize_matchup_probs(output, id_col='GAME_ID', prob_col='Model Win Prob')
    output = output[['TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_ID', 'GAME_DATE', "MATCHUP", 'WL', "Model Win Prob"]]
    output.to_csv("data/25_results.csv")

    return model


def main():
    matplotlib.use('Agg')
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    def convert_wl(x):
        if x == 'W':
            return 1
        elif x == 'L':
            return 0
        else:
            return pd.NA

    data = pd.read_csv("data/19_25_with_upcoming.csv", dtype={'GAME_ID': str})
    exclude =['WL']
    subset = data.columns.difference(exclude)
    data.dropna(subset=subset, inplace=True)
    data['WL'] = data['WL'].apply(convert_wl)
    data['GAME_ID'] = data['GAME_ID'].str[2:].astype(int)

    numeric = data[data.columns.difference(['WL', 'GAME_ID'])].select_dtypes(include='number')
    norm_data = (numeric - numeric.mean()) / numeric.std()

    ids = data['GAME_ID']
    target = data['WL']
    columns_to_drop = ['MIN', 'MIN_OPP', 'TEAM_ID', 'TEAM_ID_OPP']
    features = norm_data.drop(columns=columns_to_drop)

    train_cutoff = 22500001

    # Use only the home-team row per game for train/test/predict (one sample per matchup)
    train_mask = (ids < train_cutoff) & (target.notna()) & (data['HOME'] == 1)
    test_mask  = (ids >= train_cutoff) & (target.notna()) & (data['HOME'] == 1)
    predict_mask = (ids >= train_cutoff) & (target.isna()) & (data['HOME'] == 1)

    features_train = features.loc[train_mask].to_numpy(dtype='float32')
    target_train = target.loc[train_mask].to_numpy(dtype='float32')

    features_test = features.loc[test_mask].to_numpy(dtype='float32')
    target_test = target.loc[test_mask].to_numpy(dtype='float32')

    features_predict = features.loc[predict_mask].to_numpy(dtype='float32')

    print_final_accuracy_callback = PrintFinalAccuracy()

    initializer = tf.keras.initializers.GlorotUniform()
    regularizer = tf.keras.regularizers.L2(0.01)

    batch_size = 64
    global epochs
    epochs = 120

    # Uncomment tuning if you want to run hyperparameter tuning
    # tuning_params(initializer, regularizer, features_train, features_test, target_train, target_test,
    #               print_final_accuracy_callback, batch_size, epochs)

    model = best_model(data, ids, initializer, regularizer,
                       features_train, features_test, target_train, target_test,
                       print_final_accuracy_callback, batch_size, epochs)

    # Save trained model for later reuse by explainability scripts
    try:
        os.makedirs('models', exist_ok=True)
        # Save as SavedModel directory
        # model.save('models/best_model')
        # Also save HDF5 copy for compatibility
        model.save('models/best_model.h5')
        print('Saved trained model to models/best_model and models/best_model.h5')
    except Exception as e:
        print('Warning: failed to save model:', e)

    upcoming_preds = model.predict(features_predict)
    # Make upcoming_data for home rows only (one row per matchup)
    home_upcoming = data.loc[predict_mask].copy()
    home_upcoming["Model Win Prob"] = upcoming_preds.flatten()

    # Construct away rows (one row per matchup) with probability = 1 - home_prob
    away_upcoming = home_upcoming.copy()
    # Assign away team columns from opponent columns
    away_upcoming['TEAM_ID'] = away_upcoming['TEAM_ID_OPP']
    away_upcoming['TEAM_NAME'] = away_upcoming['TEAM_NAME_OPP']
    away_upcoming['TEAM_ABBREVIATION'] = away_upcoming['TEAM_ABBREVIATION_OPP']
    away_upcoming['HOME'] = 0
    away_upcoming["Model Win Prob"] = 1.0 - away_upcoming["Model Win Prob"]

    # Keep only the columns downstream expects and concatenate both rows
    both_upcoming = pd.concat([home_upcoming, away_upcoming], ignore_index=True, sort=False)
    both_upcoming = both_upcoming[['TEAM_ID', 'TEAM_NAME', 'GAME_DATE', 'MATCHUP', 'HOME', 'Model Win Prob']]

    # Optional: if you had a normalization step, it's redundant now, since p_home + p_away = 1
    both_upcoming.to_csv("data/upcoming_game_predictions.csv", index=False)
    print(f"Saved predictions for {len(both_upcoming)} upcoming game rows to 'data/upcoming_game_predictions.csv'.")


if __name__ == '__main__':
    main()
