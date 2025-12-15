import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def create_model(init, reg):
    import tensorflow as tf
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


def load_and_prepare_data(csv_path="data/19_25_with_upcoming.csv"):
    # Mirrors preprocessing used in model.py / model_2.py
    pd.set_option('display.max_columns', None)

    def convert_wl(x):
        if x == 'W':
            return 1
        elif x == 'L':
            return 0
        else:
            return pd.NA

    data = pd.read_csv(csv_path, dtype={'GAME_ID': str})
    exclude = ['WL']
    subset = data.columns.difference(exclude)
    data.dropna(subset=subset, inplace=True)
    data['WL'] = data['WL'].apply(convert_wl)
    # Some GAME_ID values have a prefix; try to strip non-digits
    try:
        data['GAME_ID'] = data['GAME_ID'].str[2:].astype(int)
    except Exception:
        # fallback: extract digits
        data['GAME_ID'] = data['GAME_ID'].astype(str).str.extract(r'(\d+)').astype(int)

    numeric = data[data.columns.difference(['WL', 'GAME_ID'])].select_dtypes(include='number')
    norm_data = (numeric - numeric.mean()) / numeric.std()

    ids = data['GAME_ID']
    target = data['WL']
    columns_to_drop = ['MIN', 'MIN_OPP', 'TEAM_ID', 'TEAM_ID_OPP']
    features = norm_data.drop(columns=[c for c in columns_to_drop if c in norm_data.columns])

    train_cutoff = 22500001
    train_mask = (ids < train_cutoff) & (target.notna())
    test_mask = (ids >= train_cutoff) & (target.notna())

    features_train = features.loc[train_mask].to_numpy(dtype='float32')
    target_train = target.loc[train_mask].to_numpy(dtype='float32')

    features_test = features.loc[test_mask].to_numpy(dtype='float32')
    target_test = target.loc[test_mask].to_numpy(dtype='float32')

    return {
        'data': data,
        'features': features,
        'features_train': features_train,
        'features_test': features_test,
        'target_train': target_train,
        'target_test': target_test,
        'feature_names': features.columns.tolist()
    }


def train_model(features_train, target_train, epochs=50, batch_size=64):
    import tensorflow as tf
    from tensorflow import keras

    initializer = tf.keras.initializers.GlorotUniform()
    regularizer = tf.keras.regularizers.L2(0.01)

    model = create_model(initializer, regularizer)
    opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, weight_decay=0.0)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

    model.fit(features_train, target_train, batch_size=batch_size, epochs=epochs, verbose=1)
    return model


def compute_permutation_importance(model, X_test, y_test, feature_names, n_repeats=500, out_dir='graphs'):
    from sklearn.metrics import log_loss
    os.makedirs(out_dir, exist_ok=True)

    X = X_test.astype('float32')
    y = y_test.astype('int32')

    y_pred_base = model.predict(X).flatten()
    baseline_logloss = log_loss(y, np.clip(y_pred_base, 1e-7, 1 - 1e-7))
    importances = []
    losses_per_feature = []
    for i, fname in enumerate(feature_names):
        losses = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, i])
            y_pred = model.predict(X_perm).flatten()
            losses.append(log_loss(y, np.clip(y_pred, 1e-7, 1 - 1e-7)))
        losses_per_feature.append(losses)
        importance = float(np.mean(losses) - baseline_logloss)
        importances.append((fname, importance))

    imp_df = pd.DataFrame(importances, columns=['feature', 'delta_logloss']).sort_values('delta_logloss', ascending=False)
    csv_path = os.path.join(out_dir, 'permutation_importance_logloss.csv')
    imp_df.to_csv(csv_path, index=False)

    # compute statistics per feature and save detailed stats
    stats_rows = []
    for fname, losses in zip(feature_names, losses_per_feature):
        mean_loss = float(np.mean(losses))
        std_loss = float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0
        stderr = float(std_loss / np.sqrt(len(losses))) if std_loss > 0 else 0.0
        delta = mean_loss - baseline_logloss
        if stderr > 0:
            z = delta / stderr
        else:
            # If no variance, set z to +inf/-inf depending on sign
            z = float('inf') if delta > 0 else (float('-inf') if delta < 0 else 0.0)
        stats_rows.append((fname, delta, mean_loss, std_loss, stderr, z))

    stats_df = pd.DataFrame(stats_rows, columns=['feature', 'delta_logloss', 'mean_loss', 'std_loss', 'stderr', 'z'])
    stats_csv = os.path.join(out_dir, 'permutation_stats.csv')
    stats_df.to_csv(stats_csv, index=False)

    # plot horizontal bar of importances
    plt.figure(figsize=(8, max(4, len(imp_df) * 0.25)))
    plt.barh(imp_df['feature'][::-1], imp_df['delta_logloss'][::-1])
    plt.xlabel('Increase in LogLoss when permuted (higher = more important)')
    plt.tight_layout()
    png_path = os.path.join(out_dir, 'permutation_importance_logloss.png')
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"Wrote permutation importance CSV: {csv_path}")
    print(f"Wrote permutation importance plot: {png_path}")
    print(f"Wrote detailed permutation stats CSV: {stats_csv}")

    return imp_df, stats_df


def try_shap(model, X_background, X_explain, feature_names, out_dir='graphs'):
    try:
        import shap
    except Exception as e:
        print('SHAP not available or import failed:', e)
        return

    os.makedirs(out_dir, exist_ok=True)
    print('Running SHAP GradientExplainer (may take time)...')
    try:
        explainer = shap.GradientExplainer(model, X_background)
        shap_values = explainer.shap_values(X_explain)

        shap.summary_plot(shap_values, X_explain, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'shap_summary_beeswarm.png'), dpi=150)
        plt.close()

        shap.summary_plot(shap_values, X_explain, feature_names=feature_names, plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'shap_summary_bar.png'), dpi=150)
        plt.close()

        print('Saved SHAP summary plots to', out_dir)
    except Exception as e:
        print('SHAP explanation failed:', e)


def main():
    print('Loading and preparing data...')
    ctx = load_and_prepare_data()

    feature_names = ctx['feature_names']
    X_train = ctx['features_train']
    y_train = ctx['target_train']
    X_test = ctx['features_test']
    y_test = ctx['target_test']

    # Try to load a saved model first
    model = None
    model_dir = 'models/best_model'
    model_h5 = 'models/best_model.h5'
    try:
        import tensorflow as tf
        if os.path.exists(model_dir):
            print('Loading model from', model_dir)
            model = tf.keras.models.load_model(model_dir)
        elif os.path.exists(model_h5):
            print('Loading model from', model_h5)
            model = tf.keras.models.load_model(model_h5)
    except Exception as e:
        print('Failed to load existing model:', e)

    if model is None:
        print('No saved model found — training a new model (this may take some time)')
        model = train_model(X_train, y_train, epochs=60, batch_size=64)
        try:
            os.makedirs('models', exist_ok=True)
            print('Saving trained model to models/best_model')
            model.save('models/best_model')
        except Exception as e:
            print('Failed to save model:', e)

    print('Computing permutation importance (LogLoss)...')
    imp_df = compute_permutation_importance(model, X_test, y_test, feature_names, n_repeats=5, out_dir='graphs')

    # Optionally run SHAP if available (use a small background and sample)
    try:
        bg_size = min(100, X_train.shape[0])
        bg_idx = np.random.choice(X_train.shape[0], bg_size, replace=False)
        background = X_train[bg_idx]
        explain_size = min(500, X_test.shape[0])
        explain_idx = np.arange(explain_size)
        X_explain = X_test[explain_idx]
        try_shap(model, background, X_explain, feature_names, out_dir='graphs')
    except Exception as e:
        print('SHAP preparation failed:', e)

    print('Done.')


if __name__ == '__main__':
    main()
