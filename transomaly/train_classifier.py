import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Input
from keras.layers import LSTM, GRU
from keras.layers import Dropout, BatchNormalization, Activation, TimeDistributed

from transomaly import prepare_data


COLPB = {'g': 'tab:blue', 'r': 'tab:orange'}
MARKPB = {'g': 'o', 'r': 's', 'z': 'd'}
ALPHAPB = {'g': 0.3, 'r': 1., 'z': 1}


def train_model(X_train, X_test, y_train, y_test, fig_dir='.', epochs=20, retrain=False, passbands=('g', 'r')):
    model_filename = os.path.join('..', "keras_model.hdf5")
    npb = len(passbands)

    if not retrain and os.path.isfile(model_filename):
        model = load_model(model_filename)
    else:
        model = Sequential()

        model.add(LSTM(100, return_sequences=True))
        # model.add(Dropout(0.2, seed=42))
        # model.add(BatchNormalization())

        # model.add(LSTM(100, return_sequences=True))
        # # model.add(Dropout(0.2, seed=42))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.2, seed=42))

        model.add(TimeDistributed(Dense(npb)))
        model.compile(loss='mse', optimizer='adam')
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64, verbose=2)

        print(model.summary())
        model.save(model_filename)

    return model


def plot_metrics(model, X_test, y_test, timesX_test, passbands, fig_dir):
    y_pred = model.predict(X_test)
    print(y_pred)
    print(y_test)

    # Plot predictions vs time per class
    font = {'family': 'normal',
            'size': 36}
    matplotlib.rc('font', **font)

    for idx in np.arange(0, 100):
        print("Plotting example vs timel", idx)
        argmax = -1  # timesX_test[idx].argmax() + 1

        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(13, 15), num="lc_{}".format(idx))

        for pbidx, pb in enumerate(passbands):
            ax1.plot(timesX_test[idx][:argmax], X_test[idx][:, pbidx][:argmax], c='b', lw=3, label=f"X:{pb}", marker=MARKPB[pb], markersize=10, alpha=ALPHAPB[pb])
            ax1.plot(timesX_test[idx][:argmax], y_test[idx][:, pbidx][:argmax], c='g', lw=3, label=f"y_test:{pb}", marker=MARKPB[pb], markersize=10, alpha=ALPHAPB[pb])
            ax1.plot(timesX_test[idx][:argmax], y_pred[idx][:, pbidx][:argmax], c='r', lw=3, label=f"y_pred:{pb}", marker=MARKPB[pb], markersize=10, alpha=ALPHAPB[pb])

        ax1.legend(frameon=True, fontsize=33)
        plt.tight_layout()

        plt.savefig(os.path.join(fig_dir, f"lc_{idx}.pdf"))


def main():
    data_dir = '/Users/danmuth/PycharmProjects/transomaly/data'
    fig_dir = '/Users/danmuth/PycharmProjects/transomaly/plots'
    train_epochs = 20
    retrain = True
    passbands = ('g', 'r')

    X_train, X_test, y_train, y_test, timesX_train, timesX_test = prepare_data.get_data(data_dir=data_dir)

    model = train_model(X_train, X_test, y_train, y_test, fig_dir=fig_dir, epochs=train_epochs, retrain=retrain, passbands=passbands)

    plot_metrics(model, X_test, y_test, timesX_test, passbands=passbands, fig_dir=fig_dir)


if __name__ == '__main__':
    main()



