from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def train_lstm_model(model, train_seq, train_label, val_seq, val_label, epochs=100):

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-4, verbose=1)

    history = model.fit(train_seq, train_label, epochs=epochs, validation_data=(val_seq, val_label),
                        verbose=1, callbacks=[early_stopping, reduce_lr])

    return model, history
