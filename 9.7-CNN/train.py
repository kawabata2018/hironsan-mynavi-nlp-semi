import os
import time
import argparse
import numpy as np
from utils import *
from preprocessing import *
from inference import InferenceAPI
from models import RNNModel, LSTMModel, CNNModel
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def return_time(startime):
    return "\n[{:.2f}s]".format(time.time() - startime)


def main(args):
    print(args)
    startime = time.time()

    # Set hyper-parameters.
    batch_size = 128
    epochs = 100
    maxlen = 300
    model_path = 'models/model_{}.h5'
    num_words = 40000
    num_label = 2

    # Data loading.
    print(return_time(startime), "1. Loading data ...")
    x, y = load_dataset('data/amazon_reviews_multilingual_JP_v1_00.tsv')

    # pre-processing.
    print(return_time(startime), "2. Preprocessing dataset ...")
    x = preprocess_dataset(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2,
                                                        random_state=42)
    vocab = build_vocabulary(x_train, num_words)
    x_train = vocab.texts_to_sequences(x_train)
    x_test = vocab.texts_to_sequences(x_test)
    x_train = pad_sequences(x_train, maxlen=maxlen, truncating='post')
    x_test = pad_sequences(x_test, maxlen=maxlen, truncating='post')

    # Preparing word embedding.
    if args.loadwv:
        print(return_time(startime), "3. Loading word embedding ...")
        wv_path = 'data/wv_{0}_{1}.npy'.format(maxlen, num_words)
        if os.path.exists(wv_path):
            wv = np.load(wv_path)
            print(return_time(startime), "Loaded word embedding successfully!")
        else:
            print(return_time(startime), "Word embedding file doesn't exist")
            exit()

    else:
        print(return_time(startime), "3. Preparing word embedding ...")
        wv = load_fasttext('data/cc.ja.300.vec.gz')
        wv = filter_embeddings(wv, vocab.word_index, num_words)
        # Saving word embedding.
        if args.savewv:
            wv_path = 'data/wv_{0}_{1}.npy'.format(maxlen, num_words)
            np.save(wv_path, wv)
            print(return_time(startime), "Saved word embedding successfully!", wv_path)

    # Build models.
    models = [
        RNNModel(num_words, num_label, embeddings=None).build(),
        LSTMModel(num_words, num_label, embeddings=None).build(),
        CNNModel(num_words, num_label, embeddings=None).build(),
        RNNModel(num_words, num_label, embeddings=wv).build(),
        LSTMModel(num_words, num_label, embeddings=wv).build(),
        CNNModel(num_words, num_label, embeddings=wv).build(),
        CNNModel(num_words, num_label, embeddings=wv, trainable=False).build()
    ]

    model_names = [
        "RNN-None",
        "LSTM-None",
        "CNN-None",
        "RNN-wv",
        "LSTM-wv",
        "CNN-wv",
        "CNN-wv-notrain"
    ]

    print(return_time(startime), "4. Start training ...")
    for i, model in enumerate(models):
        print("***********************************")
        print(return_time(startime), "Model:", model_names[i])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['acc'])

        # Preparing callbacks.
        callbacks = [
            EarlyStopping(patience=3),
            ModelCheckpoint(model_path.format(model_names[i]), save_best_only=True)
        ]

        # Train the model.
        model.fit(x=x_train,
                  y=y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.2,
                  callbacks=callbacks,
                  shuffle=True)

        # Inference.
        model = load_model(model_path.format(model_names[i]))
        api = InferenceAPI(model, vocab, preprocess_dataset)
        y_pred = api.predict_from_sequences(x_test)
        print('precision: {:.4f}'.format(precision_score(y_test, y_pred, average='binary')))
        print('recall   : {:.4f}'.format(recall_score(y_test, y_pred, average='binary')))
        print('f1       : {:.4f}'.format(f1_score(y_test, y_pred, average='binary')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # python train.py -sf => False
    # python train.py    => True
    parser.add_argument('--savewv', '-sf', action='store_false')
    # python train.py -lt => True
    # python train.py    => False
    parser.add_argument('--loadwv', '-lt', action='store_true')

    args = parser.parse_args()
    main(args)
