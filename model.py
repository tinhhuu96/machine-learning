import tensorflow as tf
import logging
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn import model_selection, metrics
import numpy as np

logging.basicConfig(level=logging.DEBUG)

nb_classes = 3    # 5段階にランク判定
max_words = 0

batch_size = 2
nb_epoch = 16


def main():
    logging.debug("*** model_news start ***")
    global max_words

    logging.debug("Reading data...")
    data = json.load(open("review-data.json"))
    X = data["X"] # メッセージ文
    Y = data["Y"] # ランク
    max_words=len(X[0])
    print('max_words:'+str(max_words))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    print(len(X_train), len(Y_train))
    # Y_train = np_utils.to_categorical(Y_train, nb_classes)
    print(len(X_train), len(Y_train))

    model = model_train(X_train, Y_train)
    model_save(model)
#     del model
#     reloaded_model = KerasClassifier(build_fn=model_load, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)
#     reloaded_model = KerasClassifier(build_fn=self.model_load, nb_epoch=nb_epoch, batch_size=batch_size)
#     loaded_model = model_load()
#     model_eval(reloaded_model, X_test, Y_test)
    model_eval(model, X_test, Y_test)
    logging.debug("*** model_news end ***")


def model_train(X_train, Y_train):
    model = KerasClassifier(
        build_fn=model_build,
        batch_size=batch_size)
    logging.debug("Training model...")
    model.fit(np.array(X_train), np.array(Y_train),  epochs=nb_epoch)
    return model


def model_save(savemodel):
    logging.debug("Saving model...")
    sav_file = "model_news.h5"
    savemodel.model.save(sav_file)


''' picle
    filename = "./news/model_news.sav"
    pickle.dump(savemodel.model, open(filename, 'wb'))
'''

''' weights and json   
    hdf5_file = "./news/model_news.hdf5"
    savemodel.model.save_weights(hdf5_file)
    model_json = savemodel.model.to_json()
    f = open('./news/model_news.json', 'w')
    f.write(model_json)
    f.close()
'''
'''
def model_load():
    logging.debug("Loading model...")
    filename = "./news/model_news.h5"
    loaded_model = load_model(filename)
    return loaded_model
'''
'''
    filename = "./news/model_news.hdf5"
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model
'''


def model_eval(model, X_test, Y_test):
    logging.debug("Evaluating model...")
    y = model.predict(np.array(X_test))
    ac_score = metrics.accuracy_score(np.array(Y_test), y)
    cl_report = metrics.classification_report(np.array(Y_test), y)
    print("Rate:", ac_score)
    print("レポート=\n", cl_report)


def model_build():
    logging.debug("Building model...")
    model = Sequential()
    # 入力層
    model.add(Dense(128, input_shape=(max_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    #　中間層1
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    # # # 中間層2
    # model.add(Dense(128))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    #
    # model.add(Dense(32))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))

    #　出力層
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model


if __name__ == '__main__':
    main()

