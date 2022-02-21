import tensorflow.keras as keras
import datapreprocess
import cnnModel
import json
import numpy as np
from keras.models import load_model
import tensorflow as tf

dataset_path = "genres"
DATA_PATH = "data.json"

if __name__ == "__main__":
    # preprocessing data
    # datapreprocess.save_mfcc(dataset_path, DATA_PATH)

    # get train, validation, test splits
    # X_train, X_validation, X_test, y_train, y_validation, y_test = cnnModel.prepare_datasets(0.25, 0.2)
    #
    # # create network
    # input_shape = (X_train.shape[1], X_train.shape[2], 1)
    # model = cnnModel.build_model(input_shape)
    #
    # # compile model
    # optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    # model.compile(optimizer=optimiser,
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])
    #
    # model.summary()
    #
    # # train model
    # history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs= 30)
    #
    # model.save('cnn.h5')
    #
    # # Convert the model.
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # tflite_model = converter.convert()
    #
    # # Save the model.
    # with open('cnn.tflite', 'wb') as f:
    #     f.write(tflite_model)
    #
    # # evaluate model on test set
    # test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    # print('\nTest accuracy:', test_acc)



    model = load_model('cnn.h5')

    # predict sample
    datapreprocess.save_mfcc("rootdir", "data_1.json")
    a,b = cnnModel.load_data("data_1.json")
    a = a[..., np.newaxis]
    X_to_predict = a[3]
    y_to_predict = b[3]
    cnnModel.predict(model, X_to_predict, y_to_predict)