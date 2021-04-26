from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.applications import Xception
from tensorflow.keras import Model
from Config import *


class GetModel:

    def __init__(self):
        self.input_tensor = Input(shape=img_shape)
        self.model = self.create_model()

    def create_model(self):
        model = Xception(input_tensor=self.input_tensor, pooling='None')
        last_layer = model.get_layer('block14_sepconv2_act').output
        x = Flatten(name='flatten')(last_layer)
        # x = Dense(512, activation='relu', name='dense1')(x)
        x = Dropout(0.4, name='dropout1')(x)
        x = Dense(1024, activation='relu', name='dense2')(x)
        # x = Dropout(0.4, name='dropout2')(x)
        out = Dense(1, activation='sigmoid', name='output')(x)
        model = Model(self.input_tensor, out, name='MyXception_model')
        for layers in model.layers:
            layers.trainable = True
        return model

    def get_model_summary(self):
        return self.model.summary()

    def set_trainable_layers(self, trainable_layers=0):
        for layers in self.model.layers[:-trainable_layers]:
            layers.trainable = False

    def get_layer_info(self):
        trainable = 0
        non_trainable = 0
        total = 0
        for layers in self.model.layers:
            total += 1
            if layers.trainable:
                trainable += 1
            else:
                non_trainable += 1
        print('Total layers: {}'.format(total))
        print('Trainable layers: {}'.format(trainable))
        print('Non-Trainable layers: {}'.format(non_trainable))

    def get_model(self):
        return self.model

    def train_model(self, train, val, patience=10):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
        early_stop = EarlyStopping(patience=patience, verbose=1, monitor='val_accuracy')
        checkpoint = ModelCheckpoint('./{epoch:02d}_{val_accuracy:.04f}.h5',
                                     save_weights_only=False,
                                     monitor='val_accuracy',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='max')
        history = self.model.fit(train, epochs=epochs, validation_data=val, verbose=1,
                                 callbacks=[early_stop, checkpoint])
        return history

    def load_saved_model(self, model_path):
        self.model = load_model(model_path)
        return self.model

    def evaluate_model(self, test):
        return self.model.evaluate(test, verbose=1)

    def generate_full_report(self, test):
        pred = (self.model.predict(test, verbose=1) > 0.5).astype('int64')
        print(confusion_matrix(test.classes, pred.ravel()))
        print(classification_report(test.classes, pred.ravel()))
