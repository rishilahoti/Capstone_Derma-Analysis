"""
  This file used to build our model
"""
import pickle
from keras import Model
import tensorflow.keras as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout

x_train = pickle.load(open('dir/x_train.bin', 'rb'))
y_train = pickle.load(open('dir/y_train.bin', 'rb'))
x_val = pickle.load(open('dir/x_val.bin', 'rb'))
y_val = pickle.load(open('dir/y_val.bin', 'rb'))
x_test = pickle.load(open('dir/x_test.bin', 'rb'))
y_test = pickle.load(open('dir/y_test.bin', 'rb'))

base_model = K.applications.Xception(include_top=False,
                                     weights='imagenet',
                                     input_tensor=None,
                                     input_shape=(299, 299, 3),
                                     pooling=None,
                                     classifier_activation="softmax",
                                     )
base_model.trainable = False

inputs = K.Input(shape=(299, 299, 3))

x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

outputs = Dense(6, activation='softmax')(x)           # final layer

model = Model(inputs, outputs)

opt = Adam(learning_rate=0.001)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
acc_checkpoint = ModelCheckpoint("first_train/max_acc.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
loss_checkpoint = ModelCheckpoint("first_train/min_loss.keras", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [acc_checkpoint, loss_checkpoint]

hist = model.fit(x_train, y_train, epochs=1, validation_data=(x_val, y_val), batch_size=32, callbacks=callbacks_list)

with open('dir/hist/hist2', 'wb') as file_pi:
    pickle.dump(hist.history, file_pi)


base_model.trainable = True

opt = Adam(learning_rate=0.00001)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

acc_checkpoint = ModelCheckpoint("second_train/max_acc.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
loss_checkpoint = ModelCheckpoint("second_train/min_loss.keras", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [acc_checkpoint, loss_checkpoint]

hist = model.fit(x_train, y_train, epochs=1, validation_data=(x_val, y_val), batch_size=32, callbacks=callbacks_list)
model.save("model.h5")
with open('./dir/hist/hist2', 'wb') as file_pi:
    pickle.dump(hist.history, file_pi)
