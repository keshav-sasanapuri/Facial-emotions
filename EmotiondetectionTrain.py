import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

train_data_generator = ImageDataGenerator(rescale=1./255)
test_data_generator = ImageDataGenerator(rescale=1./255)

train_generator = train_data_generator.flow_from_directory(
        'data/train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

test_generator = test_data_generator.flow_from_directory(
        'data/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

neural_model = Sequential()

neural_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
neural_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
neural_model.add(MaxPooling2D(pool_size=(2, 2)))
neural_model.add(Dropout(0.25))
neural_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
neural_model.add(MaxPooling2D(pool_size=(2, 2)))
neural_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
neural_model.add(MaxPooling2D(pool_size=(2, 2)))
neural_model.add(Dropout(0.25))
neural_model.add(Flatten())
neural_model.add(Dense(1024, activation='relu'))
neural_model.add(Dropout(0.5))
neural_model.add(Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)
neural_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

neural_model_info = neural_model.fit_generator(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=50,
        test_data=test_generator,
        test_steps=7178 // 64)

model_json = neural_model.to_json()
with open("neural_model.json", "w") as json_file:
    json_file.write(model_json)

neural_model.save_weights('neural_model.h5')

