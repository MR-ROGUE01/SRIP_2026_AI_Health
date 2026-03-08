from tensorflow import keras

def build_cnn(input_shape, num_classes=3):
    
    model = keras.Sequential()
    
    model.add(keras.layers.Conv1D(32, kernel_size=5, activation='relu', padding='same', input_shape=input_shape))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    
    model.add(keras.layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model