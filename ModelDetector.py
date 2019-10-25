from keras.applications import MobileNetV2
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization
from keras.models import Model


class ModelDetector:

    def __init__(self):
        default_model = MobileNetV2(include_top=False,
                                    weights='imagenet',
                                    input_shape=(224, 224, 3))

        # Extends default model
        x = default_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(units=1024,
                  activation='relu',
                  kernel_initializer='random_uniform',
                  bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        output = Dense(units=20, activation='softmax')(x)

        # Create a extended model
        self.model = Model(inputs=default_model.input, outputs=output)

        # Change all layers trainable
        for layer in self.model.layers:
            layer.trainable = True
