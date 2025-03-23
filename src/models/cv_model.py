import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def create_image_model():
    base_model = ResNet50(weights='imagenet', include_top=False)

    # Freezing the layers of ResNet50 base model to retain pre-trained features
    for layer in base_model.layers:
        layer.trainable = False

    # Creating the custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(3, activation='softmax')(x)

    # Defining the full model
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def train_image_model(image_dir, epochs=10, batch_size=32):
    # Data augmentation for better generalization
    datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,
                                 height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                 horizontal_flip=True, fill_mode='nearest')

    train_generator = datagen.flow_from_directory(image_dir, target_size=(224, 224), 
                                                  batch_size=batch_size, class_mode='sparse')

    # Creating and train the model
    model = create_image_model()
    model.fit(train_generator, epochs=epochs, verbose=1)
    return model
