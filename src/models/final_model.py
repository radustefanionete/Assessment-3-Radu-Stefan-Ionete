import torch
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate, Input

def create_fusion_model(text_model, image_model):
    # Text model outputs
    text_input = Input(shape=(256,))
    text_output = text_model(text_input)
    
    # Image model outputs
    image_input = Input(shape=(224, 224, 3))
    image_output = image_model(image_input)

    # Concatenating the outputs of both models
    combined = Concatenate()([text_output, image_output])

    # Adding a dense layer to perform final classification
    final_output = Dense(3, activation='softmax')(combined)

    # Final model
    final_model = Model(inputs=[text_input, image_input], outputs=final_output)
    final_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return final_model
