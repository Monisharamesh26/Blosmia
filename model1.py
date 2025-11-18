import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt

class BloodCellClassifier:
    def __init__(self, input_shape=(299, 299, 3), num_classes=5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.class_mapping = {
            0: "Baso",
            1: "Eos",
            2: "Lympho",
            3: "Mono",
            4: "Neutro"
        }

    def check_gpu(self):
        # Check if GPU is available.
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                print(f"Found GPU: {gpu}")
            return True
        print("No GPU detected. Using CPU.")
        return False

    def create_data_generators(self, train_dir, test_dir, batch_size=64):
        # Create training and validation data generators.
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input, validation_split=0.2
        )

        train_generator = datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42,
            subset="training",
        )

        val_generator = datagen.flow_from_directory(
            test_dir,
            target_size=self.input_shape[:2],
            batch_size=1,
            class_mode="categorical",
            shuffle=False,
            seed=42,
            subset="validation",
        )

        return train_generator, val_generator

    def build_model(self):
        # Build and compile the InceptionV3 model.
        base_model = InceptionV3(
            weights="imagenet", include_top=False, input_shape=self.input_shape
        )

        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False

        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation="relu")(x)
        predictions = Dense(self.num_classes, activation="softmax")(x)

        self.model = Model(inputs=base_model.input, outputs=predictions)
        self.model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

    def train(self, train_generator, val_generator, epochs=25):
        # Train the model and return training history.
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        history = self.model.fit(
            train_generator, epochs=epochs, validation_data=val_generator
        )
        return history

    def plot_training_history(self, history):
        # Plot training and validation metrics.
        # Accuracy plot
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="train acc")
        plt.plot(history.history["val_accuracy"], label="val acc")
        plt.title("Model Accuracy")
        plt.legend()

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"], label="train loss")
        plt.plot(history.history["val_loss"], label="val loss")
        plt.title("Model Loss")
        plt.legend()
        plt.show()

    def save_model(self, filepath):
        # Save the trained model.
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        self.model.save(filepath)

    def load_saved_model(self, filepath):
        # Load a previously saved model.
        self.model = load_model(filepath)

    def predict_image(self, image_path):
        # Predict the class of a single image.
        if self.model is None:
            raise ValueError("No model loaded. Either train or load a model first.")

        img = image.load_img(image_path, target_size=self.input_shape[:2])
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        return {
            "class_name": self.class_mapping[predicted_class],
            "probabilities": predictions[0],
            "class_index": predicted_class,
        }


def modeltrain():
    # Initialize the classifier
    classifier = BloodCellClassifier()

    # Check for GPU
    classifier.check_gpu()

    # Define paths
    train_dir = "E:/Project25/J_Blosmia1/static/wbc/training"
    test_dir = "E:/Project25/J_Blosmia1/static/wbc/testing"
    model_path = "E:/Project25/J_Blosmia1/preprocess/v3_wbc_only1.h5"

    # Create data generators
    train_generator, val_generator = classifier.create_data_generators(
        train_dir, test_dir
    )

    # Build and train model
    classifier.build_model()
    history = classifier.train(train_generator, val_generator)

    # Plot training results
    classifier.plot_training_history(history)

    # Save the model
    classifier.save_model(model_path)

    # Example prediction
    test_image = "E:/Project25/J_Blosmia1/static/wbc/testing/Mono/crop_MONO_image2209_70_0.png"
    result = classifier.predict_image(test_image)
    print(f"Predicted class: {result['class_name']}")
    print(f"Confidence scores: {result['probabilities']}")
    return True

if __name__ == "__main__":
    modeltrain()