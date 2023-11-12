import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

os.chdir('C:/Users/notna/OneDrive/Dokumente/GitHub/image_manipulation')

def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            images.append(image)
    return images

def create_labels(num_samples, label_value):
    return np.full((num_samples,), label_value)

def get_images(path):
    images = []
    image_types = []
    image_names = []
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            images.append(image)
            im = filename.split('.')
            image_types.append(im[1])
            image_names.append(im[0])
    return images, image_names, image_types


def main():
    # Load images from directories
    printed_letters_directory = 'images/printed_letters'
    handwritten_letters_directory = 'images/handwritten_letters'

    printed_contours = load_images_from_directory(printed_letters_directory)
    handwritten_contours = load_images_from_directory(handwritten_letters_directory)

    # Create combined dataset and labels
    X = np.array(printed_contours + handwritten_contours)
    y = np.hstack((create_labels(len(printed_contours), 0), create_labels(len(handwritten_contours), 1)))

    # Shuffle the data
    random_indices = np.random.permutation(len(X))
    X = X[random_indices]
    y = y[random_indices]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42)
    classifier.fit(X_train, y_train)
    # im stupid we cannot train on the images, we need to convert the images into values that make sense. My bad

    # Make predictions on the test set
    predictions = classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy}")

    # Save the trained model
    model_filename = 'trained_model.joblib'
    joblib.dump(classifier, model_filename)
    print(f"Trained model saved to {model_filename}")

if __name__ == '__main__':
    main()