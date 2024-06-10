# /mnt/data/extract_gist_hog_features.py

import os
import numpy as np
from PIL import Image
from leargist import color_gist
#from skimage.feature import hog
#from skimage.color import rgb2gray
#from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def load_images(folder):
    images = []
    labels = []
    for root, dirs, files in os.walk(folder):
        if dirs == []:
            for name in files:
#                print(root+name)
                img = Image.open(os.path.join(root, name))
                images.append(img)
                if "Other" not in root:
#                if "malware" in filename:
                    labels.append(1)
                else:
                    labels.append(0)
    return images, labels

def extract_gist_features(images):
    features = []
    for img in images:
#        img = img.resize((128, 128))  # Resize for consistency if needed
        gist_descriptor = color_gist(img)
        features.append(gist_descriptor)
    return np.array(features)

def extract_hog_features(images):
    features = []
    for img in images:
#        img = img.resize((128, 128))  # Resize for consistency if needed
#        img_gray = rgb2gray(np.array(img))
        hog_descriptor, _ = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        features.append(hog_descriptor)
    return np.array(features)

def main():
    # Path to the folder containing images
    train_folder = "/home/kali/Documents/Major_Project/dumpware10_images/4096/300/TRAIN/"
    test_folder = "/home/kali/Documents/Major_Project/dumpware10_images/4096/300/TEST/"
    
    # Load images and labels
    train_images, train_labels = load_images(train_folder)
    test_images, test_labels = load_images(test_folder)
    
    # Extract GIST features
    train_gist_features = extract_gist_features(train_images)
    test_gist_features = extract_gist_features(test_images)
    
    # Extract HOG features
#    hog_features = extract_hog_features(images)
    
    # Combine GIST and HOG features
#    features = np.hstack((gist_features, hog_features))
#    print(test_images)
#    print(test_labels)
    # Split data into training and test sets
#    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train = train_gist_features
    y_train = train_labels
    X_test = test_gist_features
    y_test = test_labels
    
    # Initialize and train the SVM classifier
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    
    # Predict and evaluate the classifier
    y_pred = svm_classifier.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    main()
