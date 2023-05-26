

DATASET LINK- https://drive.google.com/drive/folders/1QGOx4H_bQHCdHmmOu2yt8mbmgqe8tMPq?usp=share_link

from google.colab import drive
drive.mount('/content/drive')

# Import the pathlib module to work with directories and file paths
import pathlib

# Create a Path object representing the directory containing the image data
data_dir = pathlib.Path("/content/drive/MyDrive/ca2assignML")

# Use the glob() method to generate a list of file paths matching the pattern "*/.jpg" in the data directory,
# which corresponds to all the JPEG images in subdirectories of the data directory
image_paths = list(data_dir.glob('*/*.jpg'))

# Get the number of images by finding the length of the list of image paths
image_count = len(image_paths)

# Print the number of images found
print(image_count)

"""**This code generates a CSV file containing the file paths and labels for all JPEG, JPG, and PNG images in subdirectories of a specified directory, and writes the data to the CSV file. The labels are mapped to integers using a predefined dictionary. The CSV file can be used to train a machine learning model.**"""

import os
import csv
from tqdm.notebook import tqdm
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# Specify directory path
product_path = '/content/drive/MyDrive/ML_CA2/MLPR Images'

# Define label mapping
label_map = {
    'Product_1': 1,
    'Product_2': 2,
    'Product_3': 3,
    'Product_4': 4,
    'Product_5': 5
}

# Define output file path
output_path = 'dataset.csv'

# Create output CSV file and write headers
with open(output_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['path', 'label'])

    # Iterate over subdirectories and write file paths and labels to CSV
    for subdir in os.listdir(product_path):
        subdir_path = os.path.join(product_path, subdir)
        if os.path.isdir(subdir_path):
            label = label_map.get(subdir)
            for file_name in os.listdir(subdir_path):
                if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
                    file_path = os.path.join(subdir_path, file_name)
                    writer.writerow([file_path, label])

"""TRANSFER **LEARNING**

**Here we are defining a machine learning model based on the VGG16 architecture with pre-trained weights from ImageNet. The model is compiled with an optimizer, loss function, and metric, and then trained on image data using data generators for training and validation data. The training data is augmented using transformations such as shearing and flipping. The model is trained for a specified number of epochs, and the training and validation accuracy are logged.**

Inspired by https://keras.io/guides/transfer_learning/
"""

# Define image size and number of classes
img_width, img_height = 224, 224
num_classes = 5

# Create an instance of the VGG16 model with pre-trained weights from ImageNet
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the pre-trained layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add new classification layers on top of the pre-trained base model
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(num_classes, activation='softmax')(x)

# Define the complete model to be trained
model = Model(inputs=base_model.input, outputs=x)

# Compile the model with an optimizer, loss function, and metric
#model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Create data generators for training and validation data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('/content/drive/MyDrive/ML_CA2/MLPR Images', target_size=(img_width, img_height), batch_size=32, class_mode='categorical')
val_generator = val_datagen.flow_from_directory('/content/drive/MyDrive/ML_CA2/MLPR Images', target_size=(img_width, img_height), batch_size=32, class_mode='categorical')

# Train the model on the training data with validation at the end of each epoch
model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=10, validation_data=val_generator, validation_steps=len(val_generator))

#printing out the model summary 
model.summary()

"""**This code uses the trained model to generate predictions on a validation dataset and then calculates the confusion matrix to evaluate the performance of the model. The confusion matrix is printed to the console.**"""

from sklearn.metrics import confusion_matrix
import numpy as np

# Generate predictions
predictions = model.predict(val_generator)
y_pred = np.argmax(predictions, axis=1)

# Get true labels
y_true = val_generator.classes

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(cm)

"""**Here the code has generated a confusion matrix plot using the seaborn library. It first generates predictions and gets true labels for a validation dataset using a trained model. It then calculates a confusion matrix and defines class labels. Finally, it plots a heatmap of the confusion matrix with annotated values and labels for predicted and true labels.**"""

import seaborn as sns
import matplotlib.pyplot as plt

# Generate predictions
predictions = model.predict(val_generator)
y_pred = np.argmax(predictions, axis=1)

# Get true labels
y_true = val_generator.classes

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Define class labels
class_names = ['class1', 'class2', 'class3', 'class4', 'class5']

# Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

"""**We have tried to train a deep learning model on a set of training data using an ImageDataGenerator to preprocess the images, and then evaluates the performance of the trained model using validation data. The history of the training and validation accuracy and loss over each epoch is stored in a history object, which is then used to create two plots. The first plot shows the training and validation accuracy over each epoch, and the second plot shows the training and validation loss over each epoch. These plots can be used to evaluate the performance of the model and identify potential issues such as overfitting or underfitting.**"""

import matplotlib.pyplot as plt

# Train the model on the training data with validation at the end of each epoch
history = model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=10, validation_data=val_generator, validation_steps=len(val_generator))

# Plot accuracy curve
plt.figure(figsize=(10,10))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.show()

# Plot loss curve
plt.figure(figsize=(10,10))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.show()

"""**Here we have plotted the learning curve for a trained model. It uses the training and validation loss values stored in the history object, which was generated during training. The learning curve shows the change in the loss function over each epoch, for both the training and validation sets. The plt.plot() function is used to plot the curves, and plt.xlabel(), plt.ylabel(), plt.title(), and plt.legend() are used to set the axis labels, title, and legend for the plot. Finally, plt.show() is called to display the plot.**"""

# Plot learning curve
plt.figure(figsize=(10,10))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.show()

from sklearn.metrics import classification_report

# Generate classification report
report = classification_report(y_true, y_pred, target_names=val_generator.class_indices.keys())
print(report)

"""################## **CNN MODEL** ###################

**This code reads an input CSV file containing image paths and their corresponding labels, and splits them into training and testing sets based on a specified split ratio. It then writes the selected training and testing image paths along with their labels to separate CSV files named 'train.csv' and 'test.csv', respectively. The split is done randomly for each label, and the number of training and testing images for each label is determined based on a 90:10 split ratio. The output CSV files contain the file paths of the images and their corresponding labels, which can be used for image classification tasks.**
"""

import csv
import random
from collections import defaultdict
from math import ceil
import pandas as pd
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array,to_categorical
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# Specify the file paths
input_file = 'dataset.csv'
train_file = 'train.csv'
test_file = 'test.csv'


# Define the split ratio
split_ratio = 0.4

# Read the input file and create a dictionary with the label as the key and a list of image paths as the value
label_dict = defaultdict(list)
with open(input_file, 'r') as file:
    reader = csv.reader(file)
    next(reader) # skip header
    for row in reader:
        label_dict[row[1]].append(row[0])

# Calculate the number of images for each label and the number of training and testing images required based on a 90:10 ratio
num_labels = len(label_dict)
num_train = {}
num_test = {}
for label, images in label_dict.items():
    num_images = len(images)
    num_train[label] = ceil(num_images * split_ratio)
    num_test[label] = num_images - num_train[label]

# For each label, randomly select the required number of training and testing images
train_images = []
test_images = []
for label, images in label_dict.items():
    random.shuffle(images)
    train_images += [(image, label) for image in images[:num_train[label]]]
    test_images += [(image, label) for image in images[num_train[label]:]]

# Write the selected training and testing image paths along with their labels to the 'train.csv' and 'test.csv' files, respectively
with open(train_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['path', 'label'])
    writer.writerows(train_images)

with open(test_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['path', 'label'])
    writer.writerows(test_images)

"""**This code reads in the train and test CSV files and then counts the frequency of each label in both datasets using the value_counts() method from pandas. It then prints out the frequency of each label in both datasets. This can be useful for understanding the distribution of the data and whether there is a class imbalance that needs to be addressed.**"""

# read train and test csv files
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# count frequency of each label in train data
train_counts = train_df['label'].value_counts()

# count frequency of each label in test data
test_counts = test_df['label'].value_counts()

# print frequency of each label in train data
print("Train data label frequencies:")
print(train_counts)

# print frequency of each label in test data
print("\nTest data label frequencies:")
print(test_counts)

"""**This code defines a function preprocess_image which takes an image path as input, loads the image using load_img function from tensorflow.keras.utils, and resizes the image to (224, 224) pixels. The function returns the image as a numpy array.**

**Another function create_dataset is defined, which takes a dataframe as input, iterates over each row of the dataframe, and extracts the image path and label information. It then calls the preprocess_image function to preprocess each image and appends it to a list X, and appends the corresponding label to a list Y. Finally, the function converts the X and Y lists into numpy arrays, and applies one-hot encoding to the Y array using to_categorical function from tensorflow.keras.utils.**

**The code then loads the train.csv file using pandas, calls the create_dataset function on the train data to create the training dataset, and prints the shape of the resulting X_train and Y_train arrays.**
"""

def preprocess_image(path):
    img = load_img(path, target_size=(224, 224))
    img_array = img_to_array(img)
    return img_array

def create_dataset(df):
    X = []
    Y = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row['path']
        label = row['label']
        img = preprocess_image(img_path)
        X.append(img)
        Y.append(label)
    X = np.array(X)
    Y =  to_categorical(np.array(Y) - 1, num_classes=5)
    return X, Y

# Load train.csv
train_df = pd.read_csv('train.csv')

# Create train dataset
X_train, Y_train = create_dataset(train_df)

# Print the shape of X_train and Y_train
print("Shape of X_train:", X_train.shape)
print("Shape of Y_train:", Y_train.shape)

"""**This code loads the 'test.csv' file and creates a test dataset by calling the 'create_dataset' function. It then prints the shape of the test dataset, which consists of the feature vectors X_test and the corresponding labels Y_test.**"""

# Load train.csv
test_df = pd.read_csv('test.csv')

# Create train dataset
X_test, Y_test = create_dataset(test_df)

# Print the shape of X_train and Y_train
print("Shape of X_train:", X_test.shape)
print("Shape of Y_train:", Y_test.shape)

"""**This code defines a convolutional neural network (CNN) architecture using Keras API. The CNN model has three convolutional layers with increasing number of filters, each followed by a max pooling layer to reduce the spatial dimensions. The output of the convolutional layers is then flattened and passed to a fully connected layer with 512 units and a ReLU activation function. A dropout layer is added to reduce overfitting, followed by an output layer with 5 units and a softmax activation function. The model is compiled with categorical crossentropy loss, Adam optimizer, and accuracy metric. The summary of the model is printed using the summary() method.**

insipired by https://www.tensorflow.org/tutorials/images/cnn
"""

def create_model2():
  # Define the model architecture
  model = Sequential()

  # Add the first convolutional layer
  model.add(Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
  model.add(MaxPooling2D(pool_size=(2,2)))

  # Add the second convolutional layer
  model.add(Conv2D(64, (3,3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  # Add the third convolutional layer
  model.add(Conv2D(128, (3,3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  # Flatten the output of the convolutional layers
  model.add(Flatten())

  # Add a fully connected layer with 512 units and a relu activation function
  model.add(Dense(512, activation='relu'))

  # Add a dropout layer to reduce overfitting
  model.add(Dropout(0.5))

  # Add the output layer with 5 units (one for each class) and a softmax activation function
  model.add(Dense(5, activation='softmax'))

  # Compile the model with categorical crossentropy loss, adam optimizer, and accuracy metric
  return model

cnn_model = create_model2()
print(cnn_model.summary())

"""**This code defines a callback reduce_lr which reduces the learning rate of the optimizer when the validation loss stops improving. The monitor parameter specifies the metric to monitor, factor specifies the factor by which the learning rate will be reduced, and patience specifies the number of epochs to wait before reducing the learning rate.The optimizer variable is initialized with the Adam optimizer with a learning rate of 0.001.Finally, the cnn_model is compiled with the optimizer, loss set to Categorical Crossentropy, and metrics set to accuracy.**"""

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), patience=5)
optimizer = Adam(learning_rate=0.001)
cnn_model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(), metrics=['accuracy'])

"""**This code trains a convolutional neural network (cnn_model) on the training data (X_train and Y_train) for 10 epochs, with validation data (X_test and Y_test). The training progress is printed out (verbose=2) and the learning rate is reduced using the ReduceLROnPlateau callback after 5 epochs of no improvement in validation loss. The training history is saved in history.**"""

history = cnn_model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test) ,
                       verbose=2,
                       callbacks=[reduce_lr])

# Get the training accuracy and validation accuracy from the training history
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Get the training loss and validation loss from the training history
train_loss = history.history['loss']
val_loss = history.history['val_loss']

"""**The code plots two subplots using matplotlib to show the training and validation accuracy and loss over the 10 epochs of training the convolutional neural network model. The first subplot shows the accuracy and the second subplot shows the loss. The x-axis represents the epochs and the y-axis represents the accuracy or loss values. The blue line represents the training set and the orange line represents the validation set. The graphs show how the accuracy improves and the loss decreases with each epoch.**"""

import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

ax[0].set_title('Training Accuracy vs. Epochs')
ax[0].plot(train_accuracy, 'o-', label='Train Accuracy')
ax[0].plot(val_accuracy, 'o-', label='Validation Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[0].legend(loc='best')

ax[1].set_title('Training/Validation Loss vs. Epochs')
ax[1].plot(train_loss, 'o-', label='Train Loss')
ax[1].plot(val_loss, 'o-', label='Validation Loss')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
ax[1].legend(loc='best')

plt.tight_layout()
plt.show()

predictions = cnn_model.predict(X_test)

"""**This code is creating two lists, y_true and y_pred, which will store the true and predicted class labels for each image in the test set. It then loops through each image in the test set and predicts its class label using the trained model. If the predicted class matches the true class, it prints a message indicating that they are the same. Finally, it appends the true and predicted class labels to y_true and y_pred respectively. These lists can be used to evaluate the performance of the model on the test set using metrics such as accuracy, precision, and recall.**"""

y_true=[]
y_pred =[]
for  i in range(X_test.shape[0]):
  predicted_label = np.argmax(predictions[i])
  predicted_class = predicted_label + 1
 
  original_label = np.argmax(Y_test[i])
  original_class = predicted_label + 1

  if predicted_class==original_class:
    print(predicted_class,'==',original_class)
  y_true.append(original_class)
  y_pred.append(predicted_class)

"""**This code creates a confusion matrix using the predicted and true labels of the test set. It then uses seaborn to create a heatmap visualization of the matrix, where the diagonal shows the number of correctly classified samples for each class, and the off-diagonal elements represent misclassifications. The heatmap is annotated with the counts for each cell. Finally, the plot is labeled with x and y axis indicating predicted and true classes respectively.**"""

import seaborn as sns
import matplotlib.pyplot as plt

# create confusion matrix
cf_mtx = confusion_matrix(y_true, y_pred)

# plot heatmap
sns.heatmap(cf_mtx, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()