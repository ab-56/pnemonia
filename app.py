import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split


from datasets import load_dataset


dataset = load_dataset("Abeeha007/chest-x-rays")


print(dataset)


def predict_pneumonia(img_path, model, class_labels):
    # Preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))  
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]  
    predicted_class = class_labels[predicted_class_idx]  
    print(f"Prediction: {predicted_class} (Confidence: {predictions[0][predicted_class_idx]:.2f})")
    return predicted_class


    class_labels = {0: 'Normal', 1: 'Pneumonia'}


from sklearn.model_selection import train_test_split

from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Load dataset from Hugging Face
dataset = load_dataset("Abeeha007/chest-x-rays")

# Access the 'train' split
train_data = dataset['train']

# Convert dataset to a list of examples
train_data_list = [example for example in train_data]

# Now you can use train_test_split
train_data_split, test_data_split = train_test_split(train_data_list, test_size=0.2)

# Now you have train_data_split, test_data_split
print(f"Train size: {len(train_data_split)}")
print(f"Test size: {len(test_data_split)}")

# Extract image paths and labels
image_paths = [example['image_path'] for example in train_data_list]
labels = [example['label'] for example in train_data_list]

# Split the data
image_paths_train, image_paths_test, labels_train, labels_test = train_test_split(image_paths, labels, test_size=0.2)

# Now you have train and test data (paths and labels)




# Access the 'train' split
train_data = dataset['train']

# Manually create a test split (e.g., 80% train, 20% test)
train_data_split, test_data_split = train_test_split(train_data, test_size=0.2)

# Now you have train_data_split, test_data_split
print(f"Train size: {len(train_data_split)}")
print(f"Test size: {len(test_data_split)}")


# Access the train and test splits
train_data = dataset['train']
test_data = dataset['test']

# Split the train data into a new train/validation split (e.g., 80/20 split)
train_data_split, val_data_split = train_test_split(train_data, test_size=0.2)

# Now you have train_data_split, val_data_split, and test_data

    
# Access the train, validation, and test splits
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Convert the Hugging Face dataset to a list of images and labels (or process accordingly)
train_images = [example['image'] for example in train_data]  # Assuming dataset has 'image' field
train_labels = [example['label'] for example in train_data]  # Assuming dataset has 'label' field

val_images = [example['image'] for example in val_data]
val_labels = [example['label'] for example in val_data]


train_datagen = ImageDataGenerator(rescale=1.0/255, rotation_range=20, zoom_range=0.2, horizontal_flip=True,shear_range=0.2
    )
val_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(224, 224), batch_size=32, class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=32, class_mode='binary'
)

from tensorflow.keras.applications import MobileNetV2


base_model = MobileNetV2(weights='/kaggle/input/mobile-v2-1-0-224/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5', 
                         include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout


model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),  
    Dropout(0.5),  
    Dense(1, activation='sigmoid')  
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

import os


print("Train directory content:", os.listdir(train_dir))
print("Validation directory content:", os.listdir(val_dir))
print("Test directory content:", os.listdir(test_dir))

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)


class_weights = {i: weight for i, weight in enumerate(class_weights)}


if 1 in class_weights:
    class_weights[1] = class_weights[1] * 2  


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=10,  #
    class_weight=class_weights  
)

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.2f}")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.show()

true_labels = test_generator.classes
predicted_probs = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)
predicted_labels = np.argmax(predicted_probs, axis=1)

import numpy as np
from sklearn.metrics import classification_report


test_generator.reset()  
predictions = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)


predicted_labels = (predictions > 0.5).astype(int)


true_labels = test_generator.classes

# Step 3: Classification report
report = classification_report(true_labels, predicted_labels, target_names=['Normal', 'Pneumonia'])

print("Classification Report:")
print(report)

true_labels = test_generator.classes


predicted_probs = model.predict(test_generator)


roc_auc = roc_auc_score(true_labels, predicted_probs)
print(f"ROC-AUC Score: {roc_auc:.2f}")

true_labels = test_generator.classes


predicted_probs = model.predict(test_generator)



fpr, tpr, _ = roc_curve(true_labels, predicted_probs)

roc_auc = auc(fpr, tpr)


plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

img_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA/person1951_bacteria_4882.jpeg'  # Path to the X-ray image
predicted_class = predict_pneumonia(img_path, model, class_labels)
print(f"The X-ray image is classified as: {predicted_class}")
