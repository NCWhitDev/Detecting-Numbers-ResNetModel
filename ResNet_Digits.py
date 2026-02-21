import numpy as np # For numerical operations
import struct # To read the binary file format of the dataset
import matplotlib.pyplot as plt # To visualize images
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # Clear prompt in terminal
import tensorflow as tf
from tensorflow.keras import layers, models, Input

# Source: https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download

train_images_path = "/VSCODE/Python/Neural Networks/archive/train-images-idx3-ubyte/train-images-idx3-ubyte"
train_labels_path = "/VSCODE/Python/Neural Networks/archive/train-labels-idx1-ubyte/train-labels-idx1-ubyte"
test_images_path = "/VSCODE/Python/Neural Networks/archive/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
test_labels_path = "/VSCODE/Python/Neural Networks/archive/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"

# We need to read the IDX file format by MNIST (read only and Binary files rb)
def read_IDX(filename):
    with open(filename, 'rb') as f:
        # Read the header to understand file structure (magic number, dimensions, etc.)
        zero, data_type, dimensions = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dimensions))
        # Read the rest of the data and reshape it
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

# Load Data ------------------------------------------------------------------------------------------
try:
    x_train = read_IDX(train_images_path)
    y_train = read_IDX(train_labels_path)
    x_test = read_IDX(test_images_path)
    y_test = read_IDX(test_labels_path)

    print("Data Loaded successfully!!")
    print(f"Traing Data Shape: {x_train.shape}")
    print(f"Testing Data Shape: {y_train.shape}")
    print(f"Test Data Shape: {x_test.shape}")
    print(f"Test Labels Shape: {y_test.shape}")

except FileNotFoundError:
    print("ERROR: File not found. Please check dataset location.")
    print("download data set here: https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download")

# Analyze images ------------------------------------------------------------------------------------
""" 
    The images are 28x28 pixels. 
    The labels are numbers from 0 to 9.
"""
plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2,5, i+1)
    plt.imshow(x_train[i], cmap='gray') # Show as gray
    plt.title(f"Label: {y_train[i]}") # Labels
    plt.axis('off')
plt.show()

# Data Preprocessing -------------------------------------------------------------------------------
"""
    Neural Networks work best when numbers are small (between 0 and 1) and shapes are specific.
    Currently, pixel values are between 0 and 255. We divide by 255 to make them 0-1.
    Convolutional Neural Networks (CNNs) expect images to have 3 dimensions: (Height, Width, Channels). 
    Since MNIST is grayscale, we have 1 channel. We need to reshape our data from (28, 28) to (28, 28, 1).
"""
# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape to add channel dimensions (28, 28) -> (28, 28, 1)
x_train = np.expand_dims(x_train, axis= -1)
x_test = np.expand_dims(x_test, axis= -1)

# Convert Labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

print(f"New shape of x_train: {x_train.shape}")

# ResNet Model ----------------------------------------------------------------------------------
"""
    Why a ResNet Model? Normal Neural Network models tend to stop learning the further the network goes
    a thing called "The Vanishing Gradient problem".

    Residual Networks solve this by using "Skip connections" Allow us to remember the input.
"""

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x

    # First Convolution Layer
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Second Convolution Layer
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Adjust shortcut if dimensions change
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add the shortcut to the main path
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def build_Residual_Networks_Model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Initial Convolution
    x = layers.Conv2D(32, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Stacking Residual Blocks
    x = residual_block(x, 32)
    x = residual_block(x, 64, stride=2)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ResNet_NCWhit")
    return model

# Building the Model
ResNetModel = build_Residual_Networks_Model(input_shape=(28,28,1), num_classes=10)
ResNetModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train ResNet Model -----------------------------------------------------------------------------
history = ResNetModel.fit(
    x_train, y_train,
    epochs=10,              # How many full runs through the dataset
    batch_size=64,          # How many images processed at once
    validation_split=0.1,   # We keep 10% of training data aside to check performance during training.
    verbose=1
)

# Plot accuracy and loss
plt.figure(figsize=(12, 4))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
test_loss, test_acc = ResNetModel.evaluate(x_test, y_test, verbose=0)
print(f"Final Test Accuracy: {test_acc*100:.2f}%")

# Predictions ----------------------------------------------------------------------------------
# Predict on the test set
predictions = ResNetModel.predict(x_test)

plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i+1)
    
    # Reshape back to 28x28 for display
    img = x_test[i].reshape(28, 28)
    
    plt.imshow(img, cmap='gray')
    
    # Get the predicted label (index with highest probability)
    predicted_label = np.argmax(predictions[i])
    true_label = np.argmax(y_test[i])
    
    color = 'green' if predicted_label == true_label else 'red'
    
    plt.title(f"Pred: {predicted_label}\nTrue: {true_label}", color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()
