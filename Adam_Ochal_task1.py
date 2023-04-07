import os
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
import joblib
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report

""" Definition of the source training data directory """
train_dir = 'data/'

""" Definition of the source validation data directory """
validation_dir = 'data_val/'

""" Definition of the source test data directory """
test_dir = 'data_test'

""" Definition of image size. All photos """
IMG_SIZE = (50, 50)

""" Definition of labels """
class_names = sorted(os.listdir(train_dir))

""" Definition of labels """
NUM_CLASSES = len(class_names)

""" Definition of batch size """
batch_size = 20

""" Definition of epochs number """
epochs = 50

""" Definition of train_datagen - ImageDataGenerator. 
Used for dynamic modification and transformation of the data. 
Include:
rescale, rotation_range(rotate image during training),shift of the photo
shear_range - transforming data to observe them from a different perspective.
horizontal_flip and vertical_flip for mirrorring photo
"""
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
)

""" Definition of validation_datagen - ImageDataGenerator - for validation data """
validation_datagen = ImageDataGenerator(rescale=1./255)

""" Loading training data """
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=batch_size,
    class_mode='categorical',
    classes=class_names)

""" Loading validation data """
validation_data = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=IMG_SIZE,
    batch_size=batch_size,
    class_mode='categorical',
    classes=class_names)

""" Definition of the model """
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

""" Model Compile with learning_rate=0.0001 """
optimizer = RMSprop(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

""" Model training """
history = model.fit(
    train_data,
    steps_per_epoch=len(train_data),
    epochs=epochs,
    validation_data=validation_data,
    validation_steps=len(validation_data)
)

""" Model save to file model1.h5 """
model.save('model1.h5')

""" Model history save to file model_history """
joblib.dump(history, 'model_history')

""" Model load from file model1.h5 """
model = load_model('model1.h5')


""" Generate plot for accuracy """
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

""" Generate plot for accuracy """
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

""" Definition of test_datagen - ImageDataGenerator - for testing model"""
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE[0], IMG_SIZE[1]),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

""" Prediction of test data """
Y_pred = model.predict(test_generator, test_generator.samples // batch_size)

""" Compute the highest probability class index for each input sample in Y_pred """
y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(test_generator.classes, y_pred)

""" Show Confusion Matrix """
print('\nConfusion Matrix')
print(cm)
print('\nClassification Report')

y_pred = model.predict(test_generator)
y_pred_labels = np.argmax(y_pred, axis=1)

"""Generate  precision,recall,f1-score,support values"""
target_names = list(test_generator.class_indices.keys())
print(classification_report(test_generator.classes, y_pred_labels, target_names=target_names,
                            output_dict=False, zero_division=1))

"""compute accuracy for each class"""
class_acc = np.zeros(cm.shape[0])
for i in range(cm.shape[0]):
    class_acc[i] = cm[i, i] / np.sum(cm[i, :])
    print(f"Accuracy for class {i}: {class_acc[i]:.2f}")

"""Generate random list of file to predict and print on plot"""
files = []
for cl in class_names:
    file_to_add = test_dir + '\\' + cl + '\\'
    files_to_check = os.listdir(file_to_add)
    for file in files_to_check:
        files.append([cl, file_to_add + file])
list_ = random.sample(range(0, len(files)-1), 10)

"""Display image with the real and predicted value"""
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
axs = axs.flatten()

for i, image_name in enumerate(list_):
    """Load image"""
    img_path = files[image_name][1]
    print(img_path)
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    """Predict Value"""
    pred = model.predict(img)
    pred_class = np.argmax(pred, axis=-1)

    """Show plot for each value"""
    axs[i].imshow(image.load_img(img_path))
    axs[i].axis('off')
    true_label = files[image_name][0]
    pred_label = class_names[pred_class[0]]
    axs[i].set_title(f"True label: {true_label}, Pred label: {pred_label}")

""""Show plot with predicted Value"""
plt.show()
