import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import numpy as np
np.random.seed(123)
from keras.models import Sequential
import seaborn as sns

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

data = []
labels = []
classes = ["potentiel","pas_potentiel"]
cur_path = r"C:\Projets\sat_caribou\cnn"

for i in classes:
    path = os.path.join(cur_path, 'train', str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(path + '//' + a)
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            if i == "potentiel":
                labels.append(0)
            else:
                labels.append(1)

        except:
            print('Error loading image')

data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=2)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))



model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
#model.add(Dense(2, activation='softmax'))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 20
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
model.save("my_model.h5")

plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'])
label = 'val accuracy'
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

from sklearn.metrics import accuracy_score

#y_test = pd.read_csv('Test.csv')
#labels = y_test['ClassId'].values
#imgs = y_test['Path'].values

#data = []


classes = ["potentiel","pas_potentiel"]
cur_path = r"C:\Projets\sat_caribou\cnn"
data_test = []

#image = Image.open(path + '//' + a)
#image = image.resize((30, 30))
#image = np.array(image)
#data.append(image)

#for (root,dirs,files) in os.walk(r'C:\Projets\sat_caribou\classement\potentiel',topdown=True):
#    for file in files:
#        img = os.path.join(root, file)
#        image = Image.open(img)
#        image = image.resize((30, 30))
        #data_test.append(np.array(image))

#        X_test = np.array(image)
#        pred = model.predict([X_test])
#        print(pred)

#X_test = np.array(data_test)
#pred = model.predict_classes(X_test)
#print(pred)
#
#from sklearn.metrics import accuracy_score

#print(accuracy_score(labels, pred))
# confusion matrix
# Predict the value from the validation dataset
Y_pred = model.predict(X_test)
# Convert Predictions classes to one hot vectors
Y_pred_classes = np.argmax(Y_pred, axis = 1)
# Convert Validation observation to one hot vectors
Y_true = np.argmax(y_test, axis = 1)
# Compute the confision matrix
condision_mtrx = confusion_matrix(Y_true, Y_pred_classes)
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(condision_mtrx, annot = True, linewidths = 0.01, cmap = "Greens", linecolor = "gray", fmt = ".2f", ax = ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()