import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import cv2
from skimage.feature import hog
from skimage.transform import resize
from sklearn.model_selection import cross_val_score
import shap

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def extract_hog_features(image):
    hog_features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False
    )
    return hog_features

def load_and_extract_features(directory):
    X, y = [], []
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        for filename in os.listdir(label_dir):
            image_path = os.path.join(label_dir, filename)
            img = cv2.imread(image_path)
            img_resized = cv2.resize(img, (128, 128))
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            hog_features = extract_hog_features(img_gray)
            X.append(hog_features)
            y.append(label)
    return X, y


def getImageReset(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (128, 128))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog_features = extract_hog_features(img_gray)
    return hog_features



def getImagesData(df):
    flat_data_arr = []
    images = []

    for index, row in df.iterrows():

        if os.path.exists(row["fichiers"].replace("clip_raster_", "clip_raster__").replace("valdor", "val_dor")):
            img_array = getImageReset(
                row["fichiers"].replace("clip_raster_", "clip_raster__").replace("valdor", "val_dor"))
            img_resized = resize(img_array, (150, 150, 3))
            flat_data_arr.append(img_resized.flatten())
            images.append(row["fichiers"])

        elif os.path.exists(row["fichiers"]):
            img_array = getImageReset(row["fichiers"])
            img_resized = resize(img_array, (150, 150, 3))
            flat_data_arr.append(img_resized.flatten())
            images.append(row["fichiers"])

        else:
            basename = os.path.basename(row["fichiers"]).replace("clip_raster_", "clip_raster__").replace("20250927_valdor",
                                                                                                          "20250927_val_dor")
            picture = os.path.join(r"C:\Projets\sat_caribou\raster", basename)
            # img_array = imread(picture)
            # img_resized = resize(img_array, (150, 150, 3))
            # flat_data_arr.append(img_resized.flatten())

            img_array = getImageReset(picture)
            img_resized = resize(img_array, (150, 150, 3))
            flat_data_arr.append(img_resized.flatten())
            images.append(row["fichiers"])


    flat_data = np.array( flat_data_arr)



    df_pictures=pd.DataFrame(flat_data)
    df_pictures["fichiers"] = images

    return df_pictures


df = pd.read_csv('Data_Sat_Caribou_v16.csv')
df = df.fillna(0)
df = df.fillna(0)
df_potentielle = df.query("classement == 'potentiel'")
df_potentielle = df_potentielle.iloc[:2000]

#df_potentielle = df_potentielle.drop('classement',axis=1)

df_pas_potentielle = df.query("classement == 'pas potentiel'")
df_pas_potentielle = df_pas_potentielle.iloc[:2000]

newdf = pd.concat([df_potentielle,df_pas_potentielle])

df_pictures = getImagesData(newdf)
newdf = newdf.merge(df_pictures, left_on='fichiers', right_on='fichiers')
newdf = newdf.drop("fichiers", axis=1)

zero_max_column = []

for c in df.columns:

    if c != "classement" and  c != "fichiers":
        if newdf[c].max() == 0:
            zero_max_column.append(c)
print(zero_max_column)
for c in zero_max_column:
    newdf.drop(c, axis=1)



maps_classment = {"pas potentiel":0,"potentiel":1}
newdf['classement'] = newdf['classement'].map(maps_classment)

x,y = newdf.iloc[:,1:].values, newdf.iloc[:,0].values



# Splitting the data into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,
                                               random_state=77,
                                               stratify=y)

# Defining the parameters grid for GridSearchCV
param_grid={'C':[0.1,1,10,100],
            'gamma':[0.0001,0.001,0.1,1],
            'kernel':['rbf','poly']}

# Creating a support vector classifier
#svc=svm.SVC(probability=True)

#svm_linear = svm.SVC(kernel='linear')
svm_linear = LinearSVC()
#svm_rbf = svm.SVC(kernel='rbf')

#score_linear = cross_val_score(svm_linear, x_train,y_train, cv=5).mean()
#score_rbf = cross_val_score(svm_rbf, x_train,y_train, cv=5).mean()

#print("Score linéaire :", score_linear)
#print("Score RBF :", score_rbf)

# Creating a model using GridSearchCV with the parameters grid
#if score_rbf > score_linear:

#    model=GridSearchCV(svc,param_grid)

    # Training the model using the training data
#    model.fit(x_train,y_train)

    # Testing the model using the testing data
#    y_pred = model.predict(x_test)

#else:
svm_linear.fit(x_train,y_train)
y_pred = svm_linear.predict(x_test)

# Calculating the accuracy of the model
accuracy = accuracy_score(y_pred, y_test)

# Print the accuracy of the model
print(f"The model is {accuracy*100}% accurate")



from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()

explainer = shap.Explainer(svm_linear)

shap_values = explainer(x_test)

shap.summary_plot(shap_values, x_test, plot_type="bar")
#
print(shap_values)