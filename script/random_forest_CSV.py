from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.transform import resize
import random
import cv2
import pandas as pd
import numpy as np

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


df = pd.read_csv('Data_Sat_Caribou_v8.csv')
df = df.fillna(0)
print(df.columns)
df_potentielle = df.query("classement == 'potentiel'")
df_potentielle = df_potentielle.iloc[:1400]

#df_potentielle = df_potentielle.drop('classement',axis=1)

df_pas_potentielle = df.query("classement == 'pas potentiel'")
df_pas_potentielle = df_pas_potentielle.iloc[:1400]

newdf = pd.concat([df_potentielle,df_pas_potentielle])

theMatrix = []

for i in range(0, 150):
    rows = 150
    cols = 3
    zero_matrix = np.zeros((rows, cols))
    theMatrix.append(zero_matrix)



flat_data_arr = []
images = []

for index, row in newdf.iterrows():



    if os.path.exists(row["fichiers"].replace("clip_raster_","clip_raster__").replace("valdor","val_dor")):
        img_array = getImageReset(row["fichiers"].replace("clip_raster_","clip_raster__").replace("valdor","val_dor"))
        img_resized = resize(img_array, (150, 150, 3))
        flat_data_arr.append(img_resized.flatten())
        images.append(row["fichiers"])

    elif os.path.exists(row["fichiers"]):
        img_array = getImageReset(row["fichiers"])
        img_resized = resize(img_array, (150, 150, 3))
        flat_data_arr.append(img_resized.flatten())
        images.append(row["fichiers"])

    else:
        basename = os.path.basename(row["fichiers"]).replace("clip_raster_","clip_raster__").replace("20250927_valdor","20250927_val_dor")
        picture = os.path.join(r"C:\Projets\sat_caribou\raster", basename)
        #img_array = imread(picture)
        #img_resized = resize(img_array, (150, 150, 3))
        #flat_data_arr.append(img_resized.flatten())

        img_array = getImageReset(picture)
        img_resized = resize(img_array, (150, 150, 3))
        flat_data_arr.append(img_resized.flatten())
        images.append(row["fichiers"])


from sklearn.model_selection import train_test_split

index = 0
flat_data = np.array( flat_data_arr)



df_pictures=pd.DataFrame(flat_data)
df_pictures["fichiers"] = images
#csv_features = df_test.drop(columns=["classement"]).values
#X = np.hstack((newdf, df_pictures))
newdf = newdf.merge(df_pictures, left_on='fichiers', right_on='fichiers')
newdf = newdf.drop("fichiers", axis=1)

#f_t_r = ['diff_altitue', 'pts_plus_eleve', 'couvert_surface_dominant', '_num_groupe_essenece_2_2', '_num_groupe_essenece', '_num_surface_2_2', '_num_surface', '_num_type_terrain_2_2', '_num_type_terrain', '_num_surface_2_2_2', '_num_surface_2', '_num_drainage_2_2_2', '_num_drainage_2', '_num_type_ecologie_2_2_2', '_num_type_ecologie_2', '_num_type_terrain_2_2_2', '_num_type_terrain_2']
#for f in f_t_r:
#    newdf = newdf.drop(f, axis=1)


maps_classment = {"pas potentiel":0,"potentiel":1}
newdf['classement'] = newdf['classement'].map(maps_classment)
X = np.hstack((newdf, df_pictures))

x,y = newdf.iloc[:,1:].values, newdf.iloc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)

# import Random Forest classifier

from sklearn.ensemble import RandomForestClassifier



# instantiate the classifier

rfc = RandomForestClassifier(n_estimators=100,random_state=0)



# fit the model

rfc.fit(X_train, y_train)



# Predict the Test set results

y_pred = rfc.predict(X_test)



# Check accuracy score

from sklearn.metrics import accuracy_score

print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


#habitat_train_X, habitat_train_y = load_and_extract_features(
#    r'C:\Projets\sat_caribou\entrainement')

#clf = RandomForestClassifier(max_depth=2, random_state=0)

#habitat_rf_classifier = clf.fit(habitat_train_X, habitat_train_y)


#habitat_test_X, habitat_test_y = load_and_extract_features(
#    r'C:\Projets\sat_caribou\test')


#habitat_predictions = habitat_rf_classifier.predict(habitat_test_X)

#habitat_accuracy = accuracy_score(habitat_test_y, habitat_predictions)



from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Get feature importances
importances = rfc.feature_importances_

# Create a DataFrame for better visualization
colums = newdf.columns
colums = colums[1:]
feature_importance_df = pd.DataFrame({'feature': colums, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

print("Feature Importances:")
print(feature_importance_df)

# Access the individual trees
individual_trees = rfc.estimators_

# You can then work with each individual tree, for example:
print(f"Number of trees in the forest: {len(individual_trees)}")
print(f"Type of the first tree: {type(individual_trees[0])}")

# You can also inspect properties of a single tree, like its depth
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Visualize the first tree (optional)
plt.figure(figsize=(10, 8))

#for tree in individual_trees:
#    plot_tree(tree)
#    plt.title("First Decision Tree in the Random Forest")
#    plt.show()