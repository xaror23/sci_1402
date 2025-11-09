import pandas as pd
import os
import shutil

df = pd.read_csv('C:\Projets\sat_caribou\python\Data_Sat_Caribou_V9.csv')

print(df.to_string())

df_potentielle = df.query("classement == 'potentiel'")
df_potentielle = df_potentielle.iloc[:1400]

#df_potentielle = df_potentielle.drop('classement',axis=1)

df_pas_potentielle = df.query("classement == 'pas potentiel'")
df_pas_potentielle = df_pas_potentielle.iloc[:1400]



for index, row in df_potentielle.iterrows():
    fichier = row["fichiers"].replace("clip_raster_", "clip_raster__").replace("20250927_valdor", "20250927_val_dor")
    if row["classement"] == "potentiel":
        bn = os.path.basename(fichier)
        if os.path.isfile(fichier):
            shutil.copyfile(fichier, os.path.join(r"C:\Projets\sat_caribou\cnn\train\potentiel", bn))

    else:
        bn = os.path.basename(fichier)
        if os.path.isfile(fichier):
            shutil.copyfile(fichier, os.path.join(r"C:\Projets\sat_caribou\cnn\train\pas_potentiel", bn))


for index, row in df_pas_potentielle.iterrows():
    fichier = row["fichiers"].replace("clip_raster_", "clip_raster__").replace("valdor", "val_dor")
    if row["classement"] == "potentiel":
        bn = os.path.basename(fichier)
        if os.path.isfile(fichier):
            shutil.copyfile(fichier, os.path.join(r"C:\Projets\sat_caribou\cnn\train\potentiel", bn))

    else:
        bn = os.path.basename(fichier)
        if os.path.isfile(fichier):
            shutil.copyfile(fichier, os.path.join(r"C:\Projets\sat_caribou\cnn\train\pas_potentiel", bn))