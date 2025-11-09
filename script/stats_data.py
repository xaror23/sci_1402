import pandas as pd
#imports Matplotlib library and assigns shorthand 'plt'
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pearsonr
from matplotlib.colors import ListedColormap

def plot_decision_regions(x, y , classifier, test_idx=None, resolution=0.02):

    markers = ('o','s','^','v','<')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap= ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = x[:,0].min() - 1, x[:,0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)

    #plt.contourf(xx1, xx2, lab, alphat=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0],y=x[y ==cl, 1],alpha=0.8, c=colors[idx], marker=markers[idx],label=f'Class {cl}',edgecolor='black')



df = pd.read_csv('Data_Sat_Caribou_v8.csv')
df = df.fillna(0)


print(df.columns)

df_potentielle = df.query("classement == 'potentiel'")
#df_potentielle = df_potentielle.drop('classement',axis=1)

df_pas_potentielle = df.query("classement == 'pas potentiel'")
#df_pas_potentielle = df_pas_potentielle.drop('classement',axis=1)

print(df_potentielle)

# Generates grouped data
#data_group1 = [df_potentielle["sup_feu"]]
#data_group2 = [ df_pas_potentielle["sup_feu"]]
# Combines two data groups into a dataset
#data = data_group1 + data_group2
# Creates grouped boxplots
#plt.boxplot(data, positions=[1, 2], labels=['Potentiel', 'Pas Potentiel'])
#plt.title('Grouped Boxplots')
#plt.xlabel('Group-Dataset')
#plt.ylabel('Value')
#plt.show()

#from sklearn.model_selection import train_test_split

#x,y = df.iloc[:,1:].values, df.iloc[:,0].values

#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, stratify=y, random_state=0)

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#x_train_std = sc.fit_transform(x_train)

#from sklearn.linear_model import LogisticRegression
#from sklearn.decomposition import PCA
#
#pca = PCA(n_components=2)
#lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')

#x_train_pca = pca.fit_transform(x_train_std)

#lr.fit(x_train_pca, y_train)

#plot_decision_regions(x_train_pca, y_train, classifier=lr)

#plt.xlabel('PC 1')
#plt.xlabel('PC 2')
#plt.legend(loc='lower left')
#plt.tight_layout()
#plt.show()



df = pd.read_csv('Data_Sat_Caribou_v8.csv')
df = df.fillna(0)

groups = df.groupby("classement").groups

potentiel = groups["potentiel"]
non_potentiel = groups["pas potentiel"]

#print(stats.f_oneway(potentiel, non_potentiel))

import statsmodels.api as sm
from statsmodels.formula.api import ols

plt.hist(df["classement"])
plt.show()
plt.savefig(rf"C:\Projets\sat_caribou\statistiques\histogrammes.png")
fields_reject=[]
for c in df.columns:
    print(c)
    if c != "classement" and c != "fichiers":
        #model = ols(c + ' ~ classement',                 # Model formula
        #            data = df).fit()

        #anova_result = sm.stats.anova_lm(model, typ=2)
        #with open('stats_anova.txt', 'a') as f:
        #    f.write(c + '\n')
        #    f.write(str(anova_result) + '\n')

        df2 = df
        maps_classment = {"pas potentiel":0,"potentiel":1}
        df_potentielle['classement'] = df_potentielle['classement'].map(maps_classment)
        df_pas_potentielle['classement'] = df_pas_potentielle['classement'].map(maps_classment)

        df['classement'] = df['classement'].map(maps_classment)

        #for c in df.columns:
        for group in [df_potentielle, df_pas_potentielle]:

            if c != "classement" and c != "fichiers":
                x = df[c]
                y = df["classement"]
                corr, _ = pearsonr(x, y)

                with open('coor_pearson.txt', 'a') as f:
                    f.write("#####" + c + '#####\n')
                    f.write("#####" + str(corr) + '#####\n')

        #    print(c)
        #    print(str(corr))

                with open('stats_descriptives.txt', 'a') as f:
                    f.write(c + '\n')
                    f.write('This is the second line.\n')

                    f.write("potentiel\n")
                    f.write(str(df_potentielle[c].describe()) + '\n')
                    f.write("pas potentiel\n")
                    f.write(str(df_pas_potentielle[c].describe()) + '\n')


        # Perform the paired samples t-test

        data_group1 = df_potentielle[c][:100]
        data_group2 = df_pas_potentielle[c][:100]
        t_statistic, p_value = stats.ttest_rel(data_group1, data_group2)

        #print(f"T-statistic: {t_statistic}")
        #print(f"P-value: {p_value}")

        # Interpret the result
        alpha = 0.05
        if p_value < alpha:
            analystet = "Reject the null hypothesis: There is a statistically significant difference between the paired samples."

        else :
            fields_reject.append(c)
            analystet ="Fail to reject the null hypothesis: There is no statistically significant difference between the paired samples."

        with open('test_t.txt', 'a') as f:
            f.write("#####" + c + '#####\n')
            f.write(str(t_statistic) + "\n")
            f.write(str(p_value) + "\n")
            f.write(analystet + "\n")

        # Combines two data groups into a dataset
        data_group1 = [df_potentielle[c]]
        data_group2 = [df_pas_potentielle[c]]
        data = data_group1 + data_group2
        # Creates grouped boxplots
        plt.boxplot(data, positions=[1, 2], labels=['Potentiel', 'Pas Potentiel'])
        plt.title('Diagramme à moustache Variable ' + c)
        plt.xlabel('Catégorie')
        plt.ylabel('Valeur')

        plt.savefig(rf"C:\Projets\sat_caribou\statistiques\{c}_stats.png")
        #plt.show()


print("faut que tules rejettes")
print(fields_reject)