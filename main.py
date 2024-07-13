import sklearn
import sklearn.model_selection
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

#DATA SOURCE: https://www.kaggle.com/datasets/vinesmsuic/star-categorization-giants-and-dwarfs

#Vmag: visual apparent magnitude of the star
#Plx distance between the star and Earth
#e_Plx standard error of the plx
#B-V: B-V color index
#SpType: spectral type
#Amag: absolute magnitude of the star
#TargetClass: 0=Dwarf, 1=Giant

data = pd.read_csv("star type classification/Star3642_balanced.csv")

type_conversions = {
    0:"Dwarf",
    1:"Giant",
}
data["TargetClass"] = data["TargetClass"].replace(type_conversions) 

for i in range(len(data["e_Plx"])): #this loop is to take out any rows where the standard error is too high. In this particular dataset it isn't acutally needed because the standard error values are low, but good to have
    if data["e_Plx"][i] > 2:
        data.drop(i)

a = preprocessing.LabelEncoder() #label encoder object, used to transform string values into integers
Vmag = a.fit_transform(list(data["Vmag"])) #turns the L column into a list, and then returns an array (also converts to integer values if needed)
Plx = a.fit_transform(list(data["Plx"])) 
BVcolorindex= a.fit_transform(list(data["B-V"])) 
SpType = a.fit_transform(list(data["SpType"])) 
Amag= a.fit_transform(list(data["Amag"])) 
TargetClass = a.fit_transform(list(data["TargetClass"])) 

predict = "TargetClass" #label
x = list(zip(Vmag, Plx, BVcolorindex, SpType, Amag)) #zip produces tuples where each element in the tuple is a value from one of the columns
y = list(TargetClass)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size=0.1) #10% of the data is used for testing
bot = KNeighborsClassifier(n_neighbors=7)
bot.fit(x_train, y_train)
acc = (bot.score(x_test, y_test))*100

predictions = bot.predict(x_test) #returns an array
names = ["Dwarf", "Giant"]

#for x in range(len(predictions)):
    #print("predicted: ", names[predictions[x]], "acutal: ", names[y_test[x]])

cf_matrix = confusion_matrix(y_test, predictions) #creates a confusion matrix

total = len(data)
correct = 0
for i in range(len(predictions)):
    if predictions[i] == y_test[i]:
        correct += 1
percent_correct = (correct/total)*100
stats = "\n\nAccuracy={:0.3f}%\nTotal Stars={:0.3f}\nPercent Correct={:0.3f}%".format(
                acc,total,percent_correct)

sns.heatmap(cf_matrix, annot=True, cmap="BuPu", fmt=" ",
            xticklabels=names, yticklabels=names)
plt.xlabel("predicted classifications")
plt.ylabel("actual classifications")
plt.title(stats)
plt.show()



