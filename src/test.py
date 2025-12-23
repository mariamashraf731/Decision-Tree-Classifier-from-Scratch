import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import time

start_time = time.time()

data = pd.read_csv(r"C:/Users/3line/OneDrive/Desktop/Task3/cardio_train.csv", sep=';')

# data = pd.read_csv(r"C:/Users/3line/OneDrive/Desktop/Task3/500_Person_Gender_Height_Weight_Index.csv")
# print(data)
train_size = 0.9 * len(data)

# le=LabelEncoder()

# for i in data:
#     data[i]=le.fit_transform(data[i])

# print(data)
# print(train_size)
train_data = data.values[:int(train_size),:-1]
test_data = data.values[int(train_size):,:-1]
train_labels = data.values[:int(train_size),-1:]
test_labels =data.values[int(train_size):,-1:]

model=DecisionTreeClassifier(criterion="gini",max_depth=5,min_samples_split=20)
model.fit(train_data,train_labels)
pred_labels= model.predict(test_data)

# TN, FP, FN, TP = confusion_matrix(test_labels,pred_labels).ravel()

C_M = confusion_matrix(test_labels,pred_labels)

TN=C_M[0][0]  
FP=C_M[0][1]  
FN=C_M[1][0]  
TP=C_M[1][1] 

# print(TN,FP,FN,TP)
accuracy =  (TP+TN) /(TP+FP+TN+FN)
# print("Accuray Based on Entropy",accuracy)

print("Accuray Based on Gini Impurity",accuracy)


end_time = time.time()
print("run time without average: ",(end_time-start_time)/60)


# data.head()



# print(data)

# print(
#   " Misclassified when cutting at 100kg:",
#   data.loc[(data['Weight']>=100) & (data['obese']==0),:].shape[0], "\n",
#   "Misclassified when cutting at 80kg:",
#   data.loc[(data['Weight']>=80) & (data['obese']==0),:].shape[0]
# )

# gini = gini_impurity(data.Gender)
# entropy = entropy(data.Gender)   
# ig = information_gain(data['obese'], data['Gender'] == 'Male')

# weight_ig, weight_slpit, _, _ = max_information_gain_split(data['Weight'], data['obese'],)  
# print(
#   "The best split for Weight is when the variable is less than ",
#   weight_slpit,"\nInformation Gain for that split is:", weight_ig
# )

# data.drop('obese', axis= 1).apply(max_information_gain_split, y = data['obese'])


# decisiones_tree = train_tree(data,'obese',True, max_depth,min_samples_split,min_information_gain)

# # print(decisiones)

# print("Predictions: ",entropy_prediction_labels,
# "\n\nReal values:", test_labels.to_numpy())


# data['cardio'] = (data.Index >= 4).astype('int')
# data.drop('Index', axis = 1, inplace = True)
