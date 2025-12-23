import numpy as np
import itertools
import pandas as pd
from sklearn.metrics import confusion_matrix
import time

start_time = time.time()


data = pd.read_csv(r"C:/Users/3line/OneDrive/Desktop/Task3/cardio_train.csv", sep=';')

# data = pd.read_csv(r"C:/Users/3line/OneDrive/Desktop/Task3/500_Person_Gender_Height_Weight_Index.csv")

# print(data)
train_size = 0.9 * len(data)
# print(train_size)
train_data = data.iloc[:int(train_size),:]
test_data = data.iloc[int(train_size):,:]
test_labels =data.iloc[int(train_size):,-1:]

max_depth = 5
min_samples_split = 20
min_information_gain  = 1e-5

def gini_impurity(feature):

  p = feature.value_counts()/feature.shape[0]
  gini = 1-np.sum(p**2)
  return(gini)
  
def entropy(feature):

  a = feature.value_counts()/feature.shape[0]
  entropy = np.sum(-a*np.log2(a))
  return(entropy)

def information_gain(feature, mask, function):
  
  a = sum(mask)
  b = mask.shape[0] - a
  
  if(a == 0 or b ==0): 
    ig = 0
  
  else:
    ig = function(feature)-a/(a+b)*function(feature[mask])-b/(a+b)*function(feature[-mask])
  
  return ig

def categorical_options(a):

  a = a.unique()

  options = []
  for L in range(0, len(a)+1):
      for subset in itertools.combinations(a, L):
          subset = list(subset)
          options.append(subset)

  return options[1:-1]

def max_information_gain_split(x, y, function):

  split_value = []
  ig = [] 

  numeric_variable = True if x.dtypes != 'O' else False

  # Create options according to variable type
  if numeric_variable:
    options = x.sort_values().unique()[1:]
  else: 
    options = categorical_options(x)

  # Calculate ig for all values
  for val in options:
    mask =   x < val if numeric_variable else x.isin(val)
    val_ig = information_gain(y, mask, function)
    # Append results
    ig.append(val_ig)
    split_value.append(val)

  # Check if there are more than 1 results if not, return False
  if len(ig) == 0:
    return(None,None,None, False)

  else:
  # Get results with highest IG
    best_ig = max(ig)
    best_ig_index = ig.index(best_ig)
    best_split = split_value[best_ig_index]
    return(best_ig,best_split,numeric_variable, True)

def get_best_split(y, data,method):

  masks = data.drop(y, axis= 1).apply(max_information_gain_split, y = data[y], function = method)
  if sum(masks.loc[3,:]) == 0:
    return(None, None, None, None)

  else:
    # Get only masks that can be splitted
    masks = masks.loc[:,masks.loc[3,:]]

    # Get the results for split with highest IG
    split_variable = max(masks)
    #split_valid = masks[split_variable][]
    split_value = masks[split_variable][1] 
    split_ig = masks[split_variable][0]
    split_numeric = masks[split_variable][2]

    return(split_variable, split_value, split_ig, split_numeric)

def make_split(variable, value, data, is_numeric):

  if is_numeric:
    data_1 = data[data[variable] < value]
    data_2 = data[(data[variable] < value) == False]

  else:
    data_1 = data[data[variable].isin(value)]
    data_2 = data[(data[variable].isin(value)) == False]

  return(data_1,data_2)

def make_prediction(data, target_factor):

  # Make predictions
  if target_factor:
    pred = data.value_counts().idxmax()
  else:
    pred = data.mean()

  return pred

def train_tree(data,y, target_factor,function, max_depth = None,min_samples_split = None, min_information_gain = 1e-20, counter=0, max_categories = 20):

  # Check that max_categories is fulfilled
  if counter==0:
    types = data.dtypes
    check_columns = types[types == "object"].index
    for column in check_columns:
      var_length = len(data[column].value_counts()) 
      if var_length > max_categories:
        raise ValueError('The variable ' + column + ' has '+ str(var_length) + ' unique values, which is more than the accepted ones: ' +  str(max_categories))

  # Check for depth conditions
  if max_depth == None:
    depth_cond = True

  else:
    if counter < max_depth:
      depth_cond = True

    else:
      depth_cond = False

  # Check for sample conditions
  if min_samples_split == None:
    sample_cond = True

  else:
    if data.shape[0] > min_samples_split:
      sample_cond = True

    else:
      sample_cond = False

  # Check for ig condition
  if depth_cond & sample_cond:

    var,val,ig,var_type = get_best_split(y, data,function)

    # If ig condition is fulfilled, make split 
    if ig is not None and ig >= min_information_gain:

      counter += 1

      left,right = make_split(var, val, data,var_type)

      # Instantiate sub-tree
      split_type = "<=" if var_type else "in"
      question =   "{} {}  {}".format(var,split_type,val)
      # question = "\n" + counter*" " + "|->" + var + " " + split_type + " " + str(val) 
      subtree = {question: []}


      # Find answers (recursion)
      yes_answer = train_tree(left,y, target_factor,function, max_depth,min_samples_split,min_information_gain, counter)

      no_answer = train_tree(right,y, target_factor,function, max_depth,min_samples_split,min_information_gain, counter)

      if yes_answer == no_answer:
        subtree = yes_answer

      else:
        subtree[question].append(yes_answer)
        subtree[question].append(no_answer)

    # If it doesn't match IG condition, make prediction
    else:
      pred = make_prediction(data[y],target_factor)
      return pred

   # Drop dataset if doesn't match depth or sample conditions
  else:
    pred = make_prediction(data[y],target_factor)
    return pred

  return subtree

def classifing_nodes(observation, dtree):
  question = list(dtree.keys())[0] 

  if question.split()[1] == '<=':

    if observation[question.split()[0]] <= float(question.split()[2]):
      answer = dtree[question][0]
    else:
      answer = dtree[question][1]

  else:

    if observation[question.split()[0]] in (question.split()[2]):
      answer = dtree[question][0]
    else:
      answer = dtree[question][1]

  # If the answer is not a dictionary
  if not isinstance(answer, dict):
    return answer
  else:
    residual_tree = answer
    return classifing_nodes(observation, answer)

def Predict(test_features,decisiones_tree):
  prediction_labels = []

  for i in range(len(test_features)):
    obs_pred = classifing_nodes(test_features.iloc[i,:], decisiones_tree)
    prediction_labels.append(obs_pred)
  return prediction_labels

def evaluate_Performance(test_features_labels,prediction_labels):
  # testlabels = test_features_labels.to_numpy().flatten()
  # accuracy = (np.sum(testlabels==prediction_labels)/len(test_features_labels)) * 100
  # return accuracy
  TN, FP, FN, TP = confusion_matrix(test_features_labels,prediction_labels).ravel()
  # C_M = confusion_matrix(test_features_labels,prediction_labels)
  # TN=C_M[0][0]  
  # FP=C_M[0][1]  
  # FN=C_M[1][0]  
  # TP=C_M[1][1] 
  accuracy =  (TP+TN) /(TP+FP+TN+FN)

  return accuracy

entropy_decisiones_tree = train_tree(train_data,'cardio',True,entropy, max_depth,min_samples_split,min_information_gain)

entropy_prediction_labels = Predict(test_data,entropy_decisiones_tree)

gini_decisiones_tree = train_tree(train_data,'cardio',True,gini_impurity, max_depth,min_samples_split,min_information_gain)

gini_prediction_labels = Predict(test_data,gini_decisiones_tree)


entropy_accuracy =  evaluate_Performance(test_labels,entropy_prediction_labels)
print("Accuray Based on Entropy",entropy_accuracy)

gini_accuracy =  evaluate_Performance(test_labels,gini_prediction_labels)
print("Accuray Based on Gini Impurity",gini_accuracy)

end_time = time.time()
print("Total run time: ",(end_time-start_time)/60)