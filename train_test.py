# %%
#Imported modules

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
import pickle

# %%
#Load dataset

anonymous_data = pd.read_csv('symtomps_disease.csv')

# %%
#Train && Test Phase

X = anonymous_data[['symtomps']]
y = anonymous_data[['disease']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# %%
#Performed OHE

OHE = OneHotEncoder()
OHE.fit_transform(X_train)
X_train_OHE = OHE.transform(X_train).toarray()

OHE_anonymous_data = pd.DataFrame(X_train_OHE, columns=OHE.get_feature_names(X_train.columns))

OHE_anonymous_data.head()

# %%
#Train the algorithm - Create the classifier, fit it on the training data and make predictions on the test set

clf = DecisionTreeClassifier(criterion='entropy')

clf.fit(X_train_OHE, y_train)

DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy', max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, random_state=None, splitter='best')

# %%
#Plot the decision tree
fig, axes = plt.subplots(nrows = 1,ncols = 1, figsize = (3,3), dpi=300)

tree.plot_tree(clf, feature_names = OHE_anonymous_data.columns, class_names=np.unique(y).astype('str'), filled = True)

plt.show()

# %%
#Evaluate the predictive performance
X_test_ohe = OHE.transform(X_test)
y_preds = clf.predict(X_test_ohe)

print('Accuracy: ', accuracy_score(y_test, y_preds))


pickle.dump(DecisionTreeClassifier, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[4, 300, 500]]))


