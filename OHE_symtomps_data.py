# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# %%
anonymous_data = pd.read_csv('symtomps_disease.csv')
anonymous_data.head()

# %%
anonymous_data.tail()

# %%
anonymous_data.dtypes

# %%
anonymous_data['symtomps'].unique()

# %%
anonymous_data['disease'].unique()

# %%
OHE =OneHotEncoder()
print(OHE)

# %%
OHE.fit_transform(anonymous_data[['symtomps', 'disease']]).toarray()

# %%
f_array = OHE.fit_transform(anonymous_data[['symtomps', 'disease']]).toarray()
OHE.categories_

# %%

f_labels = OHE.categories_

# %%
np.array(f_labels).ravel()

# %%
f_labels = np.array(f_labels).ravel()
print(f_labels)

# %%
pd.DataFrame(f_array,columns=f_labels)

# %%
features = pd.DataFrame(f_array,columns=f_labels)

print(features)

# %%
features.head()

# %%
features.tail()

# %%
pd.concat([anonymous_data, features], axis=1)

# %%
anonymous_data_new_dataset = pd.concat([anonymous_data, features], axis=1)

# %%
anonymous_data_new_dataset.head()


