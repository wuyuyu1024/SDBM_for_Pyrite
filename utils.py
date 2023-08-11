import pandas as pd 
import numpy as np
# Makes sure we see all columns
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns



class DataLoader():
    def __init__(self):
        self.data = None
        self.features = ['Mn', 'Co', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Mo', 'Sn', 'W', 'Ca', 'Cd',
                        'Ag', 'Sb', 'Te' ,'Au', 'Bi', 'Pb', 'Mo', 'Ti', 'Tl', 'V', 'Cr', 'Hg', 
        ]

    def load_dataset(self, path="pyrite_data/py_dataset_main.xlsx", fill_lmt=True, groups='location'):
        summary = pd.read_excel(path)
        summary_path = summary[summary.infocheck == 1]
        dfs = []
        for path, type, subclass, citation, group in zip(summary_path['path'], summary_path['class'], summary_path['subclass'],summary_path['citation'], summary_path[groups]):
            # print(path)
            temdf = pd.read_excel('pyrite_data\\'+ path).dropna(how='all').dropna(axis=1, how='all')
            # print(temdf.shape)
            temcol = list(temdf.columns)
            for i, col in enumerate(temcol):
                col = col.strip()
                temcol[i] = col
            subset = set(temdf.columns) & set(self.features)
            

            temdf['group'] = group
            temdf['type'] = type
            temdf['reference'] = citation
            temdf['subclass'] = subclass
#                 temdf.drop(group, axis=1, inplace=True)
            if groups=='location' and 'Location/strata age' in temdf.columns:
                # print('path:', path)
                temdf['group'] = temdf['Location/strata age']
                temdf['type'] = temdf['Type']
                #### direct source
                # temdf['reference'] = temdf['Reference']

                # print(temdf[['group', 'type', 'reference']])
#                 print('@')
#             else:
#                 temdf['group'] = author
            
            for element in self.features:
                try:
                    temdf[element] = pd.to_numeric(temdf[element].astype('float', errors='ignore'), errors="coerce")
                except:
                    # print(f'{element} not exist')
                    pass

            # temdf.replace({'0':np.nan, 0:np.nan}, inplace=True)

            

            oldl = len(temdf)
            temdf.dropna(how='all', inplace=True, subset=subset)
#             print(f'{len(temdf)-oldl} empty rows dropes')
            if fill_lmt:
                min_value = temdf.min()
                min_value[subset] *= 0.5
                temdf.fillna(value=min_value, inplace=True)
            # print(temdf.head())
            dfs.append(temdf)

        self.data = pd.concat(dfs, ignore_index=True)#.drop('SAMPLENAME', axis=1)

    def isdepo(self, row):
        if row['y'] in {'Hydrothermal', 'Orogenic gold', 'Magmatic-hydrothermal'}:
            # print('@')
            row['isdepo'] = 1
        return row

    def preprocess_data(self, threshhold=900):
        cols_select = [] #'Metamophic_grade', 'summary'
        for col in self.data.columns:
            if self.data[col].notna().sum() > threshhold:
                print(col, ':', self.data[col].notna().sum())
                cols_select.append(col)
        self.data = self.data[cols_select]

        # self.data['isdepo'] = 0
        # self.data = self.data.apply(self.isdepo, axis=1)
        
#         print('citations: ', self.data.citation.value_counts())

                # Standardization 
                # Usually we would standardize here and convert it back later
        # But for simplification we will not standardize / normalize the features


    def get_data_split(self, log=False, target=None):
        X = self.data#.iloc[:,:-1]
        # y = self.data.iloc[:,-1]
        y_ = self.data.pop('y')
        group = self.data.pop('group')
        isdepo = self.data.pop('isdepo')
        if not target:
            y = y_
        elif target == 'isdepo':
            y = isdepo
        if log:
            X = np.log(X)
            
        return train_test_split(X, y, test_size=0.20, random_state=2021, shuffle=True, stratify=None)
    
    def oversample(self, X_train, y_train, sampling_strategy='auto', shrinkage=None):
        oversample = RandomOverSampler(sampling_strategy=sampling_strategy, shrinkage=shrinkage)
        # Convert to numpy and oversample
        x_np = X_train.to_numpy()
        y_np = y_train.to_numpy()
        x_np, y_np = oversample.fit_resample(x_np, y_np)
        # Convert back to pandas
        x_over = pd.DataFrame(x_np, columns=X_train.columns)
        y_over = pd.Series(y_np, name=y_train.name)
        return x_over, y_over

    