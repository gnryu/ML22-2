# -*- coding: utf-8 -*-
"""PHW2-graphs3-dbscan-em.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12yUzIZCvy1VM-oIy4H_9fHlY9LiF42J7
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold

from itertools import product

data = pd.read_csv('./housing.csv', sep=',')
print(data)

# replace the value of ? to NaN
data.replace("?", np.NaN, inplace = True)
print(data.isna().sum())

# drop the row which has NaN value
data.dropna(inplace = True)

X = data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 9]]
print(X)

y = data.iloc[:, 8] # target feature
print(y)

from sklearn.manifold import TSNE
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def tsne(data):
    tsne_result = TSNE(n_components=2).fit_transform(data)
    return tsne_result

data_copy = data.copy()

corr_matrix = data_copy.corr()
corr_matrix_n = corr_matrix['median_house_value'].sort_values(ascending = False).nlargest(6)
corr_matrix_n

import seaborn as sns
import matplotlib.pyplot as plt

sns.clustermap(corr_matrix, annot = True, cmap = 'RdYlBu_r', vmin = -1, vmax = 1)

from sklearn import metrics

def silhouette_score(estimator, X):
    clusters = estimator.fit_predict(X)
    # if clusters.labels_ < 2:
    #     return 0
    score = metrics.silhouette_score(X, clusters)
    return score

def combine_data_features(data, feature_combination):
  
  data = data[feature_combination]
  return data

from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
import category_encoders

pd.set_option('mode.chained_assignment',  None) # <==== 경고를 끈다

def encode_data(data, house_price, encoder_type):
  if encoder_type == 'LabelEncoder':
    encoder = LabelEncoder()
    encoder.fit(data['ocean_proximity'])
    data['ocean_proximity'] = encoder.transform(data['ocean_proximity'])
  
  elif encoder_type == 'OneHotEncoder':
    data = pd.get_dummies(data, columns = ['ocean_proximity'], drop_first = True)

  elif encoder_type == 'TargetEncoder':
    encoder = TargetEncoder()
    data['ocean_proximity_encoded'] = encoder.fit_transform(data['ocean_proximity'], house_price)
    data.drop('ocean_proximity', axis = 1, inplace = True)
  
  return data

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def scale_data(data, scaler_type):
  if scaler_type == 'StandardScaler':
    scaler = StandardScaler()
  
  elif scaler_type == 'MinMaxScaler':
    scaler = MinMaxScaler()

  elif scaler_type == 'RobustScaler':
    scaler = RobustScaler()

  scaler.fit(data)
  data = scaler.transform(data)

  return data

# label_standard_data = scale_data(encode_data(X, y, 'LabelEncoder'), 'StandardScaler')
# label_standard_tsne_data = tsne(label_standard_data)
# label_minmax_data = scale_data(encode_data(X, y, 'LabelEncoder'), 'MinMaxScaler')
# label_minmax_tsne_data = tsne(label_standard_data)
# label_robust_data = scale_data(encode_data(X, y, 'LabelEncoder'), 'RobustScaler')
# label_robust_tsne_data = tsne(label_standard_data)
# onehot_standard_data = scale_data(encode_data(X, y, 'OneHotEncoder'), 'StandardScaler')
# onehot_standard_tsne_data = tsne(label_standard_data)
# onehot_minmax_data = scale_data(encode_data(X, y, 'OneHotEncoder'), 'MinMaxScaler')
# onehot_minmax_tsne_data = tsne(label_standard_data)
# onehot_robust_data = scale_data(encode_data(X, y, 'OneHotEncoder'), 'RobustScaler')
# onehot_robust_tsne_data = tsne(label_standard_data)
# target_standard_data = scale_data(encode_data(X, y, 'TargetEncoder'), 'StandardScaler')
# target_standard_tsne_data = tsne(label_standard_data)
# target_minmax_data = scale_data(encode_data(X, y, 'TargetEncoder'), 'MinMaxScaler')
# target_minmax_tsne_data = tsne(label_standard_data)
# target_robust_data = scale_data(encode_data(X, y, 'TargetEncoder'), 'RobustScaler')
# target_robust_tsne_data = tsne(label_standard_data)

# def encode_scale_tsne_data(encoder, scaler, tsneonoff):
#     if encoder == 'LabelEncoder' and scaler == 'StandardScaler':
#         if tsneonoff:
#             return label_standard_tsne_data
#         else: 
#             return label_standard_data
#     elif encoder == 'LabelEncoder' and scaler == 'MinMaxScaler':
#         if tsneonoff:
#             return label_minmax_tsne_data
#         else: 
#             return label_minmax_data
#     elif encoder == 'LabelEncoder' and scaler == 'RobustScaler':
#         if tsneonoff:
#             return label_robust_tsne_data
#         else: 
#             return label_robust_data
#     elif encoder == 'OneHotEncoder' and scaler == 'StandardScaler':
#         if tsneonoff:
#             return onehot_standard_tsne_data
#         else: 
#             return onehot_standard_data
#     elif encoder == 'OneHotEncoder' and scaler == 'MinMaxScaler':
#         if tsneonoff:
#             return onehot_minmax_tsne_data
#         else: 
#             return onehot_minmax_data
#     elif encoder == 'OneHotEncoder' and scaler == 'RobustScaler':
#         if tsneonoff:
#             return onehot_robust_tsne_data
#         else: 
#             return onehot_robust_data
#     elif encoder == 'TargetEncoder' and scaler == 'StandardScaler':
#         if tsneonoff:
#             return target_standard_tsne_data
#         else: 
#             return target_standard_data
#     elif encoder == 'TargetEncoder' and scaler == 'MinMaxScaler':
#         if tsneonoff:
#             return target_minmax_tsne_data
#         else: 
#             return target_minmax_data
#     elif encoder == 'TargetEncoder' and scaler == 'RobustScaler':
#         if tsneonoff:
#             return target_robust_tsne_data
#         else: 
#             return target_robust_data

#####################################################################################3
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from pyclustering.cluster.clarans import clarans
from pyclustering.utils import timedcall

def cluster_data(X, model, hyper_parameter_dict, n_cluster, quality_measure):
  if model == 'KMeans':
    clustering_algorithm = KMeans()
    gs = GridSearchCV(clustering_algorithm, param_grid = hyper_parameter_dict, scoring=quality_measure, refit=True).fit(X)

    m = KMeans(n_clusters = n_cluster, random_state = gs.best_params_['random_state']).fit(X)
    print("Best params : ", gs.best_params_)
    print("inertia: ", m.inertia_)


  elif model == "DBSCAN":
    clustering_algorithm = DBSCAN()
    gs = GridSearchCV(clustering_algorithm, param_grid = hyper_parameter_dict, scoring=quality_measure, refit=True).fit(X)
    print("Best params : ", gs.best_params_)
    m = DBSCAN(eps = gs.best_params_['eps'], min_samples = gs.best_params_['min_samples']).fit(X)
    
  elif model == 'EM':
    clustering_algorithm = GaussianMixture()
    gs = GridSearchCV(clustering_algorithm, param_grid = hyper_parameter_dict, scoring=quality_measure, refit=True).fit(X)
    print("Best params : ", gs.best_params_)
    m = GaussianMixture(n_components= n_cluster, random_state=gs.best_params_['random_state']).fit(X)

    gmm_labels = m.fit_predict(X)
    

  elif model == 'CLARANS':
    clarans_instance = clarans(X, n_cluster, 3, 5)
    (ticks, result) = timedcall(clarans_instance.process)
    m = clarans_instance
  return m

import matplotlib.pyplot as plt
import math
def visualize(model, model_name, scaled_data, tsne_data, house_price, feature_combination):
    label = []
    if model_name == "DBSCAN": 
        label = model.labels_
    elif model_name == 'CLARANS':
        clusters = model.get_clusters()
        print(clusters)
        label = [0 for i in range(len(scaled_data))]
        c_num = 0
        for cluster in clusters:
            for l in cluster:
                label[l] = c_num
            c_num += 1
        # medoids = model.get_medoids()
        # ### visualize
        # vis = cluster_visualizer_multidim()
        # vis.append_clusters(clusters, scaled_data, marker = "*", markersize = 5)
        # vis.show(max_row_size = 3)
    else:
        label = model.predict(scaled_data)
    #Getting unique labels
    u_labels, l_counts = np.unique(label, return_counts=True)
    print("Count labels: ", len(u_labels))
    if len(u_labels) > 1 :
        score = metrics.silhouette_score(scaled_data, label)
        print("Silhouette score: ", score)
    index_num = 1

    num = 0
    for i in range(len(feature_combination)):
        num += i
    if num < 1:
        num = 1
    # #plotting the results:
    # fig = plt.figure(figsize=(36, math.ceil(num/6)*6))
    # for a in range(len(feature_combination)):
    #     for b in range(a+1, len(feature_combination)):
    #         ax = fig.add_subplot(math.ceil(num/6), 6, index_num)
    #         index_num += 1
    #         for i in u_labels:
    #             ax.scatter(scaled_data[label == i , a] , scaled_data[label == i , b] , label = i, s=3)
    #         ax.set_xlabel(feature_combination[a])
    #         ax.set_ylabel(feature_combination[b])
    # plt.show()

    # print("\nCompare with house price")
    # index_num = 1
    # fig = plt.figure(figsize=(36, math.ceil(num/6)*6))
    # for a in range(len(feature_combination)):
    #     ax = fig.add_subplot(math.ceil(num/6), 6, index_num)
    #     index_num += 1
    #     for i in u_labels:
    #         ax.scatter(house_price[label == i] , scaled_data[label == i , a] , label = i, s=3)
    #     ax.set_xlabel('house price')
    #     ax.set_ylabel(feature_combination[a])
    # plt.show()

    fig = plt.figure(figsize=(18, 6))
    fig.tight_layout()
    ax = fig.add_subplot(131, projection='3d')
    for i in u_labels:
        ax.scatter(tsne_data[label == i , 0] , tsne_data[label == i , 1], house_price[label == i] , label = i, s=3)
    ax.set_title('Visualize by t-SNE')
    ax.set_xlabel('tsne1')
    ax.set_ylabel('tsne2')
    
    ax = fig.add_subplot(132)
    for i in u_labels:
        ax.scatter(tsne_data[label == i , 0] , tsne_data[label == i , 1], label = i, s=3)
    ax.set_title('Visualize by t-SNE')
    ax.set_xlabel('tsne1')
    ax.set_ylabel('tsne2')

    ax = fig.add_subplot(133)
    ax.bar(u_labels, l_counts)
    ax.set_xlabel('label')
    ax.set_ylabel('count')
    plt.show()

def preprocess(dataset, encoder_list, scaler_list, feature_combinations):
    preprocessed_data = [[[dataset for _ in range(len(encoder_list))] for _ in range(len(scaler_list))] for _ in range(len(feature_combinations))]
    preprocessed_tsne_data = [[[dataset for _ in range(len(encoder_list))] for _ in range(len(scaler_list))] for _ in range(len(feature_combinations))]
    feature_data = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 9]]
    house_price = dataset.iloc[:, 8] # target feature
    for i in range(len(encoder_list)):
        for j in range(len(scaler_list)):
            for k in range(len(feature_combinations)):
                combined_data = combine_data_features(feature_data, feature_combinations[k])

                if 'ocean_proximity' in feature_combinations[k]:
                    encoded_data = encode_data(combined_data, house_price, encoder_list[i])
                else:
                    encoded_data = combined_data

                scaled_data = scale_data(encoded_data, scaler_list[j])
                preprocessed_data[i][j][k] = scaled_data
                preprocessed_tsne_data[i][j][k] = tsne(scaled_data)
    return preprocessed_data, preprocessed_tsne_data

from yellowbrick.cluster import KElbowVisualizer
def AutoML(dataset, model_list, encoder_list, scaler_list, n_clusters, feature_combinations, hyper_parameter_dict, quality_measure):

  items = [model_list, encoder_list, scaler_list, n_clusters, feature_combinations]
  combinations = list(product(*items))

  preprocessed_data, preprocessed_tsne_data = preprocess(dataset, encoder_list, scaler_list, feature_combinations)

  for model, encoder, scaler, n_cluster, feature_combination in combinations:
    if model == 'DBSCAN' and n_cluster > 2:
      continue
    print("\n\n\n===================================================================")
    print("Model: ", model)
    print("Encoder: ", encoder)
    print("Scaler: ", scaler)
    print("Feature list: ", feature_combination)
    if model != 'DBSCAN': 
      print("n_cluster: ", n_cluster)

    # test with sample
    if model == 'CLARANS':
      sample_dataset = dataset.sample(n=500)
      feature_data = sample_dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 9]]
      house_price = sample_dataset.iloc[:, 8] # target feature
      combined_data = combine_data_features(feature_data, feature_combination)
      if 'ocean_proximity' in feature_combination:
        encoded_data = encode_data(combined_data, house_price, encoder)
      else:
        encoded_data = combined_data
      scaled_data = scale_data(encoded_data, scaler)
    else:
      feature_data = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 9]]
      house_price = dataset.iloc[:, 8] # target feature
      scaled_data = preprocessed_data[encoder_list.index(encoder)][scaler_list.index(scaler)][feature_combinations.index(feature_combination)]
    
    m = cluster_data(scaled_data, model, hyper_parameter_dict[model], n_cluster, quality_measure)

    if model == 'CLARANS':
      tsne_data = tsne(scaled_data)
    else:
      tsne_data = preprocessed_tsne_data[encoder_list.index(encoder)][scaler_list.index(scaler)][feature_combinations.index(feature_combination)]

    visualize(m, model, scaled_data, tsne_data, house_price, feature_combination)

    # if model != 'CLARANS' and n_cluster == 2 :
    #   visualizer = KElbowVisualizer(m, k=(1,10))
    #   visualizer.fit(scaled_data)

model_list = ['DBSCAN', 'EM'] 
encoder_list = ['LabelEncoder', 'OneHotEncoder', 'TargetEncoder'] 
scaler_list = ['StandardScaler', 'MinMaxScaler', 'RobustScaler'] 
n_clusters = [2, 4, 6, 8, 10] 

feature_combinations = [ ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity'], 
['median_income', 'total_rooms', 'housing_median_age', 'households', 'total_bedrooms'],
['median_income', 'total_rooms', 'housing_median_age']] 
hyper_parameter_dict = {
    'KMeans' : {'random_state' : [1, 2]},
    'DBSCAN' : {'eps':[0.5, 0.7, 0.9, 1.1], 'min_samples':[4, 5, 6]},
    'EM':{'n_components':[2, 4, 6, 8, 10],'random_state':[1,2]},
    'CLARANS' : {'numlocal' : [6, 8, 10], 'maxneighbor' : [4, 6]}
}

AutoML(data, model_list, encoder_list, scaler_list, n_clusters, feature_combinations, hyper_parameter_dict, silhouette_score)