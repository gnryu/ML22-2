<h1> Programming Homework 2 </h1>

Use following 4 <strong>clustering algorithms</strong>.

- K-means
- EM (GMM)
- CLARANS
- DBSCAN

<br>

<h2> Note </h2>
There are so many cases: combination of scaling methods, encoding methods, combination of features, and number of clusters for each of the four models <br>
Since it takes a long time to run on one model, we created a file separately. DBSCAN and EM are implemented in one file, and Clarans is divided into two files to visualize. <br>
(Clarans takes too long to learn, so we only use 500 samples of total dataset for Clarans) <br><br>

So a total of four files were created as follows.
- K-means
- DBSCAN_EM
- Clarans_1
- Clarans_2

<strong>
For K-means, DBSCAN, and EM, we also upload .py file because the capacity of .ipynb file is too large to be uploaded. Please refer to the attached Google Colaboratory link if you want to see the results.

- K-means: https://drive.google.com/file/d/1tu4xIGmTH_WeTAo6XGiNa38Ac9ICWtYK/view?usp=sharing
- DBSCAN_EM: https://drive.google.com/file/d/12yUzIZCvy1VM-oIy4H_9fHlY9LiF42J7/view?usp=sharing
</strong>

We also upload the full code of .py file containing all four models under the name of FullCode.py

<br>

<h2> Data </h2>
The California Housing Prices Dataset <br>
link: https://www.kaggle.com/datasets/camnugent/california-housing-prices

<br><br>

<h2> Function </h2>

<h3> 1. AutoML </h3>

```python
def AutoML (dataset, model_list, encoder_list, scaler_list, n_clusters, feature_combinations, hyper_parameter_dict, quality_measure)
```

<strong>description</strong>
  - Combine the number of all possible cases and turn them into “for” statement. Find the best hyperparameters for each case, and visualize the clustered model using the best hyperparameters.
  
<strong>parameters</strong>
  - <strong>dataset</strong>: the entire data
  - <strong>model_list</strong>: list of models<br>
    - KMeans
    - DBSCAN
    - EM
    - CLARANS
  - <strong>encoder_list</strong>: list of 3 encoding methods<br>
    - LabelEncoder
    - OneHotEncoder
    - TargetEncoder
  - <strong>scaler_list</strong>: list of 3 scaling methods<br>
    - StandardScaler
    - MinMaxScaler
    - RobustScaler
  - <strong>n_clusters</strong>: list of numbers of clusters<br>
    - 2
    - 4
    - 6
    - 8
    - 10
  - <strong>feature_combinations</strong>: various feature combinations<br>
    - [longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity]
    - [median_income, total_rooms, housing_median_age, households, total_bedrooms]
    - [median_income, total_rooms, housing_median_age]
  - <strong>hyper_parameter_dict</strong>: dictionary of hyperparameter for each model
  - <strong>quality_measure</strong>: quality measure tool
    - silhouette score

<strong>output</strong>
  - used model, encoder, scaler, feature combination, number of clusters, best hyperparameter
  - score of quality measure
  - visualize result

<br>

<h3> 2. encode_data </h3>

```python
def encode_data (data, house_price, encoder_type)
```

<strong>description</strong>
  - Encode categorical data.
  
<strong>parameters</strong>
  - <strong>data</strong>: data containing variables to encode
  - <strong>house_price</strong>: target variable // used in TargetEncoder
  - <strong>encoder_type</strong>: type of encoder<br>
    - LabelEncoder
    - OneHotEncoder
    - TargetEncoder
  
<strong>return</strong>
  - encoded data 

<br>

<h3> 3. scale_data </h3>

```python
def scale_data (data, scaler_type)
```

<strong>description</strong>
  - Scale the data.
  
<strong>parameters</strong>
  - <strong>data</strong>: data to scale
  - <strong>scaler_type</strong>: type of scaler<br>
    - StandardScaler
    - MinMaxScaler
    - RobustScaler
  
<strong>output</strong>
  - scaled data

<br>

<h3> 4. combine_data_features </h3>

```python
def combine_data_features (data, feature_combination)
```

<strong>description</strong>
  - Use the function for various combinations of features.
  
<strong>parameters</strong>
  - <strong>data</strong>: dataset with all kinds of features
  - <strong>feature_combination</strong>: various feature combinations
  
<strong>return</strong>
  - dataset after feature combination (all kinds of features or fewer features)

<br>

<h3> 5. silhouette_score </h3>

```python
def silhouette_score (estimator, X)
```

<strong>description</strong>
  - Measure the silhouette score.
  
<strong>parameters</strong>
  - <strong>estimator</strong>: model object for training
  - <strong>X</strong>: dataset for training
  
<strong>return</strong>
  - silhouette score 

<br>

<h3> 6. cluster_data </h3>

```python
def cluster_data (X, model, hyper_parameter_dict, n_cluster, quality_measure)
```

<strong>description</strong>
  - Find the best hyperparameters combination through GridSearch based on a given quality measure. And return the model to which it is applied.
  
<strong>parameters</strong>
  - <strong>X</strong>: data after encoding and scaling
  - <strong>model</strong>: type of model<br>
    - KMeans
    - DBSCAN
    - EM
    - CLARANS
  - <strong>hyper_parameter_dict</strong>: dictionary of hyperparameter for each model
  - <strong>n_cluster</strong>: number of clusters
  - <strong>quality_measure</strong>: quality measure tool
    - silhouette score
  
<strong>return</strong>
  - cluster model

<br>

<h3> 7. tsne </h3>

```python
def tsne (data)
```

<strong>description</strong>
  - Use TSNE to reduce dimensions and visualize them better.
  
<strong>parameters</strong>
  - <strong>data</strong>: data to reduce dimension
  
<strong>return</strong>
  - result of TSNE 

<br>

<h3> 8. preprocess </h3>

```python
def preprocess (dataset, encoder_list, scaler_list, feature_combinations)
```

<strong>description</strong>
  - Preprocess the data and return it.<br>
There are two datasets that are returned, one that is simply preprocessed and one that uses t-SNE.<br>
The combine_data, encode_data,, scale_data, and TSNE functions are used internally.
  
<strong>parameters</strong>
  - <strong>dataset</strong>: The dataset to be preprocessed. Basically, Kaggle's California Housing Prices are the default.
  - <strong>encoder_list</strong>: Passes the type of encoder to be used in the form of an array
    - LabelEncoder
    - OneHotEncoder
    - TargetEncoder
  - <strong>scaler_list</strong>: Passes the type of scaler to be used in the form of an array
    - StandardScaler
    - MinMaxScaler
    - RobustScaler
  - <strong>feature_combinations</strong>: Pass a list of combinations of features to combine
  
<strong>return</strong>
  - <strong>preprocessed_data</strong>: It returns preprocessed data in the form of a 3D array.<br>The first index is the encoder type, the second index is the scaler type, and the third index is the feature_combination type.
  - <strong>preprocessed_tsne_data</strong>: It returns preprocessed t-SNE data in the form of a 3D array.<br>The first index is the encoder type, the second index is the scaler type, and the third index is the feature_combination type.

<br>

<h3> 9. visualize </h3>

```python
def visualize (model, model_name, scaled_data, tsne_data, house_price, feature_combination)
```

<strong>description</strong>
  - Silhouette score output and visualization.<br>
It uses silhouette_score and matplotlib internally.
  
<strong>parameters</strong>
  - <strong>model</strong>: The dataset to be preprocessed. Basically, Kaggle's California Housing Prices are the default.
  - <strong>model_name</strong>: The model name used for training.<br>Each model has a slightly different visualization process, so we specify this.
  - <strong>scaled_data</strong>: preprocessed data
  - <strong>tsne_data</strong>: preprocessed with t-SNE data
  - <strong>house_price</strong>: deliver feature data for house values to be used as target data
  - <strong>feature_combination</strong>: list of feature_combinations used
  
<strong>output</strong>
  - visualized plots

<br>

<h2> Team Member Contribution </h2>

201835421 김범기: 25%, discussion, write the code, anaylsis
202035531 유지나: 25%, discussion, write the code, anaylsis
202037071 정지우: 25%, discussion, write the code, anaylsis
202299092 백우진: 25%, discussion, write the code, anaylsis
