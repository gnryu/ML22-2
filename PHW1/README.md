<h1> Programming Homework 1 </h1>

Compare the performance of the following <strong>classification models</strong> against the same dataset.
- Decision tree (using entropy)
- Decision tree (using Gini index)
- Logistic regression
- Support vector machine

<br>

<h2> Data </h2>
The Wisconsin Cancer Dataset <br>
link: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/

<br><br>

<h2> Function </h2>

<h3> 1. main </h3>

```python
def main (X, target, model_code, scaling_TF, k, dict)
```

<strong>description</strong>
  - determine whether you use scaling and which model to use, then use k-fold cross validation and get the average score of each group
  
<strong>parameters</strong>
  - <strong>X</strong>: independent variables
  - <strong>target</strong>: dependent variable (target variable)
  - <strong>model_code</strong>: [int], which model you will use
    - 0 for decision tree using entropy
    - 1 for decision tree using Gini index
    - 2 for logistic regression
    - 3 for support vector machine
  - <strong>scaling_TF</strong>: [bool], whether you use scaling
    - True for using robust scaling
    - False for not using scaling
  - <strong>k</strong>: [int], cross validation generator or an iterable, k of k-fold cross validation
  - <strong>dict</strong>: [dictionary], hyper parameter dictionary for each model (decision tree, logistic regression, SVM)
  
<strong>output</strong>
  - <strong>n-fold(K)</strong>: k of k-fold cross validation 
  - <strong>Scaling</strong>: scaling_TF, {True, False}
  - <strong>Used model</strong>: {Decision tree (using entropy), Decision tree (using Gini index), Logistic regression, Support vector machine}
  - <strong>Params for grid search</strong>: list of hyper parameters and its various values for each model
  - <strong>best_params</strong>: best parameter values which were found using GridSearchCV for each model
  - <strong>cv_scores mean</strong>: average score of each group of k-fold cross validation
 
 <br>
 
 <h3> 2. DecisionTree </h3>

```python
def DecisionTree (criterion, X, Y, dt)
```

<strong>description</strong>
  - find the best parameters for the best accuracy in Decision Tree model


<strong>parameters</strong>
  - <strong>criterion</strong>: the function to measure the quality of a split, {"gini", "entropy"}
  - <strong>X</strong>: independent variables
  - <strong>Y</strong>: dependent variable (target variable)
  - <strong>dt</strong>: hyperparameter dictionary for decision tree model
  
<strong>return</strong>
  - decision tree model with the best parameters for the best accuracy
  
<br>

<h3> 3. Logistic </h3>

```python
def Logistic (X, Y, dt)
```

<strong>description</strong>
  - find the best parameters for the best accuracy in Logistic Regression model


<strong>parameters</strong>
  - <strong>X</strong>: independent variables
  - <strong>Y</strong>: dependent variable (target variable)
  - <strong>dt</strong>: hyperparameter dictionary for logistic regression model
  
<strong>return</strong>
  - logistic regression model with the best parameters for the best accuracy
  
<br>

<h3> 4. SVM </h3>

```python
def SVM (X, Y, dt)
```

<strong>description</strong>
  - find the best parameters for the best accuracy in SVM model


<strong>parameters</strong>
  - <strong>X</strong>: independent variables
  - <strong>Y</strong>: dependent variable (target variable)
  - <strong>dt</strong>: hyperparameter dictionary for SVM model
  
<strong>return</strong>
  - SVM model with the best parameters for the best accuracy
  
<br>
  
<h2> Team Member Contribution </h2>

201835421 김범기: 25%, decision tree <br>
202035531 유지나: 25%, main, documentation <br>
202037071 정지우: 25%, support vector machine <br>
202299092 백우진: 25%, logistic regression
