# OSDA Big Homework
## Lazy FCA

### Datasets:
1. [Mushroom Classification](https://www.kaggle.com/datasets/uciml/mushroom-classification?resource=download)

### Datasets preparation
1. The `class` which is considered as the target was mapped to True and False where True indicates as Poisenous and False otherwise.
2. For all other features we created one-hot mapping where all the classes are mapped into binary data. For example if for class cap-shape we have multiple values like (bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s) we create 5 new features as (cap-shape_b, cap-shape_c, cap-shape_x, cap-shape_f, cap-shape_k, cap-shape_s) which each all the values are as booleans.
3. In total we get 117 new features and if we use all these features, we will have overfitting in our model. So we drop the features which have correlation larger than `0.15`. This threshold gives us a total of 58 features.
4. Duplicates and NaNs are dropped from the training data
5. The dataset in also downslaced to 300 features

Dataset sizes after preparation:
1. 300 rows x 59 columns

## Standard Models

In this assignment, we utilized the following standard models:
- **Logistic Regression**
- **K-Nearest Neighbors**
- **Naive Bayes**:
  - Multinomial NB
  - Gaussian NB
  - Complement NB
- **Decision Tree**
- **Random Forest**

---

## Validation

Stratified 10-fold cross-validation was used to calculate **accuracy** and **F1-score** for each dataset and to tune model parameters.

---

## Hyperparameter Tuning

Hyperparameter tuning was performed using **GridSearchCV** from **scikit-learn** for standard models.  
For **BinarizedBinaryClassifier** and **PatternBinaryClassifier** (FCALC classifiers), optimal parameters were determined using simple **for-loops** since they do not support GridSearchCV.

### Tuned Hyperparameters:

- **Logistic Regression**:  
  - `C`

- **K-Nearest Neighbors**:  
  - `n_neighbors`

- **MultinomialNB**:  
  - `alpha`

- **GaussianNB**:  
  - `var_smoothing`

- **ComplementNB**:  
  - `alpha`

- **Decision Tree**:  
  - `criterion`
  - `max_depth`
  - `min_samples_split`

- **Random Forest**:  
  - `criterion`
  - `max_depth`
  - `min_samples_split`
  - `n_estimators`

- **BinarizedBinaryClassifier**:  
  - `method`
  - `alpha`

- **PatternBinaryClassifier**:  
  - `method`
  - `alpha`

### Special Case: Binary Classifiers  
For binary classifiers:
- `alpha` was searched in the interval `[0, 1]` (including edges) for all methods.
- Additionally, for the **ratio-support method**, `alpha` was searched in the interval `[1, 10]` (including edges).


### Optimal parameters

Optimal parameters for each method and dataset are presented in a table below.

<table>
    <colgroup>
        <col style="border: 1px solid #ddd" span="13" />
    </colgroup>
    <tr>
        <th style="text-align: center" rowspan="1"><u>Dataset</u> Method</th>
        <th style="text-align: center" colspan="3">Mushrooms</th>
    </tr>
    <tr style ="border-bottom: 1px solid #ddd">
    </tr>
    <tr>
        <th rowspan="1">Logistic regression</th>
        <td>C</td>
        <td>4.32</td>
    </tr>
    <tr>
        <th rowspan="1">Logistic regression</th>
        <td>n_neighbors</td>
        <td>5</td>
    </tr>
    <tr>
        <th rowspan="1">Multinomial Naive Bayes</th>
        <td>alpha</td>
        <td>0.301</td>
    </tr>
    <tr style ="border-bottom: 1px solid #ddd">
        <th rowspan="1">Gaussian Naive Bayes</th>
        <td>var_smoothing</td>
        <td>0.5336</td>
    </tr>
    <tr>
        <th rowspan="1">Complement Naive Bayes</th>
        <td>alpha</td>
        <td>0.201</td>
    </tr>
    <tr>
        <th rowspan="2">Decision Tree</th>
        <td>min_samples_split</td>
        <td>max_depth</td>
        <td>criterion</td>
        <td> </td>
    </tr>
    <tr style ="border-bottom: 1px solid #ddd">
        <td>8</td>
        <td>8</td>
        <td>entropy</td>
    </tr>
    <tr>
        <th rowspan="2">Random Forest</th>
        <td>min_samples_split</td>
        <td>max_depth</td>
        <td>criterion</td>
        <td>n_estimators</td>
    </tr>
    <tr style ="border-bottom: 1px solid #ddd">
        <td>6</td>        
        <td>16</td>        
        <td>gini</td>        
        <td>100</td>
    </tr>
    <tr style ="border-top: 1px solid #ddd">
        <th rowspan="2">Binarized Binary Classifier</th>
        <td>method</td>
        <td>alpha</td>
    </tr>
    <tr style ="border-bottom: 1px solid #ddd">
        <td>standard-support</td>
        <td>1</td>
    </tr>
    <tr>
        <th rowspan="2">Pattern Binary Classifier</th>
        <td>method</td>
        <td>alpha</td>
    </tr>
    <tr style ="border-bottom: 1px solid #ddd">
        <td>standard</td>
        <td>0.0</td>
    </tr>

</table>

### Results
Performance of models is presented in the table below. 

<table>
    <colgroup>
        <col style="border: 1px solid #ddd" span="10" />
    </colgroup>
    <tr style ="border-bottom: 1px solid #ddd; margin-left: 0">
        <th style="text-align: center">Dataset</th>
        <th style="text-align: center" colspan="3">Mushroom</th>
    </tr>
    <tr style ="border-bottom: 1px solid #ddd">
        <th style="text-align: center">Method\ Metric</th>
        <td style="text-align: center">Accuracy</td>
        <td style="text-align: center">F1 binary</td>
        <td style="text-align: center">F1 macro</td>
    </tr>
    <tr>
        <th>Logistic regression</th>
        <td>0.8167</td>
        <td>0.8293</td>
        <td>0.8149</td>
    </tr>
    <tr>
        <th>K-NN</th>
        <td>0.9000</td>
        <td>0.9105</td>
        <td>0.8983</td>
    </tr>
    <tr>
        <th>Multinomial Naive Bayes</th>
        <td>0.7600</td>
        <td>0.7828</td>
        <td>0.7562</td>
    </tr>
    <tr>
        <th>Gaussian Naive Bayes</th>
        <td>0.6633</td>
        <td>0.7579</td>
        <td>0.5977</td>
    </tr>
    <tr>
        <th>Complement Naive Bayes</th>
        <td>0.9667</td>
        <td>0.9691</td>
        <td>0.9664</td>
    </tr>
    <tr>
        <th>Decision Tree</th>
        <td>0.3493</td>
        <td>0.2666</td>
        <td>0.3343</td>
    </tr>
    <tr>
        <th>Random Forest</th>
        <td>0.9600</td>
        <td>0.9636</td>
        <td>0.9596</td>
    </tr>
    <tr style ="border-top: 1px solid #ddd">
        <th>Binarized Binary Classifier</th>
        <td>0.4372</td>
        <td>0.4949</td>
        <td>0.1667</td>
    </tr>
    <tr>
        <th>Pattern Binary Classifier</th>
        <td>0.9367</td>
        <td>0.8363</td>
        <td>0.9009</td>
    </tr>

</table>

Note that for lazy classifiers it was impossible to use **f1_score** from **scikit** with **average = "binary"**, as the output consisted of three classes (**1** for positive class, **0** for negative class and **-1** for undefined). Therefore **average = "macro"** for **f1_score** from **scikit** was used.

Despite that, it is still possible to compute binary F1 if we interpret undefined class as missclassification; this is how the values presented in the table were obtained.

Macro F1 was also calculated for all standard models so as to enable comparison with lazy classifiers.

### Binarized Binary Classifier	and Pattern Binary Classifier performance
Binarized Binary Classifier: Poor performance with low accuracy (43.72%) and very weak F1-macro score (16.67%), indicating significant limitations in both prediction and balance.

Pattern Binary Classifier: Strong performance with high accuracy (93.67%) and F1-macro score (90.09%), making it a reliable choice compared to its counterpart.