# UCI Bank Marketing Data Analysis

## Project Description

The main goal of this project is to build a machine learning model that predicts if the client will subscribe a term deposit (variable y). 

The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

I used `bank-additional-full.csv` which is provided by [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing). It contains 411,888 rows and 20 feature variables, ordered by date (from May 2008 to November 2010).



## Feature Engineering

The original dataset has 10 numerical variables and 10 categorical variables. I created 2 categorical variables `age_group` and `pdays_group` by using `age` and `pdays`. By doing so, skewed distributions of both variables have monotonic relationship. 

When building the final machine learning model, I had to remove `duration` variable because the attribute information says that this input should only be included for benchmark purposes. I included `duration` when analyzing dataset. 

Therefore, the final dataset has 11 numerical variables and 8 categorical variables. 

Before building a model, we need to encode categorical variables as numerical values, because machine learning algorithms can deal with only numerical values, so I used integer encoding by using feature-engine library. Also, I used StandardScaler from scikit-learning to scale all numerical variables. 


## Imbalanced data

We have a classification problem. The target is a binary variable (no-yes), and it shows 89% of no and 11% of yes responses. The dataset is a typcial imbalanced data which means that it has many more instances of certain classes than of others. Since most machine learning algorithms assume balanced distributions, samples from the minority class are likely misclassified. 

![graph1](/images/target.png)

Therefore, I implemented EasyEnsemble technique which is an ensemble algorithms that were designed to work with imbalanced datasets. 

The final machine learning model showed 0.808 roc-auc score. 




## File Descriptions
The files structure is arranged as below:

    - Bank_Marketing_BalancingSampling.ipynb: shows the workflow what balancing sampling technique is suited for 
    - Bank_Marketing_EDA.ipynb: shows how to clean the data, and shows visualizations based on the data analysis
    - Bank_Marketing_FeatureEngineering.ipynb: shows how to choose categorical encoding methods and feature scaling methods
    - Bank_Marketing_Model.ipynb: shows the entire workflow including cleaning data, feature engineering, building the model, and evaludating model. 
    - readme.md
    



## Visualization
We created graphs to see the patterns or distributions in dataset. Here are some examples:

![graph1](/images/numeric1.png)

![graph2](/images/categorical1.png)


You can find all visualizations in [Bank_Marketing_EDA.ipynb](https://github.com/yejiseoung/BankMarketing/blob/main/Bank_Marketing_EDA.ipynb)




## Reference
[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014
