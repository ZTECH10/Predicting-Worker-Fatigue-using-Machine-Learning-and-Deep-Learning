# Predicting-Worker-Fatigue-using-Machine-Learning-and-Deep-Learning


---------------
Main code 
---------------

Please check out "Part1.ipynb" nad "Part2.ipynb" Jupyter notebook files.

---------------
Abstract
---------------
Machine learning is becoming more and more prominent in our world today, influencing every global industry.
Our projects focus is on the safety industry as we created a machine learning model that can predict whether
an individual can carry out certain tasks. Our machine learning model paired with the Muse 2 device will provide 
a cost-effective solution for when we need to determine whether an individual is feeling fatigued. Our data collection
process followed up by an in-depth data preprocessing procedure that consisted of 4 different approaches made it easier
for us to tune our machine learning model. In the end, we were able to reach an accuracy score of 99.84% when differentiating
between fatigue vs normal. By improving on our current work, we can expand the applications of our machine learning model to
countless other industries. With that being said, this project should be used as a steppingstone to create a reliable solution 
that cannot just improve efficiency within companies but prevent serious injuries to people and their environments.



---------------
Project Goal
---------------
-Create a machine learning model to predict the state of the brain between "Normal" and "Fatigue"


---------------
Use cases
---------------
-Use cases in high risk/dangerous jobs such as construction, pilots etc.


------------------------------
Data pre-preporcessing
------------------------------
-We tried to preprocess the data using mutiple methods to understand what gave us the best results.

-Generated 4 different datasets:

    data_preprocessing sub-folder: Here, we applied Gaussian based outlier detection and imputation of missing values on the original dataset.

    data_preprocessing_with_PCA  sub-folder: Here, we added 5 PCAs to each dataset in the data_preprocessing sub-folder.

    data_preprocessing_with_FastICA  sub-folder: Here, we added 5  FastICAs to each dataset in the data_preprocessing sub-folder.

    data_preprocessing_with_FastICA_Temporal_Freq  sub-folder: Here, we  added 120 temporal-based features and 100 frequency-based features to 
    each dataset in the data_preprocessing_with_FastICA  sub-folder.  The total number of features for each dataset in this sub-folder is now 245 features.

-------------------
AutoML Library
-------------------
-Used a library to help us define the best model to use and then created our model around that to save time and improve accuracy.

---------------
Results
---------------
-Used the AutoML library to try different ML models on the 4 preprocessed datasets to determine which ML algorithm gives the highest accuracy.
ML models considered: Baseline, Linear , Decision Tree, Random Forest, XGBoost, Neural Network.

-The highest accuracy was achieved on the data_preprocessing_with_PCA subfolder (99.1% with Neural Network).

---------------
Challenges
---------------
-Differentiating between fatigue and normal data.

-Misalignment of the Muse device (headband) leading to null values.
