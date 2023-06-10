# ML-for-Magnetic-Predictions
Code for reproducing all the results in [give ref here]

# Data

DS1-RAW.csv: 
The raw data set of ferromagnetic materials and the corresponding Curie temperatures compiled by James Nelson and Stefano Sanvito.

DS1.csv:
Cleaned version of DS1-RAW.csv. The feature vector has 85 features, each one describing a distinct element found in the data. Each compound is characterized by placing the percentage that each element occupies in the compound in the appropriate feature.

DS1-Compounds.csv:
Version of DS1.csv that only includes the compound names and Curie temperatures. Used for feature generation.

DS1-Incompatible.csv:
Version of DS1.csv that only includes features found in DS1-RAW.csv. Cannot be used with DS2.csv in machine learning models.

DS1-MASTML-Features.csv:
Version of DS1.csv where there are only 20 features. These features were generated and selected by the MAST-ML python library.

DS2-RAW.txt: 
The raw data set of ferromagnetic materials and the corresponding Curie temperatures compiled by Valentin Taufour.

DS2.csv: 
Cleaned version of DS2-RAW.csv. The feature vector has 85 features, each one describing a distinct element found in the data. Each compound is characterized by placing the percentage that each element occupies in the compound in the appropriate feature.

DS1+DS2.csv: 
Combination of DS1.csv and DS2.csv. Any overlapping magnetic compounds were only included once.

# Code

Clean-DS1-RAW.py:
Cleans DS1-RAW.csv and saves cleaned data to Data/DS1.csv

Clean-DS2-RAW.py:
Cleans DS2-RAW.csv and saves cleaned data to Data/DS2.csv

Combine-DS1+DS2.py:
Combines DS1.csv and DS2.csv into one dataset. Any overlapping magnetic compounds are only included once. Saves combined dataset to Data/DS1+DS2.py.

DatavsMAE-RF-Above-600K.py:
Extracts all the compounds form DS1+DS2.csv with a Tc > 600. Uses this new data to determine how the amount of training data used in a random forest model affects the mean absolute error of the predicitons. Uses 1/3 of the data as test data and samples different sizes of training data from the remaining compounds. Saves a plot of training data size vs. MAE to "Plots/MAE vs Training Data Size Above 600K.png" and the same plot with a log log scale to "Plots/MAE vs Training Data Above 600K loglog.png".

DatavsMAE-RF.py:
Uses DS1+DS2.csv to determine how the amount of training data used in a random forest model affects the mean absolute error of the predicitons. Uses 1/3 of the data as test data and samples different sizes of training data from the remaining compounds. Saves a plot of training data size vs. MAE to "Plots/MAE vs Training Data Size.png" and the same plot with a log log scale to "Plots/MAE vs Training Data loglog.png".

EvenDistributionRandomForest-DS1.py:
Creates an evenly distributed sub-dataset of DS1. Sorts data into 10 bins based on Curie temperature. Takes a random sample of 100 points from each bin and adds them all to a new set of data. 2/3 of this data is then used to train a random forest model. The model predicts on the remaining 1/3 of the data. MAE is printed in the terminal. A plot with the predicted Tc values vs. the experimental Tc values is saved to "Plots/Even Distribution Random Forest DS1.png". It also creates two error plots. One plots the experimental Tc value vs the difference between the experimental and predicted value for each point in the test set. This plot is saved to "Plots/Even Distribution Random Forest Experimental Error DS1.png". The other plots the predicted Tc value vs the difference between the experimental and predicted value for each point in the test set. This plot is saved to "Plots/Even Distribution Random Forest Predicted Error DS1.png".

Generate-MASTML-Features.ipynb:
Python notebook used to create DS1-MASTML-Features.csv. Uses MAST-ML python library to generate many features for DS1-Compounds.csv then selects the 20 most meaningful features. Saves new dataset with generated features to "Data/DS1-MASTML-Features.csv".

GenerateTernaryMaterialsCo+Fe+X.py:
Generates all possible ternary combinations that include Cobalt, Iron, and any other element found in compounds with a Tc > 600 K. The amount of each element in a generated compound changes in increments of 1%. Eliminates any duplicates in the generated data, formats it to be compatible with DS1.csv and DS2.csv and saves the new data to "Data/Generated Materials/GC_Ternary_Co+Fe+X.csv".

GenerateTernaryMaterialsFe+XX+X.py:
Generates all possible ternary combinations that include Iron, any other elements found in compounds with a Tc > 600 K, and excludes Cobalt. Holds the Iron concentration at a minimum of 80% for all generated compounds. The amount of each element in a generated compound changes in increments of 1%. Eliminates any duplicates in the generated data, formats it to be compatible with DS1.csv and DS2.csv and saves the new data to "Data/Generated Materials/GC_Ternary_Fe80+XX+X.csv".

GroupedRandomForest.py:
Creates a subset of data containing all compounds with a specified majority element found in DS1+DS2.csv. The chosen majority element must be specified for the variable MAJORITY_ELEMENT near the top of the script. 2/3 of this data is then used to train a random forest model. The model predicts on the remaining 1/3 of the data. MAE is printed in the terminal. A plot with the predicted Tc values vs. the experimental Tc values is saved to "Plots/Random Forest " + MAJORITY_ELEMENT + "-majority.png". It also creates two error plots. One plots the experimental Tc value vs the difference between the experimental and predicted value for each point in the test set. This plot is saved to "./Plots/Random Forest Experimental Error " + MAJORITY_ELEMENT + "-majority.png". The other plots the predicted Tc value vs the difference between the experimental and predicted value for each point in the test set. This plot is saved to "./Plots/Random Forest Predicted Error " + MAJORITY_ELEMENT + "-majority.png".

KfoldCV.py:
Performs 50 rounds of 3-fold cross validation on DS1.csv. Prints the mean MAE and the standard deviation in the terminal.

KNN-DS1-Random.py:
Imports DS1.csv and randomly shuffles the Tc values. 2/3 of the data is then used to train a 2-nearest-neighbors model. MAE is printed in the terminal. The model predicts on the remaining 1/3 of the data. A plot with the predicted Tc values vs. the experimental Tc values is saved to "Plots/2 Nearest Neighbors Random.png". It also creates two error plots. One plots the experimental Tc value vs the difference between the experimental and predicted value for each point in the test set. This plot is saved to "./Plots/2 Nearest Neighbors Experimental Error Random.png". The other plots the predicted Tc value vs the difference between the experimental and predicted value for each point in the test set. This plot is saved to "./Plots/2 Nearest Neighbors Predicted Error Random.png".

KNN-DS1.py:
Imports DS1.csv. 2/3 of the data is then used to train a 2-nearest-neighbors model. The model predicts on the remaining 1/3 of the data. MAE is printed in the terminal. A plot with the predicted Tc values vs. the experimental Tc values is saved to "Plots/2 Nearest Neighbors.png". It also creates two error plots. One plots the experimental Tc value vs the difference between the experimental and predicted value for each point in the test set. This plot is saved to "./Plots/2 Nearest Neighbors Experimental Error.png". The other plots the predicted Tc value vs the difference between the experimental and predicted value for each point in the test set. This plot is saved to "./Plots/2 Nearest Neighbors Predicted Error.png".

RandomForest-DS1_train-DS2_test.py:
Imports DS1.csv and DS2.csv. DS1 is then used to train a random forest model. The model predicts on all the data in DS2. MAE is printed in the terminal. A plot with the predicted Tc values vs. the experimental Tc values is saved to "Plots/DS1_train DS2_test Random Forest.png". It also creates two error plots. One plots the experimental Tc value vs the difference between the experimental and predicted value for each point in the test set. This plot is saved to "./Plots/DS1_train DS2_test Random Forest Experimental Error.png". The other plots the predicted Tc value vs the difference between the experimental and predicted value for each point in the test set. This plot is saved to "./Plots/DS1_train DS2_test Random Forest Predicted Error.png".

RandomForest-DS1-MASTML.py:
Imports DS1.csv and DS1-MASTML-Features.csv. 2/3 of DS1 is then used to train a random forest model. The model predicts on the remaining 1/3 of the data. MAE is printed in the terminal. This same process is repeated with DS1-MASTML-Features. Plots with the predicted Tc values vs. the experimental Tc values from both model are overlaid for comparison and saved to "Plots/MASTML Random Forest.png". It also creates two error plots for the model that uses the MAST-ML features. One plots the experimental Tc value vs the difference between the experimental and predicted value for each point in the test set. This plot is saved to "./Plots/MASTML Random Forest Experimental Error.png". The other plots the predicted Tc value vs the difference between the experimental and predicted value for each point in the test set. This plot is saved to "./Plots/MASTML Random Forest Predicted Error.png".


