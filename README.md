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


