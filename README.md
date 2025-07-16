# Assignment_Internship
The project builds a robust machine learning model to assign a credit score between 0 and 1000 to each wallet based on their historical transaction behaviour on the Aave V2 protocol. The aim is to differentiate between a reliable and a risky wallet behaviour.

# Dataset
A dataset with a sample size of 100K raw transaction-level data was used, with each record corresponding to deposit, borrow, repay, redeemunderlying and liquidationcall. The data stored in "https://drive.google.com/file/d/1ISFbAXxadMrt7Zl96rmzzZmEKZnyW7FS/view?usp=sharing" was downloaded and added to the Kaggle notebook "https://www.kaggle.com/code/sreekanthravikumar/assignment-internship".

# Feature Engineering

Started the project by adding the necessary libraries and setting up the input and output files.

<pre>
  import json, pandas as pd, numpy as np
  from tqdm import tqdm
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.model_selection import train_test_split, GridSearchCV
  from sklearn.metrics import r2_score, mean_absolute_error 

  input_json = '/kaggle/input/input-for-ml/user-wallet-transactions.json'
  output_csv = '/kaggle/working/wallet_scores.csv'
</pre>



