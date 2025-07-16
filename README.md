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

After which a function "extract_features" was created to processes raw transaction-level JSON data to compute wallet-level behavioral features for credit scoring. For each transaction, it identifies the user wallet, transaction type (e.g., deposit, borrow, repay), asset price, and amount involved. These are used to calculate the USD value of each transaction, which is then aggregated per wallet. The function tracks total amounts deposited, borrowed, repaid, redeemed, as well as the number of transactions, liquidations, and active duration of the wallet (from the first to last transaction). It also computes derived features such as net deposit (deposit - redemption), net repayment, repayment ratio, and liquidation rate per transaction. Finally, all wallet-level data is compiled into a structured pandas DataFrame, with missing values filled as zero, ready to be used for machine learning model training or scoring.

<pre>
  def extract_features(data):
    features = {}
    for record in tqdm(data):
        wallet = record['userWallet']
        ts = record['timestamp']
        action = record['action'].lower()
        price = float(record['actionData'].get('assetPriceUSD', 0))
        amount = float(record['actionData'].get('amount', 0))
        usd_value = amount * price

        if wallet not in features:
            features[wallet] = {
                'deposit_usd': 0, 'borrow_usd': 0,
                'repay_usd': 0, 'redeem_usd': 0,
                'liquidations': 0, 'tx_count': 0,
                'first_ts': ts, 'last_ts': ts
            }

        f = features[wallet]
        f['tx_count'] += 1
        f['first_ts'] = min(f['first_ts'], ts)
        f['last_ts'] = max(f['last_ts'], ts)

        if action == 'deposit':
            f['deposit_usd'] += usd_value
        elif action == 'borrow':
            f['borrow_usd'] += usd_value
        elif action == 'repay':
            f['repay_usd'] += usd_value
        elif action == 'redeemunderlying':
            f['redeem_usd'] += usd_value
        elif action == 'liquidationcall':
            f['liquidations'] += 1

    df = pd.DataFrame.from_dict(features, orient='index')
    df['net_deposit'] = df['deposit_usd'] - df['redeem_usd']
    df['net_repay'] = df['repay_usd'] - df['borrow_usd']
    df['duration_days'] = (df['last_ts'] - df['first_ts']) / 86400
    df['repay_ratio'] = df['repay_usd'] / (df['borrow_usd'] + 1)
    df['liquidation_rate'] = df['liquidations'] / (df['tx_count'] + 1)
    df = df.fillna(0)
    return df
  </pre>

# The Model

To generate the target credit score for each wallet, a weakly supervised rule-based formula was applied. Each wallet begins with a neutral base score of 500, representing average or baseline behavior. From there, the score is adjusted using key financial indicators derived from the user's transaction history. Wallets receive a positive boost proportional to their net deposits (deposit amount minus withdrawals), reflecting their commitment to contributing funds to the protocol. Additionally, the total amount repaid on loans further increases the score, emphasizing responsible borrowing behavior. In contrast, wallets with high borrow amounts are penalized to discourage excessive borrowing without sufficient repayment. The most significant penalty comes from liquidation events, each instance of liquidation reduces the score drastically, as it signals poor risk management or overleveraging. The formula is capped to ensure scores fall within a realistic range between 0 and 1000. This approach helps simulate creditworthiness using on-chain activity, enabling the model to learn meaningful patterns even in the absence of real-world credit labels.

<pre>
  def train_model(df):
   
    df['target_score'] = (
        500 + 0.0001 * df['net_deposit']
        + 0.0002 * df['repay_usd']
        - 0.0003 * df['borrow_usd']
        - 50 * df['liquidations']
    ).clip(0, 1000)

    X = df.drop(columns=['target_score'])
    y = df['target_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [50,100, 200],
        'max_depth': [None,2,5, 10, 20],
        'min_samples_split': [2, 5,10],
    }

    base_model = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(base_model, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)

    print("Best Parameters:", grid.best_params_)
    print("R²:", r2_score(y_test, preds))
    print("MAE:", mean_absolute_error(y_test, preds))

    return best_model, df, X
</pre>

To predict these scores more accurately from transaction features, a Random Forest Regressor was tranined, a powerful ensemble learning method that constructs multiple decision trees and averages their predictions to reduce overfitting and improve robustness. To enhance the model's performance, GridSearchCV is used for hyperparameter tuning, searching over combinations of n_estimators, max_depth, and min_samples_split. The train/test split of the data was set to 80:20 and the the CV split for GridSearchCV was set to 3. The model is evaluated using R² (coefficient of determination) and MAE (mean absolute error), achieving an R² close to 0.889 for max_depth=None, min_samples_split=2 and n_estimators=200, indicating decent predictive accuracy.

# Output

A main execution pipeline was created that orchestrates the complete scoring process from raw JSON data to the final output CSV. It begins by loading the user-level Aave V2 transaction data using json.load, which reads the structured transaction records from the input file. The extract_features() function is then called to transform these records into wallet-level features, capturing important metrics like total deposits, borrows, repayments, liquidations, and activity duration.

<pre>
  with open(input_json, "r") as f:
    data = json.load(f)

  df = extract_features(data)

  best_model, df, X = train_model(df)

  df['predicted_score'] = best_model.predict(X).round().clip(0, 1000).astype(int)
  df.reset_index(inplace=True)
  df.rename(columns={'index': 'wallet'}, inplace=True)

  df[['wallet', 'predicted_score']].to_csv(output_csv, index=False)
  print("File saved")
</pre>

Once feature engineering is complete, the train_model() function is invoked. This step not only defines a synthetic target score using a rule-based formula but also trains a Random Forest Regressor with hyperparameter tuning via GridSearchCV. The trained model is then used to predict scores for all wallets in the dataset. These predicted scores are rounded, clipped between 0 and 1000, and cast to integers for consistency. The predicted score of each wallet was appended and saved to a CSV file named as wallet_score.csv.




