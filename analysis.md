# Feature Distributions

To better understand the nature of the extracted wallet features, a function to visualize the distribution of all numeric features using histograms was created. The function plot_feature_distributions(df) automatically detects all numeric columns in the DataFrame and creates histograms for each one. This helps in identifying patterns such as skewness, sparsity, or outliers in variables like deposit volume, borrow volume, liquidation count, and others. The histograms are displayed in a grid layout using matplotlib with blue coloring and clear edge outlines, making it easy to interpret. These visualizations are crucial for diagnosing feature imbalance and ensuring meaningful input to the machine learning model.

<pre>
  import matplotlib.pyplot as plt
  import seaborn as sns

  def plot_feature_distributions(df):
      numeric_cols = df.select_dtypes(include=np.number).columns
      df[numeric_cols].hist(bins=30, figsize=(16, 12), color='skyblue', edgecolor='black')
      plt.suptitle("Feature Distributions", fontsize=20)
      plt.tight_layout()
      plt.show()

  plot_feature_distributions(df)
</pre>

<img width="4800" height="3600" alt="feature_distributions" src="https://github.com/user-attachments/assets/51ad70ce-61e7-4752-abb7-5c8e4b6bcc46" />

# Feature Importance

To gain insights into which features most influence the model’s predictions, visualized the feature importances obtained from the trained RandomForestRegressor. After extracting the importance scores using best_model.feature_importances_, paired them with their corresponding feature names and sorted them in descending order. The resulting data was plotted as a horizontal bar chart using Seaborn’s barplot, providing an intuitive view of each feature’s contribution to the credit scoring model. This analysis revealed which behavioral traits such as net_deposit, borrow_usd, liquidations net_repay and deposit_usd had the strongest predictive power, thereby helping to interpret the model and validate that the learned relationships align with domain expectations. 

<pre>
  import matplotlib.pyplot as plt
  import seaborn as sns

  importances = best_model.feature_importances_
  features = X.columns
  importance_df = pd.DataFrame({'feature': features, 'importance': importances})
  importance_df = importance_df.sort_values('importance', ascending=False)

  plt.figure(figsize=(10, 6))
  sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
  plt.title('Feature Importance')
  plt.xlabel('Importance Score')
  plt.ylabel('Feature')
  plt.grid(True)
  plt.tight_layout()
  plt.show()
</pre>

<img width="3000" height="1800" alt="feature_importance" src="https://github.com/user-attachments/assets/eb2881b8-35bb-442f-9881-dd521c73ded2" />

# Credit Score Distribution

To understand how credit scores are distributed across all wallets, generated a histogram using Seaborn’s histplot function. After reading the wallet_scores.csv output file, we plotted the predicted_score column with 20 bins to capture score variations. A kernel density estimate (KDE) curve was overlaid to highlight the overall shape of the distribution. This visualization provides a quick glance at whether the majority of wallets fall into low, medium, or high credit score categories. Peaks in the histogram indicate score ranges where many wallets cluster, while tails suggest fewer but potentially riskier or more reliable users.

<pre>
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns

  df = pd.read_csv('/kaggle/working/wallet_scores.csv') 
  df.head()
  plt.figure(figsize=(10, 6))
  sns.histplot(df['predicted_score'], bins=20, kde=True, color='skyblue')
  plt.title('Wallet Credit Score Distribution')
  plt.xlabel('Predicted Score')
  plt.ylabel('Number of Wallets')
  plt.grid(True)
  plt.show()
</pre>
<img width="3000" height="1800" alt="credit_score_distribution" src="https://github.com/user-attachments/assets/4622221f-8b3e-451d-8aab-88407e81aca9" />

To gain deeper insight into how wallet scores are distributed across defined credit bands, divided the predicted credit scores into decile like bins ranging from 0–100 up to 901–1000. Using pd.cut, each wallet's score was categorized into a labeled score range, such as 0-100, 101-200, and so on. We then computed the count of wallets in each range using value_counts() and visualized the results with a bar chart. This bar graph provides a clear visual representation of how wallets are spread across score segments, making it easier to observe whether the model is biased toward assigning more scores to the higher or lower ends. 

<pre>
  bins = [0,100,200,300,400,500,600,700,800,900,1000]
  labels = ['0-100','101-200','201-300','301-400','401-500',
          '501-600','601-700','701-800','801-900','901-1000']
  df['score_range'] = pd.cut(df['predicted_score'], bins=bins, labels=labels)

  score_counts = df['score_range'].value_counts().sort_index()
  score_counts.plot(kind='bar', color='salmon', figsize=(10, 5))
  plt.title('Number of Wallets in Each Score Range')
  plt.xlabel('Score Range')
  plt.ylabel('Wallet Count')
  plt.xticks(rotation=45)
  plt.grid(True)
  plt.show()
</pre>

<img width="3600" height="1500" alt="range_with_number_of_wallets" src="https://github.com/user-attachments/assets/8b78a85d-1324-435c-8062-30a05d36851e" />

