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





