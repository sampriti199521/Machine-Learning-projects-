# STEP 1: Install required libraries (run only in Google Colab)
# !pip install pandas scikit-learn matplotlib seaborn

# STEP 2: Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# STEP 3: Load dataset (upload 'tmdb_5000_movies.csv' to your Colab/Jupyter environment)
df = pd.read_csv('/content/tmdb_5000_movies.csv')  # Update the path as needed

# STEP 4: Clean data - remove rows with zero budget or revenue
df = df[df['budget'] > 0]
df = df[df['revenue'] > 0]

# STEP 5: Add a 'success' column (1 if revenue > 2x budget)
df['success'] = (df['revenue'] > df['budget'] * 2).astype(int)

# STEP 6: Select features and label
X = df[['budget', 'runtime', 'popularity']]
y = df['success']

# STEP 7: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 8: Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# STEP 9: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# STEP 10: Predict sample outputs
print("Sample Predictions:", model.predict(X_test[:5]))
print("Actual Labels:     ", y_test[:5].values)

📦 Requirements:
	•	pandas
	•	scikit-learn

⸻

🗂 Dataset:

Use this dataset from Kaggle:
🔗 TMDB 5000 Movie Dataset

Save the CSV (tmdb_5000_movies.csv) and upload it to your runtime before running the code.

