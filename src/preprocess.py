import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
df = pd.read_csv("../data/raw/phishing_data.csv")
print("Loaded raw data")

# Drop columns that are identifiers or non-predictive
df_clean = df.drop(columns=['FILENAME','Domain','URL','Title'])
print("Dropped FILENAME, Domain, URL, Title columns for being identifiers / non-predictive")

# Check unique TLD values
print("Checking unique TLD values")
print("Unique TLDs: ", df_clean['TLD'].nunique()) # ==> 695 too sparse, dropping it is better

# Drop TLD column
df_clean = df_clean.drop(columns=['TLD'])
print("Dropped TLD column for bein too sparse")

# Split data into features and target
X = df_clean.drop(columns=['label'])
y = df_clean['label']
print("split data into features and target")

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("split data into train/test sets")

# Scale numerical values (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("trained scaler from training set and applied to testing test")

# Save scaler
joblib.dump(scaler, '../models/scaler.pkl')
print("saved scaler in models/scaler.pkl")

# Save scaled data (index false means dont include row numbers)
pd.DataFrame(X_train_scaled).to_csv('../data/processed/X_train_scaled.csv', index=False)
pd.DataFrame(X_test_scaled).to_csv('../data/processed/X_test_scaled.csv', index=False)
y_train.to_csv('../data/processed/y_train.csv', index=False)
y_test.to_csv('../data/processed/y_test.csv', index=False)
print("saved scaled data in data/processed/*")