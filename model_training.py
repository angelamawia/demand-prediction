import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import pickle

# Load the data in chunks and calculate demand
csv_file_path = r'data\transformed_data.csv'
chunk_size = 10000
transformed_chunks = []

for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size):
    demand_counts = chunk.groupby(['year', 'quarter', 'servicename']).size().reset_index(name='demand')
    transformed_chunks.append(demand_counts)

# Concatenate the transformed chunks into a single DataFrame
data = pd.concat(transformed_chunks, ignore_index=True)

# Remove rows with unknowns
data = data[(data['year'] != 'quar') & (data['servicename'] != 'servicename')]

# Ensure 'year' is numerical
data['year'] = pd.to_numeric(data['year'], errors='coerce')
data = data.dropna(subset=['year'])  # Drop rows where 'year' conversion to numeric failed
data['year'] = data['year'].astype(int)

# Initialize LabelEncoders for categorical columns
label_encoders = {}
categorical_columns = ['quarter', 'servicename']

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Save the encoders for future use
joblib.dump(label_encoders, 'label_encoders.pkl')

# Feature engineering
data['moving_avg'] = data.groupby('servicename')['demand'].transform(lambda x: x.rolling(4, min_periods=1).mean())
data['yoy_change'] = data.groupby(['servicename', 'quarter'])['demand'].transform(lambda x: x.diff(periods=4))
data['qoq_change'] = data.groupby(['servicename'])['demand'].transform(lambda x: x.diff())

mean_yoy = data['yoy_change'].mean()
mean_qoq = data['qoq_change'].mean()
data['yoy_change'] = data['yoy_change'].fillna(mean_yoy)
data['qoq_change'] = data['qoq_change'].fillna(mean_qoq)

data['demand_lag1'] = data['demand'].shift(1)
data['demand_lag2'] = data['demand'].shift(2)

data.dropna(inplace=True)

# Separate the feature variables (X) and target variable (y)
X = data.drop('demand', axis=1)
y = data['demand']
print(y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Save the trained model to a file
with open('rf_regressor_model.pkl', 'wb') as model_file:
    pickle.dump(rf_regressor, model_file)


