from django.db import IntegrityError
from django.http import JsonResponse
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .models import User
from django.contrib.auth import authenticate, login, logout, get_user_model
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_protect

import pandas as pd
import joblib
import pickle
import re


def home(request):
    return render(request, 'home.html')


@csrf_protect
def register(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')

        # Validate username
        username_pattern = r'^[a-zA-Z][a-zA-Z0-9_]*$'
        if not re.match(username_pattern, username):
            return JsonResponse({'username_error': 'Username should start with a letter and '
                                                   'contain only letters, numbers, and underscores.'})

        # Validate email
        email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        if not re.match(email_pattern, email):
            return JsonResponse({'email_error': 'Invalid email format.'})

        password_pattern = r'^(?=.*\d)(?=.*[a-z])(?=.*[A-Z])(?=.*[!@#$%^&*()_+=-]).*$'
        if not re.match(password_pattern, password) or len(password) < 8:
            messages.error(request,'Password should be at least 8 characters long and contain at least one '
                         'uppercase letter, one lowercase letter, one number, and one special character.')
            return render(request, 'login_and_signup.html')

        try:
            # Create new user
            user = User.objects.create_user(username=username, email=email, password=password)
            # Redirect to login page
            return redirect('login')
        except IntegrityError:
            return JsonResponse({'email_error': 'Email already exists.'})
    return render(request, 'login_and_signup.html')


@csrf_protect
def login_user(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        # Validate email
        email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        if not re.match(email_pattern, email):
            return render(request, 'login_and_signup.html', {'error': 'Invalid email format.'})

        # Authenticate user
        user = authenticate(request, username=email, password=password)

        if user is not None:
            login(request, user)
            return redirect('main')
        else:
            return render(request, 'login_and_signup.html', {'error': 'Invalid email or password.'})

    return render(request, 'login_and_signup.html')


@login_required(login_url='/login/')
def main_view(request):
    return render(request, 'main.html', {})


def preprocess_inputs(data, label_encoders, full_data):
    features = ['servicename', 'quarter', 'year']
    X = data[features].copy()

    print("User input data before encoding:")
    print(X)

    # Apply LabelEncoder for string columns in the user input
    for col in features:
        le = label_encoders.get(col)
        if le:
            X[col] = le.transform(X[col])

    print("User input data after encoding:")
    print(X)

    # Get the encoded servicename value from the user input
    encoded_servicename = X['servicename'].iloc[0]
    print(f"Encoded servicename value: {encoded_servicename}")

    # Filter full_data for the specific encoded servicename
    full_data_filtered = full_data[full_data['servicename'] == encoded_servicename].copy()

    # Print the filtered historical data
    print("Filtered historical data:\n", full_data_filtered)

    # Calculate moving average
    full_data_filtered['moving_avg'] = full_data_filtered.groupby('servicename')['demand'].transform(lambda x: x.rolling(4, min_periods=1).mean())

    # Calculate year-over-year change in demand
    full_data_filtered['yoy_change'] = full_data_filtered.groupby(['servicename', 'quarter'])['demand'].transform(lambda x: x.diff(periods=4))

    # Calculate quarter-over-quarter change in demand
    full_data_filtered['qoq_change'] = full_data_filtered.groupby(['servicename'])['demand'].transform(lambda x: x.diff())

    # Replace NaN values in yoy_change with 0
    full_data_filtered['yoy_change'] = full_data_filtered['yoy_change'].fillna(0)

    # Fill NaN values with the mean of the changes (excluding NaN values)
    #mean_yoy = full_data_filtered['yoy_change'].dropna().mean()
    mean_qoq = full_data_filtered['qoq_change'].dropna().mean()
    #full_data_filtered['yoy_change'] = full_data_filtered['yoy_change'].fillna(mean_yoy)
    full_data_filtered['qoq_change'] = full_data_filtered['qoq_change'].fillna(mean_qoq)

    # Drop rows with NaN values in yoy_change column
    print("full_data_filtered after  filling nan:\n",
          full_data_filtered['yoy_change'].value_counts(dropna=False))

    # Create lag features
    full_data_filtered['demand_lag1'] = full_data_filtered['demand'].shift(1)
    full_data_filtered['demand_lag2'] = full_data_filtered['demand'].shift(2)

    # Fill NaN values for lag features with zero
    full_data_filtered['demand_lag1'] = full_data_filtered['demand_lag1'].fillna(0)
    full_data_filtered['demand_lag2'] = full_data_filtered['demand_lag2'].fillna(0)


    full_data_filtered = full_data_filtered.drop(['servicename', 'quarter', 'year', 'demand'], axis=1)
    print("full_data_filtered after  dropping servicename,quarter,year and demand:\n",
          full_data_filtered['yoy_change'].value_counts(dropna=False))
    # Get the last row of the filtered data (most recent entry) and reset the index
    if not full_data_filtered.empty:
        latest_features = full_data_filtered.tail(1).reset_index(drop=True)
    else:
        # Create a default row if there's no historical data for the servicename
        default_features = {
            'moving_avg': [0],
            'yoy_change': [0],
            'qoq_change': [0],
            'demand_lag1': [0],
            'demand_lag2': [0]
        }
        latest_features = pd.DataFrame(default_features)

    # Concatenate the latest features with the encoded user input data
    data_with_history = pd.concat([latest_features, X.reset_index(drop=True)], axis=1)

    # Replace NaN values in yoy_change with 0
    data_with_history['yoy_change'] = data_with_history['yoy_change'].fillna(0)
    return data_with_history


@login_required(login_url='/login/')
def predict_demand(request):
    if request.method == 'POST':
        try:
            # Collect form data
            form_data = {
                'quarter': request.POST['quarter'],
                'year': int(request.POST['year']),
                'servicename': request.POST['servicename'],
            }

            print("Form Data:", form_data)

            # Check for empty fields
            if any(value == '' for value in form_data.values()):
                messages.error(request, 'Please fill in all required fields.')
                return render(request, 'main.html')

            data = pd.DataFrame([form_data])
            print("Data:\n", data)

            # Load the model and encoders
            with open('rf_regressor_model.pkl', 'rb') as file:
                model = pickle.load(file)
            print(type(model))

            label_encoders = joblib.load('label_encoders.pkl')
            print("Label encoders loaded successfully.")

            # Load the full historical data in chunks and calculate demand
            csv_file_path = r'data\transformed_data.csv'
            chunk_size = 10000
            transformed_chunks = []

            for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size):
                demand_counts = chunk.groupby(['year', 'quarter', 'servicename']).size().reset_index(name='demand')
                transformed_chunks.append(demand_counts)

            full_data = pd.concat(transformed_chunks, ignore_index=True)

            # Remove rows with unknowns
            full_data = full_data[(full_data['year'] != 'quar') & (full_data['servicename'] != 'servicename')]

            # Ensure 'year' is numerical
            full_data['year'] = pd.to_numeric(full_data['year'], errors='coerce')
            full_data = full_data.dropna(subset=['year'])  # Drop rows where 'year' conversion to numeric failed
            full_data['year'] = full_data['year'].astype(int)

            # Initialize LabelEncoders for categorical columns
            categorical_columns = ['quarter', 'servicename']

            for column in categorical_columns:
                full_data[column] = label_encoders[column].transform(full_data[column])

            # Feature engineering
            full_data['moving_avg'] = full_data.groupby('servicename')['demand'].transform(lambda x: x.rolling(4, min_periods=1).mean())
            full_data['yoy_change'] = full_data.groupby(['servicename', 'quarter'])['demand'].transform(lambda x: x.diff(periods=4))
            full_data['qoq_change'] = full_data.groupby(['servicename'])['demand'].transform(lambda x: x.diff())

            mean_yoy = full_data['yoy_change'].mean()
            mean_qoq = full_data['qoq_change'].mean()
            full_data['yoy_change'] = full_data['yoy_change'].fillna(mean_yoy)
            full_data['qoq_change'] = full_data['qoq_change'].fillna(mean_qoq)

            full_data['demand_lag1'] = full_data['demand'].shift(1)
            full_data['demand_lag2'] = full_data['demand'].shift(2)

            full_data.dropna(inplace=True)

            # Preprocess inputs
            preprocessed_data = preprocess_inputs(data, label_encoders, full_data)
            if preprocessed_data is None:
                raise ValueError("Preprocessing failed. Check logs for details.")

            print("Preprocessed data:\n", preprocessed_data)

            # Check if there is any historical data for the given servicename
            if preprocessed_data['moving_avg'].sum() == 0:
                messages.error(request, "No historical data found for the specified service. Unable to make a prediction.")
                return render(request, 'main.html')

            # Make the prediction
            # Get the feature columns from preprocessed_data
            feature_cols = ['year','quarter','servicename','moving_avg', 'yoy_change', 'qoq_change', 'demand_lag1', 'demand_lag2']

            X = preprocessed_data[feature_cols]

            # Fill rows with NaN values in X
            X = X.fillna(0)

            # Make the prediction
            prediction = model.predict(X)
            predicted_demand = int(prediction[0])

            input_data = data.to_dict('records')[0]

            return render(request, 'predict.html', {
                    'predicted_demand': predicted_demand,
                    'input_data': input_data
                })

        except Exception as e:
            print("Error:", str(e))
            messages.error(request, str(e))
            return render(request, 'main.html')

    return render(request, 'main.html')


@login_required(login_url='/login/')
def show_prediction(request, predicted_demand, input_data):
    return render(request, 'predict.html', {'predicted_demand': predicted_demand, 'input_data': input_data})


# logout page
def user_logout(request):
    logout(request)
    return redirect('home')
