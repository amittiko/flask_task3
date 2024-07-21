from flask import Flask, request, jsonify, render_template
from car_data_pre import prepare_data
from model_training import model
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import warnings

warnings.filterwarnings('ignore')


app = Flask(__name__)
#קריאת הנתונים
dataset = pd.read_csv('dataset.csv')
dataset = prepare_data(dataset)
#דאטה סט למודל
dataset_model=dataset.copy()
model(dataset_model)
#פתיחת המודל
model = pickle.load(open('trained_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #קבלת פיצרים מהמשתמש
    features = request.form.getlist('feature')

    # Convert features to DataFrame
    feature_df = prepare_data(features)

    # Prepare data for prediction
    transformed_dataset = dataset[['model', 'Hand', 'Gear', 'capacity_Engine', 'Engine_type','Km', 'Car_Age', 'Region','Km_per_Year', 'Days_Since_Creation', 'company_rank']]
    #חיבור הדאטה והנתונים שהתקבלו מהמשתמש וביצוע נרמול
    df_combined = pd.concat([feature_df, transformed_dataset], axis=0)


    # Normalize specified columns
    scaler = MinMaxScaler()
    columns_to_normalize = ['Car_Age', 'Hand', 'capacity_Engine', 'Km', 'Km_per_Year', 'company_rank',
                            'Days_Since_Creation']
    df_combined[columns_to_normalize] = scaler.fit_transform(df_combined[columns_to_normalize])
  

    # Select required columns
    df_combined = df_combined[
        ['model', 'Hand', 'Gear', 'capacity_Engine', 'Engine_type', 'Km', 'Car_Age', 'Region', 'Km_per_Year',
         'Days_Since_Creation', 'company_rank']]

    # One-hot encode categorical features
    transformed_dataset = pd.get_dummies(df_combined, drop_first=True)

    # Extract normalized features for the first row
    final_features = [transformed_dataset.iloc[0]]
    prediction = round(model.predict(final_features)[0])

    return render_template('index.html', prediction_text='{}'.format(prediction))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))

    app.run(host='0.0.0.0', port=port, debug=True)
