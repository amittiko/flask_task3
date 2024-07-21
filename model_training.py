from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pickle


def model(df):
    #נרמול
    scaler = MinMaxScaler()
    columns_to_normalize = ['Car_Age', 'Hand', 'capacity_Engine', 'Km', 'Km_per_Year', 'company_rank', 'Days_Since_Creation']
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    df = df[['model', 'Hand', 'Gear', 'capacity_Engine', 'Engine_type','Km', 'Car_Age', 'Region','Km_per_Year', 'Days_Since_Creation', 'company_rank','Price']] # משתני ההסברה
    df = pd.get_dummies(df, drop_first=True)
    #   מאפיינים
    X = df.drop(columns=['Price'])  # משתני ההסברה
    y = df['Price']  # משתנה המטרה
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # הגדרת המודל ElasticNet עם הפרמטרים האולטימטיביים
    model = ElasticNet(alpha=0.01, l1_ratio=0.95, random_state=40)

    # אימון המודל על נתוני האימון
    model.fit(X_train, y_train)
     #החזרת קובץ פיקל של המודל
    return pickle.dump(model, open("trained_model.pkl", "wb"))