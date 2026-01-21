import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from  sklearn.metrics import accuracy_score
df = pd.read_csv('ml5.csv')
print(df)
def run_naive_bayes_classification():
    x = df.drop(['species', 's.no.'], axis=1)
    y=df['species']

    encoder=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
    x_encoded = encoder.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x_encoded,y,test_size=0.2, random_state=78)

    model = CategoricalNB(alpha=1.0)
    model.fit(x_train, y_train)

    y_p = model.predict(x_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_p) * 100:.2f}%")
    
    new_data = pd.DataFrame([['white', 'three','short', 'yes']],
                            columns=[ 'color', 'legs', 'height','smelly'])
    
    new_encoded = encoder.transform(new_data)
    prediction = model.predict(new_encoded)

    prediction_proba = model.predict_proba(new_encoded)
    
    print("\n--- Final Result --")
    print (f"New Applicant: White, 3 Legs, Short, Smelly")
    print(f"Predicted Species: **{prediction[0]}**")
    print(f"Prediction probabilities: {prediction_proba}")

if __name__ == "__main__":
    run_naive_bayes_classification()
