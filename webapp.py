# This programme deermien whether a pateint have diabetes using machine learning and python

# import library
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

# Create a title and a sub-title
st.write("""
# Outside Lipinski Rule of 5
Detect if molecule can be orally administered
""")

# Open and display an image
image = Image.open(
    '05-Blog-Oral-Drugs-L.jpg')
st.image(image, caption='ML', use_column_width=True)

# Get the dat
df = pd.read_csv('diabetes.csv')
# Set a subheader
#st.subheader('Data Information')
# show the data as a table
#st.dataframe(df)
# Show statistics on the data
#st.write(df.describe())
# Show the data as a chart
# chart = st.bar_chart(df)

# Split the data into independent 'X' an dependent 'Y' as variables
X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values

# Split the data into 75% Training and 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)

# Get the feature input from the user


def get_user_input():
    pregnancies = st.sidebar.number_input('Molecular_weight')
    glucose = st.sidebar.number_input('LogP')
    blood_pressure = st.sidebar.number_input('LogD')
    skin_thickness = st.sidebar.number_input('Hydrogen Bond Acceptor (HBA)')
    Insulin = st.sidebar.number_input('Hydrogen Bond Doner (HDA)')
    BMI = st.sidebar.number_input('PSA')
    DPF = st.sidebar.number_input(
        'Rotatory Bond (ROT)')
    age = st.sidebar.number_input('N/A')


# Store a dictionary into a variable
    user_data = {'pregnancies_months': pregnancies,
                 'glucose_mg/dL': glucose,
                 'blood_pressure_mmHg': blood_pressure,
                 'skinthickness_0.1mm': skin_thickness,
                 'Insulin': Insulin,
                 'BMI': BMI,
                 'DiabetesPedigreeFunction_DPF': DPF,
                 'age': age
                 }

# Transform the data into a dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features


# Transform the data into a dataframe
user_input = get_user_input()

# Set a subheader and display the users input
# st.subheader('User Input: ')
# st.write(user_input)

# Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

# Show the model metrics
st.subheader('Model Test Accurancy Score:')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%')
# Store the models prediction in a variable
prediction = RandomForestClassifier.predict(user_input)

# Set a subheader to dispaly the classifiction
st.subheader('Classification: ')
st.write(prediction)


# Display link to source code
st.subheader('Soruce Code: ')
st.write('https://github.com/Bruce95317/OutsideRO5')
