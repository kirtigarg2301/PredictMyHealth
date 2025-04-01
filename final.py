from PIL import Image
import joblib
import os
import pickle
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import base64

# Set page configuration
st.set_page_config(
    page_title="Predict My Health",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# Get the working directory of the script
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load the trained machine learning models
model = joblib.load("diabetes_prediction.sav")
scaler = pickle.load(open(os.path.join(working_dir, 'scaler.sav'), 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/heart_disease_model.sav', 'rb'))


# Function to set background image
def set_bg_from_local(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()

    # Inject CSS for background image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: rgba(255, 255, 255, 0.2) url("data:image/jpeg;base64,{encoded_string}") no-repeat center center fixed;
            background-size: cover;
        }}
        .large-font {{
            font-size: 30px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Set background image (Update the file name if necessary)
set_bg_from_local("watermark14.jpg")

# Sidebar menu
st.sidebar.title("Menu")
menu = st.sidebar.radio("Go to", ["Home", "About Disease", "Disease Prediction"])

# Home Page
if menu == "Home":
    st.title("Welcome to Predict My Health")
    st.write("""
    This web application helps users predict the likelihood of developing certain diseases based on input features.
    It uses trained machine learning models to offer predictions for Diabetes and Heart Disease.
    """)

    st.subheader("How to Use:")
    st.write("""
    1. Open the Main Menu (>) in the top-left corner.
    2. Select a category: 'Diabetes Prediction' or 'Heart Disease Prediction'.
    3. Enter the required details.
    4. Click 'Predict' to get the results.
    """)

    st.subheader("Disclaimer:")
    st.write("""
    - This web app may not always provide accurate predictions. Please verify the results.
    - This tool is for informational purposes only. Always consult a doctor for medical advice.
    """)

    st.subheader("Thank You for Visiting! 💙")

# About Disease Page
elif menu == "About Disease":
    st.title("About Diseases")
    disease_type = st.selectbox("Select Disease", ["Diabetes", "Heart Disease"])

    # Information about Diabetes
    if disease_type == "Diabetes":
        st.subheader("Diabetes Disease")
        st.write("""
        Diabetes mellitus is a chronic condition where the body cannot properly regulate blood sugar levels. 
        It occurs due to either a lack of insulin production or the body's inability to use insulin effectively.
        """)

        st.subheader("Types of Diabetes:")
        st.write("""
        - **Type 1 Diabetes:** Autoimmune condition that destroys insulin-producing cells.
        - **Type 2 Diabetes:** Common form, linked to lifestyle factors like poor diet and obesity.
        - **Gestational Diabetes:** Occurs during pregnancy and increases the risk of type 2 diabetes.
         """)

        st.subheader("Causes and Risk Factors:")
        st.write("""
        - **Genetics:** Family history of diabetes.
        - **Obesity:** Excess fat, especially in the abdomen.
        - **Physical Inactivity:** Lack of exercise.
        - **Poor Diet:** High sugar and processed food intake.
        - **Age & Ethnicity:** Higher risk for people over 45 and certain ethnic groups.
        """)

        st.subheader("Symptoms of Diabetes:")
        st.write("""
        - Frequent urination
        - Excessive thirst and hunger
        - Unexplained weight loss
        - Fatigue and blurred vision
        - Slow-healing wounds and infections
        - Numbness or tingling in hands/feet
        """)

        st.subheader("Conclusion:")
        st.write("""
        Managing diabetes requires lifestyle changes, regular check-ups, and medication if necessary. 
        Early diagnosis and proper care can prevent complications.
        """)

    # Information about Heart Disease
    elif disease_type == "Heart Disease":
        st.subheader("Heart Disease")
        st.write("""
        Cardiovascular diseases (CVDs) are the leading cause of death worldwide, accounting for 17.9 million deaths annually.
        These diseases affect the heart and blood vessels, including conditions like coronary artery disease, stroke, and heart failure.
        """)

        st.subheader("Common Types of Heart Disease:")
        st.write("""
        - **Coronary Artery Disease (CAD):** Blockage of coronary arteries leading to heart attacks.
        - **Heart Attack (Myocardial Infarction):** Oxygen deprivation causes heart muscle damage.
        - **Heart Failure:** The heart cannot pump blood efficiently.
        - **Arrhythmias:** Irregular heartbeats affecting circulation.
        - **Valvular Heart Disease:** Damage to heart valves affecting blood flow.
        """)

        st.subheader("Key Risk Factors:")
        st.write("""
        - **Age & Gender:** Older individuals and men have a higher risk.
        - **High Blood Pressure & Cholesterol:** Leads to artery blockages.
        - **Diabetes & Obesity:** Increases stress on the heart.
        - **Smoking & Inactivity:** Damages arteries and increases risk.
        - **Family History:** Genetics play a role.
        """)

        st.subheader("Symptoms of Heart Disease:")
        st.write("""
        - Chest pain or discomfort
        - Shortness of breath
        - Fatigue and weakness
        - Irregular heartbeats or palpitations
        - Swelling in legs, ankles, or feet
        """)

        st.subheader("Machine Learning in Heart Disease Prediction:")
        st.write("""
        Machine learning can analyze medical data to predict heart disease risks. 
        By evaluating factors like age, cholesterol, and blood pressure, it helps doctors diagnose cases early.
        """)

        st.subheader("Conclusion:")
        st.write("""
        Awareness, regular check-ups, and a healthy lifestyle are key to preventing heart disease.
        """)

# Disease Prediction Page
elif menu == "Disease Prediction":
    st.title("Disease Prediction 🚀")
    st.write("This section will allow users to input their medical details and get predictions for diabetes and heart disease.")
    prediction_type = st.radio("Select Prediction Type", ["Diabetes Prediction", "Heart Disease Prediction"])

    # **Diabetes Prediction**
    if prediction_type == "Diabetes Prediction":
        st.subheader("Diabetes Prediction")

        # Define input fields for user data
        age = st.number_input("Age", min_value=1, max_value=120, step=1)
        gender = st.selectbox("Gender", ['Male', 'Female'])
        polyuria = st.selectbox("Polyuria (Excessive urination)", ['No', 'Yes'])
        polydipsia = st.selectbox("Polydipsia (Excessive thirst)", ['No', 'Yes'])
        sudden_weight_loss = st.selectbox("Sudden Weight Loss", ['No', 'Yes'])
        weakness = st.selectbox("Weakness", ['No', 'Yes'])
        polyphagia = st.selectbox("Polyphagia (Excessive hunger)", ['No', 'Yes'])
        genital_thrush = st.selectbox("Genital Thrush", ['No', 'Yes'])
        visual_blurring = st.selectbox("Visual Blurring", ['No', 'Yes'])
        itching = st.selectbox("Itching", ['No', 'Yes'])
        irritability = st.selectbox("Irritability", ['No', 'Yes'])
        delayed_healing = st.selectbox("Delayed Healing", ['No', 'Yes'])
        partial_paresis = st.selectbox("Partial Paresis (Muscle weakness)", ['No', 'Yes'])
        muscle_stiffness = st.selectbox("Muscle Stiffness", ['No', 'Yes'])
        alopecia = st.selectbox("Alopecia (Hair Loss)", ['No', 'Yes'])
        obesity = st.selectbox("Obesity", ['No', 'Yes'])

        # Convert categorical inputs to numerical values
        gender = 0 if gender == 'Male' else 1
        inputs = [age, gender, polyuria, polydipsia, sudden_weight_loss, weakness, polyphagia,
                genital_thrush, visual_blurring, itching, irritability, delayed_healing,
                partial_paresis, muscle_stiffness, alopecia, obesity]

        # Map categorical features to numerical values
        inputs = [1 if val == 'Yes' else 0 if val == 'No' else val for val in inputs]
        
        # Function to make diabetes predictions
        def diabetes_prediction(input_data):
    	    input_data_array = np.array(input_data).reshape(1, -1)
    	    prediction = diabetes_model.predict(input_data_array)
    	    return prediction

        # Prediction button
        if st.button("Predict"):
            # Reshape the inputs and make a prediction
            input_data = np.array(inputs).reshape(1, -1)
            prediction = model.predict(input_data)[0]

            # Display the prediction result
            if prediction == 1:
                st.error("⚠️ You may have Diabetes disease. Consult a doctor for further evaluation.")
            else:
                st.success("✅ You are unlikely to have heart disease. Stay healthy!")

    elif prediction_type == "Heart Disease Prediction":
        st.subheader("Heart Disease Prediction")

        # User input fields
        # Collect user input
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (1 - 120 years)", min_value=1, max_value=120, value=50)
            trestbps = st.number_input("Resting Blood Pressure (80 - 200 mm Hg)", min_value=80, max_value=200, value=120)
            chol = st.number_input("Serum Cholesterol (100 - 600 mg/dl)", min_value=100, max_value=600, value=200)
            thalach = st.number_input("Maximum Heart Rate Achieved (60 - 250 bpm)", min_value=60, max_value=250, value=150)
            oldpeak = st.number_input("ST Depression Induced by Exercise (0.0 - 10.0)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            ca = st.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4, value=0)

        with col2:
            sex = st.selectbox("Sex: 0= Female; 1= Male", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
            cp = st.selectbox("Chest Pain Type (0-4) : 0= Severe Pain(Typical Angina); 1= Mild Pain(Atypical Angina); 3= Discomfort(Non-anginal Pain); 4= No Pain(Asymptomatic)", [0, 1, 2, 3, 4])
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL? : 0= No; 1= Yes", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            restecg = st.selectbox("Resting ECG Results (0-2): 0= Normal; 1= Minor Issue(ST-T wave abnormality); 2= Possible Problem(Left Ventricular Hypertrophy)", [0, 1, 2])
            exang = st.selectbox("Exercise-Induced Angina(Chest Pain During Exercise): 0= No; 1= Yes", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            thal = st.selectbox("Thalassemia Type(Blood Oxygen Issue)(0-3) : 0= normal; 1= Permanent Blockage(Fixed defect); 2= Reversible Issue(Reversible defect); 3= Unknown", [0, 1, 2, 3])

        with col3:
            slope = st.selectbox("Slope of the Peak Exercise ST Segment(Heart Performance During Exercise)(0-2): 0= Upsloping (normal); 1= Flat(borderline); 2= Downsloping(abnormal)", [0, 1, 2])

        if st.button("Predict Heart Disease"):
            try:
                # Prepare input data
                input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
                input_data_df = pd.DataFrame(input_data, columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                                                              'restecg', 'thalach', 'exang', 'oldpeak', 
                                                              'slope', 'ca', 'thal'])

                # Scale input data
                input_data_scaled = scaler.transform(input_data_df)

                # Make prediction
                prediction = heart_disease_model.predict(input_data_scaled)
                probability = heart_disease_model.predict_proba(input_data_scaled)[0][1]  # Get probability

                # Display prediction
                if probability > 0.7:  # Use probability threshold instead of 0.5
                    st.error("⚠️ You may have heart disease. Consult a doctor for further evaluation.")
                else:
                    st.success("✅ You are unlikely to have heart disease. Stay healthy!")
            

            except Exception as e:
                st.error(f"Error: {e}")