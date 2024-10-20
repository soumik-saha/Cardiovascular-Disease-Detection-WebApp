import streamlit as st
import pandas as pd
import tensorflow as tf

# Load your pre-trained model
try:
    model = tf.keras.models.load_model('cardio.h5')
except Exception as e:
    st.error(f"Error loading model: {str(e)}")

# Streamlit app title
st.title("Heart Disease Prediction App")

# Input fields
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
smoking = st.selectbox("Smoking Status", options=[0, 1])  # 0: No, 1: Yes
alcohol_drinking = st.selectbox("Alcohol Drinking Status", options=[0, 1])
stroke = st.selectbox("Stroke History", options=[0, 1])
physical_health = st.number_input("Physical Health in Past 30 Days", min_value=0, max_value=30, value=0)
mental_health = st.number_input("Mental Health in Past 30 Days", min_value=0, max_value=30, value=0)
diff_walking = st.selectbox("Difficulty Walking", options=[0, 1])  # 0: No, 1: Yes
diabetic = st.selectbox("Diabetic Status", options=[0, 1])  # 0: No, 1: Yes
physical_activity = st.selectbox("Physical Activity Status", options=[0, 1])
sleep_time = st.number_input("Sleep Time (Hours)", min_value=1, max_value=24, value=7)
asthma = st.selectbox("Asthma Status", options=[0, 1])
kidney_disease = st.selectbox("Kidney Disease Status", options=[0, 1])
skin_cancer = st.selectbox("Skin Cancer Status", options=[0, 1])
sex_male = st.selectbox("Sex (Male=1, Female=0)", options=[0, 1])

age_category = st.selectbox("Age Category", options=["25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"])
race = st.selectbox("Race", options=["Asian", "Black", "Hispanic", "Other", "White"])
gen_health = st.selectbox("General Health", options=["Fair", "Good", "Poor", "Very good"])

# Button to predict
if st.button("Predict"):
    # Create DataFrame for input
    input_data = pd.DataFrame({
        "BMI": [bmi],
        "Smoking": [smoking],
        "AlcoholDrinking": [alcohol_drinking],
        "Stroke": [stroke],
        "PhysicalHealth": [physical_health],
        "MentalHealth": [mental_health],
        "DiffWalking": [diff_walking],
        "Diabetic": [diabetic],
        "PhysicalActivity": [physical_activity],
        "SleepTime": [sleep_time],
        "Asthma": [asthma],
        "KidneyDisease": [kidney_disease],
        "SkinCancer": [skin_cancer],
        "Sex_Male": [sex_male]
    })

    # Add dummy columns for categorical variables
    age_categories = ["25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"]
    for category in age_categories:
        input_data[f"AgeCategory_{category}"] = 1 if age_category == category else 0

    races = ["Asian", "Black", "Hispanic", "Other", "White"]
    for race_option in races:
        input_data[f"Race_{race_option}"] = 1 if race == race_option else 0

    gen_health_options = ["Fair", "Good", "Poor", "Very good"]
    for health_option in gen_health_options:
        input_data[f"GenHealth_{health_option}"] = 1 if gen_health == health_option else 0

    # Ensure correct data types
    input_data = input_data.astype(float)

    # Keep only the relevant columns based on your model's training
    model_input_shape = model.input_shape[1]
    relevant_columns = input_data.columns[:model_input_shape]
    input_data = input_data[relevant_columns]

    # Print shape and types for debugging
    st.write("Input Data Shape:", input_data.shape)
    st.write("Input Data Types:", input_data.dtypes)

    # Make predictions
    try:
        predictions = model.predict(input_data)
        predictions_binary = (predictions > 0.5).astype(int)

        # Display result
        st.write("Predicted Heart Disease:", predictions_binary.flatten()[0])
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# To run the app, use the command:
# streamlit run streamlit_app.py
