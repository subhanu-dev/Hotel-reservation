import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
from sklearn.preprocessing import LabelEncoder
import pickle

# Streamlit page configuration
st.set_page_config(
    page_title="Hotel Reservation Prediction",
    layout="wide",
    page_icon="🏨",
)

st.title("Hotel Reservation Status Prediction 🏨")
st.write("This app predicts the reservation status based on the details you provide.")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["M", "F"])
    ethnicity = st.selectbox("Ethnicity", ["Latino", "African American", "Caucasian", "Asian American"])
    education_level = st.selectbox("Education Level", ["Mid-School", "High-School", "College", "Grad"])
    hotel_type = st.selectbox("Hotel Type", ["City Hotel", "Airport Hotels", "Resort"])
    meal_type = st.selectbox("Meal Type", ["BB", "FB", "HB"])
    deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Non-Refundable", "Refundable"])
    

with col2:
    booking_channel = st.selectbox("Booking Channel", ["Online", "Agent", "Direct"])
    use_promotion = st.selectbox("Use Promotion", ["Yes", "No"])
    discount_rate = st.selectbox("Discount Rate (%)", [0, 5, 10, 15, 20, 25, 30])
    room_rate = st.number_input("Room Rate", min_value=100.0, max_value=1000.0, value=150.0, step=10.0)
    adults = st.number_input("Adults", min_value=1, max_value=6, value=1, step=1)
    children = st.number_input("Children", min_value=0, max_value=6, value=0, step=1)

# Booking dates with validation

booking_date = st.date_input("Booking Date")
expected_checkin = st.date_input("Expected Check-in Date", min_value=booking_date)
expected_checkout = st.date_input("Expected Checkout Date", min_value=expected_checkin)

# Submit button
if st.button("Submit"):
    # Validate dates
    if expected_checkin < booking_date:
        st.error("Check-in date cannot be before booking date")
    elif expected_checkout <= expected_checkin:
        st.error("Checkout date must be after check-in date")
    else:
        # Prepare the input dataframe
        df = pd.DataFrame(
            {
                "Use_Promotion": [use_promotion],
                "Gender": [gender],
                "Deposit_type": [deposit_type],
                "Ethnicity": [ethnicity],
                "Educational_Level": [education_level],
                "Hotel_Type": [hotel_type],
                "Booking_channel": [booking_channel],
                "Discount_Rate": [discount_rate],
                "Room_Rate": [room_rate],
                "Adults": [adults],
                "Children": [children],
                "Expected_checkin": [expected_checkin],
                "Expected_checkout": [expected_checkout]
            }
        )

        # Feature Engineering
        def feature_engineering(df):
            df["Expected_checkin"] = pd.to_datetime(df["Expected_checkin"])
            df["Expected_checkout"] = pd.to_datetime(df["Expected_checkout"])
            df["Length_of_Stay"] = (df["Expected_checkout"] - df["Expected_checkin"]).dt.days
            df["Expected_Checkin_Day_of_Week"] = df["Expected_checkin"].dt.dayofweek
            df["Group_Size"] = df["Adults"] + df["Children"]
            df["Total_Cost"] = df["Length_of_Stay"] * df["Room_Rate"]
            
            # Apply cost multipliers for large groups
            df.loc[df['Group_Size'] > 5, 'Total_Cost'] = df['Total_Cost']  * 2  # Multiply by 2
            df.loc[df['Group_Size'] > 10, 'Total_Cost'] = df['Total_Cost']  * 3
            
            # Calculate final cost after discount
            df["Final_Cost"] = df["Total_Cost"] * (1 - df["Discount_Rate"] / 100)
            df = df[['Use_Promotion', 'Gender', 'Deposit_type', 'Ethnicity', 'Educational_Level', 'Hotel_Type', 'Expected_Checkin_Day_of_Week', 'Booking_channel', 'Final_Cost', 'Discount_Rate']]
            return df

        # Data Preprocessing
        def preprocess_input_data(df):
            ordinal_mappings = {
                "Educational_Level": {"Mid-School": 0, "High-School": 1, "College": 2, "Grad": 3},
                "Use_Promotion": {"No": 0, "Yes": 1},
                "Discount_Rate": {0: 0, 5: 1, 10: 2, 15: 3, 20: 4, 25: 5, 30: 6}
            }
            
            # Apply ordinal mappings
            for col, mapping in ordinal_mappings.items():
                df[col] = df[col].map(mapping)


            with open('encoders.pkl', 'rb') as file:
                encoders = pickle.load(file)
                

                
            # Label Encoding for categorical features
            
            nominal_cols = ["Gender", "Ethnicity", "Hotel_Type", "Deposit_type", "Booking_channel"]
            for col in nominal_cols:
                encoder = encoders[col]
                df[col] = encoder.transform(df[col])

            
            # Drop unnecessary columns
            columns_to_drop = ["Expected_checkin", "Expected_checkout", "Children", "Adults"]
            df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1, inplace=True)
            df
            return df

        # Load Model and Predict
        @st.cache_resource
        def load_model():
            try:
                return joblib.load("rf_model_top_10.pkl")
            except Exception as e:
                st.error(f"Error loading model file: {e}")
                return None

        def predict_reservation(df):
            model = load_model()
            if model is None:
                return None
            try:
                df = feature_engineering(df)
                processed_data = preprocess_input_data(df)
                prediction = model.predict(processed_data)
                return prediction
            except Exception as e:
                st.error(f"Prediction error: {e}")
                return None

        # Make Prediction
        prediction = predict_reservation(df)

        # Display Result
        if prediction is not None:
            reservation_status = {1: "Check-in", 2: "Cancelled", 3: "No-show"}
            status = reservation_status.get(prediction[0], "Unknown")
            st.success(f"### The predicted reservation status is: **{status}**")