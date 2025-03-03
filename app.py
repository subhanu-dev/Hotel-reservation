import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime


# importing the model

st.set_page_config(
    page_title="Hotel Reservation Prediction",
    layout="wide",
    page_icon="🏨",
)

st.title("Hotel Reservation Status Prediction 🏨")
st.write("This app predicts the reservation status based on the details you provide.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["M", "F"])
    ethnicity = st.selectbox(
        "Ethnicity", ["Latino", "African American", "Caucasian", "Asian American"]
    )
    education_level = st.selectbox(
        "Education Level", ["High-School", "College", "Grad", "Mid-School"]
    )
    income = st.selectbox("Income", ["<25K", "50K -- 100K", ">100K", "25K --50K"])
    country_region = st.selectbox("Country Region", ["North", "East", "South", "West"])
    hotel_type = st.selectbox("Hotel Type", ["City Hotel", "Airport Hotels", "Resort"])
    meal_type = st.selectbox("Meal Type", ["BB", "FB", "HB"])
    visited_previously = st.selectbox("Visited Previously", ["No", "Yes"])
    previous_cancellations = st.selectbox("Previous Cancellations", ["No", "Yes"])
    deposit_type = st.selectbox(
        "Deposit Type", ["No Deposit", "Non-Refundable", "Refundable"]
    )
    booking_channel = st.selectbox("Booking Channel", ["Online", "Agent", "Direct"])

with col2:
    required_car_parking = st.selectbox("Required Car Parking", ["Yes", "No"])
    use_promotion = st.selectbox("Use Promotion", ["Yes", "No"])
    discount_rate = st.selectbox("Discount Rate", [0, 5, 10, 25, 20, 15, 30])
    room_rate = st.number_input(
        "Room Rate", min_value=100.0, max_value=1000.0, value=150.0
    )
    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    adults = st.number_input("Adults", min_value=1, max_value=6)
    children = st.number_input("Children", min_value=0, max_value=6)
    babies = st.number_input("Babies", min_value=0, max_value=2)
    reservation_status = st.selectbox(
        "Reservation Status", ["check-in", "canceled", "no-show"]
    )

# Dates are displayed separately below the columns
booking_date = st.date_input("Booking Date", value=datetime.date.today())
expected_checkin = st.date_input("Expected Check-in Date")
expected_checkout = st.date_input("Expected Checkout Date")


st.button("Submit")

# # Prepare the input data
# input_data = pd.DataFrame(
#     {
#         "Gender": [gender],
#         "Ethnicity": [ethnicity],
#         "Educational_Level": [education_level],
#         "Income": [income],
#         "Country_region": [country_region],
#         "Hotel_Type": [hotel_type],
#         "Meal_Type": [meal_type],
#         "Previous_Cancellations": [previous_cancellations],
#         "Deposit_type": [deposit_type],
#         "Booking_channel": [booking_channel],
#         "Required_Car_Parking": [required_car_parking],
#         "Use_Promotion": [use_promotion],
#         "Discount_Rate": [discount_rate],
#         "Room_Rate": [room_rate],
#         "Age": [age],
#         "Adults": [adults],
#         "Children": [children],
#         "Babies": [babies],
#         "Expected_checkin": [expected_checkin],
#         "Expected_checkout": [expected_checkout],
#     }
# )

# # Preprocess the input data
# processed_data = preprocess_input_data(input_data)

# # Predict the reservation status
# prediction = model.predict(processed_data)

# # Display the prediction
# reservation_status = {1: "Check-in", 2: "Cancelled", 3: "No-show"}
# st.write(f"The predicted reservation status is: {reservation_status[prediction[0]]}")
