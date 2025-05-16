# Enhancing Hotel Revenue Management Through Predictive Analytics

This project leverages machine learning to predict hotel booking outcomes—whether a customer will check in, cancel, or not show up—for Hotel A, a chain operating airport hotels, resorts, and city hotels. By analyzing historical booking data, the project aims to provide actionable insights and tools to optimize resource allocation, refine cancellation policies, and mitigate financial losses from unfulfilled bookings.

## Project Overview

The hotel industry faces significant revenue and operational challenges due to booking cancellations and no-shows. This project addresses these issues for Hotel A by:
- Conducting exploratory data analysis (EDA) to uncover patterns in reservation statuses.
- Developing predictive models to classify booking outcomes (Check-In, Cancel, No-Show).
- Deploying a user-friendly application to predict reservation statuses based on booking details.

The primary evaluation metric is the F1-score, chosen to handle the dataset's class imbalance, where "Check-In" is the majority class, and "Cancel" and "No-Show" are minority classes.

## Dataset

The dataset comprises:
- **Training Set**: 27,500 booking records.
- **Validation Set**: 2,749 booking records.

Each record includes over 20 attributes, such as reservation details (e.g., check-in/check-out dates, booking channel), customer demographics (e.g., age, gender, income), and financial details (e.g., room rate, discount rate). Due to privacy and size constraints, the dataset is not included in this repository but is described in the report.

Key characteristics:
- **Target Variable**: Reservation Status (Check-In, Cancel, No-Show).
- **Class Distribution**: Moderately imbalanced, with Check-In dominating (e.g., 21,240 in training), followed by Cancel (4,134) and No-Show (2,125).

## Methodology

The project follows a comprehensive workflow:

- **Data Inspection and Cleaning**:
  - Handled missing values (e.g., replaced nulls in "Babies" with 0).
  - Removed duplicates (none found) and dropped irrelevant columns (e.g., Reservation_ID).
  - Corrected inconsistent naming (e.g., unified "Check-Out" to "Check-In").

- **Feature Engineering**:
  - Created features like Length of Stay (checkout - checkin), Lead Time (checkin - booking date), Age Group (binned ages), and Final Cost (adjusted for discounts).
  - Encoded categorical variables (ordinal for ordered features, one-hot for nominal).

- **Exploratory Data Analysis (EDA)**:
  - Analyzed patterns in reservation statuses by hotel type, lead time, and demographics.
  - Identified key factors influencing cancellations and no-shows (e.g., lead time, promotions).

- **Model Training**:
  - Tested multiple models: Random Forest, XGBoost, LightGBM, Logistic Regression, and MLP Classifier.
  - Addressed class imbalance using techniques like class weighting, oversampling (SMOTENC), and feature selection (top 10 features via Mean Decrease in Accuracy).

- **Evaluation**:
  - Used F1-score to assess performance, focusing on minority class prediction.
  - Performed hyperparameter tuning on top models (Random Forest, Logistic Regression, MLP).

- **Customer Segmentation**:
  - Applied K-means clustering (4 clusters) to group customers based on demographics and booking behavior.

## Models and Results

Several models were trained and evaluated, with the following key findings:

| Model                                  | Accuracy | Precision | Recall | F1-Score |
|----------------------------------------|----------|-----------|--------|----------|
| Random Forest (Baseline)              | 0.66     | 0.61      | 0.45   | 0.46     |
| Random Forest (Class Balancing)       | 0.70     | 0.65      | 0.68   | 0.64     |
| **Random Forest (Balancing + Feature Selection)** | **0.72** | **0.66**  | **0.70** | **0.66** |
| XGBoost (Balanced + Feature Selection)| 0.71     | 0.63      | 0.66   | 0.64     |
| Logistic Regression (Balancing + Feature Selection) | 0.71 | 0.66 | 0.70 | 0.66 |
| MLP Classifier                        | 0.69     | 0.59      | 0.57   | 0.58     |

- **Champion Model**: Random Forest with class balancing and feature selection achieved the highest F1-score of 0.66.
- **Challenges**: The F1-score plateaued around 0.65-0.66, reflecting difficulties in distinguishing "No-Show" and "Cancel" classes due to their overlap and smaller sample sizes.

## Deployment

The final Random Forest model is deployed as an interactive web application using Streamlit:
- **Functionality**: Users input booking details (e.g., lead time, hotel type, demographics) and receive a predicted reservation status.
- **Access**: Available on the [Streamlit Community Cloud](https://hotel-reservation-numlock.streamlit.app/).
- **Implementation**: The model was serialized with `joblib`, and preprocessing steps (scaling, encoding) are applied to user inputs to match training conditions.

## Conclusion

This project highlights the potential of predictive analytics in hospitality management, offering Hotel A insights to enhance revenue management. Key takeaways:
- Predictive models can identify at-risk bookings, aiding resource planning.
- The overlap between "No-Show" and "Cancel" limits classification precision, suggesting supplementary policies (e.g., stricter cancellation rules) may be needed.
- Customer segmentation reveals distinct guest profiles (e.g., budget solo travelers vs. promotion-seeking families), enabling targeted strategies.
