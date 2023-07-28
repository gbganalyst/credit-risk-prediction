import streamlit as st
import joblib
from datetime import date, timedelta
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model = joblib.load('deployment/credit_risk_model.pkl')

# Create the Streamlit app
def main():
    # Set page configuration
    st.set_page_config(page_title='Credit Risk Model Deployment', layout='wide')

    # Add custom CSS styles
    st.markdown(
        """
        <style>
        .loan-status {
            padding: 10px;
            color: #fff;
            font-weight: bold;
            border-radius: 5px;
        }
        .approved {
            background-color: #42f57b;
        }
        .rejected {
            background-color: #f54242;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title('Credit Risk Model Deployment')
    st.write('Enter the required features below:')

    # Create input fields for the required features
    birthdate = st.number_input('Age', min_value=18, max_value=100, step=1)
    latitude_gps = st.number_input('Latitude GPS')
    longitude_gps = st.number_input('Longitude GPS')
    bank_name_clients = st.text_input('Name of the bank', "GT Bank", key = "placeholder")
    loannumber = st.number_input('The number of the loan that you have to predict')
    creation_day = st.number_input('Day that loan application is created (1-31)', min_value=1, max_value=31)
    approved_day = st.number_input('Day that loan is approved (1-31))', min_value=1, max_value=31)
    creation_dayofweek = st.number_input('Creation weekday (1-7)', min_value=1, max_value=7)
    approved_dayofweek = st.number_input('Approved weekday (1-7)', min_value=1, max_value=7)
    totaldue = st.number_input('Total repayment required to settle the loan', min_value=20.0, value=20.0, format='%f')

    # ... add more input fields for other features

    # Create a button to make predictions
    if st.button('Predict'):
        # Encode categorical features
        label_encoder = LabelEncoder()
        bank_name_clients_encoded = label_encoder.fit_transform([bank_name_clients])

        # Create a data sample with the user inputs
        sample = [[birthdate, latitude_gps, longitude_gps, bank_name_clients_encoded[0],
                   loannumber, creation_day, approved_day, creation_dayofweek,
                   approved_dayofweek, totaldue]]

        # Make predictions using the loaded model
        prediction = model.predict(sample)

        # Determine the loan status and corresponding CSS class for styling
        if prediction == 1:
            loan_status = 'Approved'
            status_class = 'approved'
        else:
            loan_status = 'Rejected'
            status_class = 'rejected'

        # Display the loan status with appropriate styling
        st.markdown(f'<div class="loan-status {status_class}">{loan_status}</div>', unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == '__main__':
    main()
