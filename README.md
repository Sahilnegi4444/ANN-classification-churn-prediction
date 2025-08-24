# Customer Churn Prediction using ANN

This project is an end-to-end Artificial Neural Network (ANN) classification model for predicting customer churn. The aim of this project was to understand the working of neural network parameters and build a generalized model for customer churn prediction across France, Germany, and Spain.

# Project Workflow
1️⃣ Data Cleaning & Feature Engineering

• Handled missing values and performed data preprocessing.

• Encoded categorical variables such as Geography (France, Germany, Spain) and Gender.

• Scaled numerical features for better model convergence.

• Split the dataset into training and testing sets.

2️⃣ Building the Deep Neural Network

• Implemented a Deep Learning model using ANN for binary classification.

• Used around 3000 paramters for training model.

• Used Dense layers, Dropout, L2 Regularization and Activation functions to prevent overfitting.

• Optimized hyperparameters for model generalization.

3️⃣ Model Training & Monitoring

• Trained the model on the prepared dataset.

• Used TensorBoard to visualize:

• Training accuracy vs Validation accuracy

• Training loss vs Validation loss

• Model learning curves

4️⃣ Deployment with Streamlit

Built a Streamlit web application to make predictions on new customer data.

The app takes user inputs (age, geography, balance, etc.) and predicts whether the customer is likely to churn or not.

🛠️ Tech Stack

• Python

• Pandas, NumPy → Data Preprocessing & Feature Engineering

• TensorFlow / Keras → Deep Neural Network

• TensorBoard → Model Monitoring

• Streamlit → Deployment

## 📂 Project Structure

```bash
📦 customer-churn-ann
├── Churn_Modelling/          # Dataset (raw/processed)
├── experiments/              # Jupyter notebooks for EDA & model building
├── model/                    # Saved models & weights
├── app/streamlit_app.py      # Streamlit application file
├── logs/                     # TensorBoard logs
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
├── label_encoder_gender      # converting categorical feature into numerical
└── onehot_encoder_geo        # converting categorical feature into numerical
```

📈 Results

• Built a generalized ANN model that predicts customer churn effectively.

• TensorBoard visualizations helped in tuning hyperparameters.

• Successfully deployed on Streamlit for real-time predictions.

🎯 Key Learnings

• Importance of data cleaning & feature engineering before model training.

• How ANN parameters (layers, neurons, dropout, activation, optimizer) impact model performance.

• Using TensorBoard for model explainability.

• Deployment of ML models using Streamlit.

# Dataset

The dataset consists of customer details including geography, gender, age, balance, credit score, tenure, and exit status.

Target variable: Exited (1 = Churn, 0 = Retained)

Countries included: France, Germany, Spain

# Contributing

Pull requests are welcome :)

# App Deployment

This app is deployed in streamlit, do check it out
link - https://ann-classification-churn-prediction-ahndagqmrttl4grpmsa4hz.streamlit.app/

# License

This project is licensed under the General Public License.
