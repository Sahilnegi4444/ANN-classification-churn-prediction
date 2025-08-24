# Customer Churn & Estimated Salary Prediction using ANN

This project is an end-to-end **Artificial Neural Network (ANN)** implementation for:
- **Customer Churn Prediction (Classification)** → Predicting whether a customer will exit or stay.  
- **Estimated Salary Prediction (Regression)** → Predicting a customer’s estimated salary based on demographic and account features.  

The aim of this project was to understand the working of neural network parameters and build **generalized ANN models** for both classification and regression tasks.

---

## 📌 Project Workflow  

### 1️⃣ Data Cleaning & Feature Engineering  
- Handled missing values and performed data preprocessing.  
- Encoded categorical variables such as Geography (France, Germany, Spain) and Gender.  
- Scaled numerical features for better model convergence.  
- Split the dataset into training and testing sets.  

### 2️⃣ Building Deep Neural Networks  
- Implemented two separate ANN models:  
  - **Churn Prediction ANN (Classification)**  
  - **Salary Prediction ANN (Regression)**  
- Used ~3000 parameters for training churn model.  
- Added **Dense layers, Dropout, L2 Regularization, and Activation functions** to prevent overfitting.  
- Optimized hyperparameters for better generalization.  

### 3️⃣ Model Training & Monitoring  
- Trained the models on the prepared dataset.  
- Used **TensorBoard** to visualize:  
  - Training vs Validation Accuracy (churn model)  
  - Training vs Validation Loss (both models)  
  - Model learning curves  

### 4️⃣ Deployment with Streamlit  
Built a **Streamlit web application** to make predictions:  
- For **Customer Churn** → Takes user inputs (age, geography, balance, etc.) and predicts churn probability.  
- For **Estimated Salary** → Predicts customer’s estimated salary based on demographic details.  

---

## 🛠️ Tech Stack  
- **Python**  
- **Pandas, NumPy** → Data Preprocessing & Feature Engineering
- **Scikit- learn** →  train_test_spliting
- **TensorFlow / Keras** → Deep Neural Networks (Classification & Regression)  
- **TensorBoard** → Model Monitoring  
- **Streamlit** → Deployment  

---

## 📂 Project Structure  

```bash
📦 customer-churn-salary-ann
├── 📂 Churn_Modelling/           # Dataset (raw/processed)
├── 📂 experiments/               # Jupyter notebooks for EDA & model building
│   ├── churn_prediction.ipynb    # ANN model for churn classification
│   └── salary_prediction.ipynb   # ANN model for salary regression
├── 📂 model/                     
│   ├── churn_model.h5            # Saved churn model weights
│   └── salary_model.h5           # Saved salary model weights
├── 📂 app/                       
│   └── streamlit_app.py          # Streamlit application file
├── 📂 logs/                      # TensorBoard logs
├── 📜 requirements.txt           # Dependencies
├── 📜 README.md                  # Project documentation
├── 📜 label_encoder_gender.py    # Converts Gender into numerical
└── 📜 onehot_encoder_geo.py      # One-hot encoding for Geography
```

## 📈 Results

Churn Model (Classification):

Achieved good accuracy and generalization.

TensorBoard visualizations guided hyperparameter tuning.

Salary Model (Regression):

Predicted estimated salary with low error rates.

Evaluated using MSE (Mean Squared Error) and R² Score.

Both models were successfully deployed in Streamlit for real-time predictions.

# Key Learnings

Importance of data cleaning & feature engineering before model training.

How ANN parameters (layers, neurons, dropout, activation, optimizer) impact performance.

Difference between classification and regression ANN models.

Using TensorBoard for model explainability.

Deployment of multiple ML models in a single Streamlit app.

# 📊 Dataset

The dataset consists of customer details including geography, gender, age, balance, credit score, tenure, estimated salary, and exit status.

Target variables:

Exited → Binary (1 = Churn, 0 = Retained)

EstimatedSalary → Continuous (salary value in currency units)

Countries included: France, Germany, Spain

# Contributing

Pull requests are welcome! 😊

# App Deployment

This app is deployed on Streamlit, check it out here: https://ann-classification-churn-prediction-ahndagqmrttl4grpmsa4hz.streamlit.app/
👉 Customer Churn & Salary Prediction App

# License

This project is licensed under the General Public License (GPL).
