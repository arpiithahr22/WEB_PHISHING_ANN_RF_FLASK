# 🔐 Web Phishing Detection Using Machine Learning

A web-based phishing URL detection system that uses **Machine Learning and Deep Learning techniques** to classify URLs as **legitimate or phishing**.

The application is built using **Python and Flask** and integrates a **Random Forest classifier** with a **Hybrid Artificial Neural Network (ANN)** model to provide phishing detection through a simple web interface.

## 🚀 Features

- 🔍 Detect whether a URL is **Phishing** or **Legitimate**
- 🌐 Web-based interface built using Flask
- 🤖 Random Forest machine learning model
- 🧠 Hybrid Artificial Neural Network model
- 📊 Displays prediction results through the web application
- 👤 User registration and login functionality
- 📈 Includes model evaluation results and visualizations
- 💾 SQLite database integration

## 🛠️ Technologies Used

### Programming & Frameworks
- Python
- Flask
- HTML5
- CSS3
- JavaScript

### Machine Learning & Deep Learning
- Random Forest
- Artificial Neural Network (ANN)
- Scikit-learn
- TensorFlow / Keras
- NumPy
- Joblib

### Database
- SQLite
- Flask-SQLAlchemy

### Other Libraries
- BeautifulSoup
- python-whois
- Google Search
- Matplotlib
- Seaborn

## 🧠 Machine Learning Approach

The system extracts different features from a given URL and uses these features to determine whether the URL is potentially malicious.

The project uses two models:

### 1. Random Forest

A Random Forest classifier is used as the primary machine learning model for phishing URL classification.

The trained model is stored in:

```text
saved_models/random_forest_model.pkl
```

### 2. Hybrid ANN Model

An Artificial Neural Network is used together with the extracted URL features and Random Forest prediction information to improve the classification process.

The trained model is stored in:

```text
saved_models/hybrid_ann_model.h5
```

## 📁 Project Structure

```text
WEB_PHISHING_ANN_RF_FLASK/
│
├── app.py
├── feature.py
├── feature_old.py
├── models.py
├── requirements.txt
│
├── saved_models/
│   ├── hybrid_ann_model.h5
│   └── random_forest_model.pkl
│
├── results/
│   ├── accuracy_curve.png
│   ├── classification_report.csv
│   ├── confusion_matrix.png
│   └── loss_curve.png
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── images/
│   │   └── favicon.ico
│   └── js/
│       └── script.js
│
└── templates/
    ├── base.html
    ├── dashboard.html
    ├── index.html
    ├── login.html
    ├── predict.html
    └── register.html
```

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/arpiithahr22/WEB_PHISHING_ANN_RF_FLASK.git
```

### 2. Navigate to the project directory

```bash
cd WEB_PHISHING_ANN_RF_FLASK
cd WEB_PHISHING_ANN_RF_FLASK
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install the required dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The trained Random Forest model was created with a specific Scikit-learn version. If you encounter a model compatibility error while loading `random_forest_model.pkl`, use the Scikit-learn version with which the model was trained.

### 6. Run the Flask application

```bash
python app.py
```

The application will be available at:

```text
http://127.0.0.1:5000
```

## 📊 Model Evaluation

The repository contains evaluation results generated during model training, including:

- Accuracy curve
- Loss curve
- Confusion matrix
- Classification report

These files are available in the `results/` directory.

## 🔄 Application Workflow

```text
User enters URL
       ↓
URL Feature Extraction
       ↓
Random Forest Prediction
       ↓
Hybrid ANN Processing
       ↓
Final Classification
       ↓
Phishing / Legitimate
```

## 🎯 Project Objective

The main objective of this project is to develop a machine-learning-based system capable of identifying potentially malicious phishing URLs and helping users recognize suspicious websites.

## 🔮 Future Enhancements

- Improve model accuracy using larger and more diverse datasets
- Add real-time URL reputation checking
- Integrate additional machine learning models
- Improve the user interface and user experience
- Deploy the application to a cloud platform
- Add browser extension support
- Implement continuous model retraining with newly identified phishing URLs

## 👩‍💻 Author

**Arpitha H R**

Computer Science and Engineering  
Malnad College of Engineering, Hassan

---

⭐ If you find this project useful, consider giving the repository a star!
