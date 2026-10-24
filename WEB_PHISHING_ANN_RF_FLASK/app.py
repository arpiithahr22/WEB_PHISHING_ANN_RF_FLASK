#!/usr/bin/env python
# coding: utf-8

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import numpy as np
import joblib
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
from feature import FeatureExtraction

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# ------------------------------
# Database Models
# ------------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(500), nullable=False)
    prediction = db.Column(db.Integer, nullable=False)  # 0: legitimate, 1: phishing
    probability_phishing = db.Column(db.Float, nullable=False)
    probability_legitimate = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ------------------------------
# ML Model Loading
# ------------------------------
try:
    print("🔄 Loading ML models...")
    RF_PATH = "saved_models/random_forest_model.pkl"
    ANN_PATH = "saved_models/hybrid_ann_model.h5"
    
    rf_model = joblib.load(RF_PATH)
    ann_model = keras.models.load_model(ANN_PATH)
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    rf_model = None
    ann_model = None

# ------------------------------
# Helper Functions
# ------------------------------
def preprocess_url(url: str) -> np.ndarray:
    """Extract features from URL using FeatureExtraction."""

    feature_extractor = FeatureExtraction(url)

    obj = FeatureExtraction(url)
    x = np.array(feature_extractor.getFeaturesList()).reshape(1,30) 
    feature_details = feature_extractor.getFeatureDetails()

    features = feature_extractor.getFeaturesList()
    return np.array(features).reshape(1, -1),feature_details


def make_hybrid_features(raw_features: np.ndarray) -> np.ndarray:
    """Append Random Forest probability to raw features to form hybrid input."""
    try:
        rf_prob = rf_model.predict_proba(raw_features)[:, 1].reshape(-1, 1)
        hybrid = np.concatenate([raw_features, rf_prob], axis=1)
        return hybrid
    except Exception as e:
        print(f"Error in hybrid features: {e}")
        return raw_features

def predict_phishing(url: str, threshold: float = 0.5):
    """
    Predict whether a single URL is phishing or legitimate.
    """
    if rf_model is None or ann_model is None:
        return 0, 0.0, 1.0  # Default to legitimate if models not loaded
    
  
    # Extract and prepare features
    raw_features,feature_details = preprocess_url(url)
    hybrid_features = make_hybrid_features(raw_features)

    # ANN prediction
    ann_pred = ann_model.predict(hybrid_features, verbose=0)
    prob_legit = float(ann_pred.flatten()[0])
    prob_phishing = 1.0 - prob_legit

    pred_label = int(prob_phishing >= threshold)
    return pred_label, prob_phishing, prob_legit,feature_details


def create_probability_plot(prob_phishing, prob_legit):
    """Create a probability comparison plot."""
    plt.figure(figsize=(8, 6))
    
    categories = ['Phishing', 'Legitimate']
    probabilities = [prob_phishing * 100, prob_legit * 100]
    colors = ['#ff6b6b', '#51cf66']
    
    # Create bar plot
    bars = plt.bar(categories, probabilities, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, prob in zip(bars, probabilities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{prob:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylabel('Probability (%)', fontsize=12)
    plt.title('URL Classification Probability', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    
    # Convert plot to base64 for HTML embedding
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return f"data:image/png;base64,{plot_data}"

# ------------------------------
# Routes
# ------------------------------
@app.route('/')
def index():
    """Home page with project information."""
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            next_page = request.args.get('next')
            flash('Login successful!', 'success')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Login failed. Check username and password.', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return render_template('register.html')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered!', 'danger')
            return render_template('register.html')
        
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password_hash=hashed_password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with prediction statistics."""
    # Get user's predictions
    user_predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    
    # Calculate statistics
    total_predictions = len(user_predictions)
    phishing_count = sum(1 for p in user_predictions if p.prediction == 1)
    legit_count = total_predictions - phishing_count
    
    # Recent predictions (last 5)
    recent_predictions = user_predictions[:5]
    
    return render_template('dashboard.html', 
                         predictions=recent_predictions,
                         total_predictions=total_predictions,
                         phishing_count=phishing_count,
                         legit_count=legit_count)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    """URL prediction page."""
    prediction_result = None
    plot_url = None
    feature_details = None
    
    if request.method == 'POST':
        url = request.form.get('url')
        
        if url:
            # Make prediction
            label, prob_phishing, prob_legit,feature_details = predict_phishing(url)
            
            # Save prediction to database
            new_prediction = Prediction(
                url=url,
                prediction=label,
                probability_phishing=prob_phishing,
                probability_legitimate=prob_legit,
                user_id=current_user.id
            )
            db.session.add(new_prediction)
            db.session.commit()
            
            # Create visualization
            plot_url = create_probability_plot(prob_phishing, prob_legit)
            
            prediction_result = {
                'url': url,
                'label': label,
                'prob_phishing': prob_phishing,
                'prob_legit': prob_legit,
                'is_phishing': label == 1
            }
            
            flash('Prediction completed!', 'success')
    
    return render_template('predict.html', 
                         prediction_result=prediction_result,
                         plot_url=plot_url,feature_details=feature_details)

@app.route('/predict_ajax', methods=['POST'])
@login_required
def predict_ajax():
    """AJAX endpoint for predictions."""
    data = request.get_json()
    url = data.get('url')
    
    if url:
        label,prob_legit, prob_phishing = predict_phishing(url)
        
        # Save prediction
        new_prediction = Prediction(
            url=url,
            prediction=label,
            probability_phishing=prob_phishing,
            probability_legitimate=prob_legit,
            user_id=current_user.id
        )
        db.session.add(new_prediction)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'prediction': label,
            'prob_phishing': round(prob_phishing * 100, 2),
            'prob_legit': round(prob_legit * 100, 2),
            'is_phishing': label == 0
        })
    
    return jsonify({'success': False, 'error': 'No URL provided'})

@app.route('/logout')
@login_required
def logout():
    """User logout."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

# ------------------------------
# Error Handlers
# ------------------------------
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

# ------------------------------
# Application Startup
# ------------------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)