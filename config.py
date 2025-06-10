import os
from pathlib import Path

class Config:
    # OpenFDA API Configuration
    OPENFDA_API_KEY = "9P6YoFmN3cYo0OTAdoi1Yia1GpcYyPpcgvQRkYTr"
    
    # Model Configuration
    MODEL_PATH = Path(__file__).parent / 'app' / 'models' / 'xgboost_model.pkl'
    
    # Training Configuration
    MAX_TRAINING_SAMPLES = 1000
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Database Configuration
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + str(Path(__file__).parent / 'app' / 'database.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Feature Configuration
    REQUIRED_FEATURES = [
        'patient_age',
        'patient_sex',
        'weight',
        'height'
    ]
    
    # Medical Conditions
    MEDICAL_CONDITIONS = [
        'Hypertension',
        'Diabetes',
        'HeartDisease',
        'Asthma',
        'Cancer'
    ]
    
    # Laboratory Tests
    LAB_TESTS = {
        'BloodPressure': {'normal_range': (90, 120)},
        'BloodSugar': {'normal_range': (70, 100)},
        'Cholesterol': {'normal_range': (150, 200)},
        'Hemoglobin': {'normal_range': (12, 16)},
        'WhiteBloodCell': {'normal_range': (4000, 11000)},
        'Platelets': {'normal_range': (150000, 450000)},
        'Creatinine': {'normal_range': (0.7, 1.3)},
        'ALT': {'normal_range': (7, 56)},
        'AST': {'normal_range': (10, 40)}
    }
    
    # Drug Interactions
    RISKY_COMBINATIONS = {
        ('aspirin', 'ibuprofen'): 'blood_thinner',
        ('metformin', 'insulin'): 'blood_sugar',
        ('warfarin', 'aspirin'): 'blood_thinner',
        ('lisinopril', 'spironolactone'): 'blood_pressure',
        ('simvastatin', 'amiodarone'): 'muscle_damage'
    }

    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    FAERS_API_BASE_URL = 'https://api.fda.gov/drug/event.json'
    FAERS_API_KEY = os.environ.get('FAERS_API_KEY') 