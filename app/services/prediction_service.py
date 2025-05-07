import joblib
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import logging
import traceback
import os

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from config import Config

class PredictionService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self._load_model()

    def _load_model(self):
        """Load the trained model and related data"""
        try:
            self.logger.info("Initializing PredictionService")
            model_path = Path(__file__).parent.parent / 'models' / 'xgboost_model.pkl'
            self.logger.info(f"Attempting to load model from: {model_path}")
            
            if not model_path.exists():
                self.logger.error("Model file not found")
                raise FileNotFoundError("Model file not found")
                
            self.logger.info(f"Model file exists: {model_path.exists()}")
            self.logger.info(f"Model file size: {model_path.stat().st_size} bytes")
            
            # Model yükleme işlemini optimize et
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            
            self.logger.info("Model data loaded successfully")
            self.logger.info(f"Model data keys: {model_data.keys()}")
            self.logger.info(f"Model type: {type(self.model)}")
            
            # Model parametrelerini logla
            self.logger.info(f"Model parameters: {self.model.get_params()}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def predict_side_effects(self, patient_data):
        """Predict side effects for given medications"""
        try:
            self.logger.info("Starting prediction process")
            self.logger.info(f"Input patient data: {patient_data}")
            
            if not self.model:
                self.logger.error("Model not loaded")
                raise ValueError("Model not loaded")

            # Validate required fields
            required_fields = ['age', 'weight', 'height', 'sex', 'medications', 'serious']
            missing_fields = [field for field in required_fields if field not in patient_data]
            if missing_fields:
                self.logger.error(f"Missing required fields: {missing_fields}")
                raise ValueError(f"Missing required fields: {missing_fields}")

            # Preprocess data
            processed_data = self._preprocess_data(patient_data)
            self.logger.info(f"Processed data shape: {processed_data.shape}")
            self.logger.info(f"Processed data columns: {processed_data.columns.tolist()}")

            # Make predictions
            predictions = []
            medications = patient_data.get('medications', [])
            
            if not medications:
                self.logger.error("No medications provided")
                raise ValueError("No medications provided")

            for medication in medications:
                try:
                    self.logger.info(f"Processing medication: {medication}")
                    
                    # Add medication-specific features
                    medication_data = processed_data.copy()
                    
                    # Get prediction and probability
                    prediction = self.model.predict(medication_data)[0]
                    probability = self.model.predict_proba(medication_data)[0][1]
                    
                    self.logger.info(f"Raw prediction: {prediction}")
                    self.logger.info(f"Raw probability: {probability}")
                    
                    # Format results
                    result = self._format_prediction_results(medication, prediction, probability)
                    self.logger.info(f"Formatted result: {result}")
                    
                    predictions.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Error processing medication {medication}: {str(e)}")
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    predictions.append({
                        'medication': medication,
                        'risk_level': 'Unknown',
                        'probability': 0.0,
                        'recommendation': f"Unable to generate prediction due to error: {str(e)}"
                    })

            if not predictions:
                self.logger.error("No predictions generated")
                raise ValueError("No predictions generated")

            self.logger.info(f"Generated {len(predictions)} predictions")
            return {'results': predictions}

        except Exception as e:
            self.logger.error(f"Error in predict_side_effects: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _preprocess_data(self, patient_data):
        """Hasta verilerini model için hazırla"""
        try:
            self.logger.info("Starting data preprocessing")
            
            # Temel özellikleri oluştur
            features = pd.DataFrame({
                'age': [float(patient_data['age'])],
                'weight': [float(patient_data['weight'])],
                'sex': [int(patient_data['sex'])],
                'serious': [int(patient_data['serious'])],
                'bmi': [float(patient_data['weight']) / ((float(patient_data['height'])/100) ** 2)]
            })
            
            self.logger.info(f"Created base features DataFrame with shape: {features.shape}")
            self.logger.info(f"Base features columns: {features.columns.tolist()}")
            
            # Tıbbi geçmiş özelliklerini ekle
            medical_conditions = {
                'heart_disease': ['heart disease', 'cardiac', 'myocardial', 'coronary', 'angina', 'arrhythmia'],
                'asthma': ['asthma', 'bronchial', 'respiratory'],
                'cancer': ['cancer', 'tumor', 'neoplasm', 'malignancy'],
                'diabetes': ['diabetes', 'diabetic', 'hyperglycemia'],
                'hypertension': ['hypertension', 'high blood pressure'],
                'kidney_disease': ['kidney disease', 'renal', 'nephropathy'],
                'liver_disease': ['liver disease', 'hepatic', 'cirrhosis'],
                'thyroid_disease': ['thyroid', 'hypothyroidism', 'hyperthyroidism'],
                'arthritis': ['arthritis', 'rheumatoid', 'osteoarthritis'],
                'depression': ['depression', 'anxiety', 'mental health'],
                'allergies': ['allergy', 'allergic', 'hypersensitivity'],
                'gastrointestinal': ['gastrointestinal', 'stomach', 'intestinal', 'ulcer'],
                'neurological': ['neurological', 'brain', 'nervous system', 'epilepsy'],
                'respiratory': ['respiratory', 'lung', 'pneumonia', 'bronchitis'],
                'immune_system': ['immune system', 'immunodeficiency', 'autoimmune']
            }
            
            medical_history = patient_data.get('medical_history', [])
            for condition, keywords in medical_conditions.items():
                features[f'has_{condition}'] = [1 if any(keyword in ' '.join(medical_history).lower() for keyword in keywords) else 0]
            
            self.logger.info(f"Added medical history features. Total columns: {features.columns.tolist()}")
            
            # Laboratuvar test özelliklerini ekle
            lab_tests = {
                'BloodPressure': {'normal_range': (90, 120), 'unit': 'mmHg'},
                'BloodSugar': {'normal_range': (70, 100), 'unit': 'mg/dL'},
                'Cholesterol': {'normal_range': (150, 200), 'unit': 'mg/dL'},
                'Hemoglobin': {'normal_range': (12, 16), 'unit': 'g/dL'},
                'WhiteBloodCell': {'normal_range': (4000, 11000), 'unit': 'cells/µL'},
                'Platelets': {'normal_range': (150000, 450000), 'unit': 'cells/µL'},
                'Creatinine': {'normal_range': (0.7, 1.3), 'unit': 'mg/dL'},
                'ALT': {'normal_range': (7, 56), 'unit': 'U/L'},
                'AST': {'normal_range': (10, 40), 'unit': 'U/L'},
                'Potassium': {'normal_range': (3.5, 5.0), 'unit': 'mEq/L'},
                'Sodium': {'normal_range': (135, 145), 'unit': 'mEq/L'},
                'Calcium': {'normal_range': (8.5, 10.2), 'unit': 'mg/dL'},
                'Magnesium': {'normal_range': (1.7, 2.2), 'unit': 'mg/dL'},
                'Phosphorus': {'normal_range': (2.5, 4.5), 'unit': 'mg/dL'},
                'Bilirubin': {'normal_range': (0.3, 1.2), 'unit': 'mg/dL'},
                'Albumin': {'normal_range': (3.5, 5.0), 'unit': 'g/dL'},
                'UricAcid': {'normal_range': (3.4, 7.0), 'unit': 'mg/dL'},
                'TSH': {'normal_range': (0.4, 4.0), 'unit': 'µIU/mL'},
                'VitaminD': {'normal_range': (30, 100), 'unit': 'ng/mL'}
            }
            
            laboratory_tests = patient_data.get('laboratory_tests', [])
            for test_name, params in lab_tests.items():
                test_value = None
                for test in laboratory_tests:
                    if test.get('testname', '').lower() == test_name.lower():
                        try:
                            test_value = float(test.get('testresult', '0').split('/')[0])
                        except (ValueError, IndexError):
                            continue
                        break
                
                features[f'{test_name}_exists'] = [1 if test_value is not None else 0]
                features[f'{test_name}_value'] = [test_value if test_value is not None else None]
                features[f'{test_name}_normal'] = [1 if test_value is not None and params['normal_range'][0] <= test_value <= params['normal_range'][1] else 0]
            
            self.logger.info(f"Added laboratory test features. Total columns: {features.columns.tolist()}")
            
            # İlaç etkileşimlerini ekle
            medications = patient_data.get('medications', [])
            features['interaction_blood_thinner'] = [1 if all(drug in medications for drug in ['aspirin', 'ibuprofen']) else 0]
            features['interaction_blood_sugar'] = [1 if all(drug in medications for drug in ['metformin', 'insulin']) else 0]
            
            self.logger.info(f"Added drug interaction features. Total columns: {features.columns.tolist()}")
            
            # Eksik değerleri doldur
            features = features.fillna(0)
            
            # Özellikleri ölçeklendir
            if self.scaler is not None:
                features = pd.DataFrame(
                    self.scaler.transform(features),
                    columns=features.columns
                )
            
            self.logger.info(f"Final features DataFrame shape: {features.shape}")
            self.logger.info(f"Final features columns: {features.columns.tolist()}")
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error in data preprocessing: {str(e)}")
            raise

    def _format_prediction_results(self, medication, prediction, probability):
        """Format prediction results"""
        try:
            self.logger.info(f"Formatting results for {medication}")
            self.logger.info(f"Raw prediction: {prediction}, probability: {probability}")
            
            # Convert NumPy types to Python native types
            prediction = int(prediction)
            probability = float(probability)
            
            # Determine risk level
            if probability < 0.3:
                risk_level = 'Low'
            elif probability < 0.7:
                risk_level = 'Medium'
            else:
                risk_level = 'High'
            
            # İlaç alternatifleri
            alternatives = {
                'aspirin': {
                    'low': 'Consider using acetaminophen (Tylenol) for pain relief',
                    'medium': 'Try acetaminophen (Tylenol) or naproxen (Aleve)',
                    'high': 'Use acetaminophen (Tylenol) instead. If blood thinning is needed, consult your doctor about other options'
                },
                'metformin': {
                    'low': 'Continue with metformin, monitor blood sugar regularly',
                    'medium': 'Consider adding lifestyle changes or consult about other diabetes medications',
                    'high': 'Discuss alternative diabetes medications with your doctor, such as sulfonylureas or DPP-4 inhibitors'
                },
                'lisinopril': {
                    'low': 'Continue with lisinopril, monitor blood pressure',
                    'medium': 'Consider adding lifestyle changes or consult about other blood pressure medications',
                    'high': 'Discuss alternative blood pressure medications with your doctor, such as calcium channel blockers or beta blockers'
                },
                'ibuprofen': {
                    'low': 'Continue with ibuprofen, monitor for side effects',
                    'medium': 'Consider using acetaminophen (Tylenol) or naproxen (Aleve)',
                    'high': 'Use acetaminophen (Tylenol) instead. If anti-inflammatory is needed, consult your doctor'
                },
                'atorvastatin': {
                    'low': 'Continue with atorvastatin, monitor cholesterol levels',
                    'medium': 'Consider adding lifestyle changes or consult about other statins',
                    'high': 'Discuss alternative cholesterol medications with your doctor, such as other statins or PCSK9 inhibitors'
                }
            }
            
            # Generate recommendation with alternatives
            medication_lower = medication.lower()
            if medication_lower in alternatives:
                recommendation = f"{medication} appears to be safe for use. Monitor for any side effects." if risk_level == 'Low' else \
                               f"Use {medication} with caution. Monitor closely for side effects." if risk_level == 'Medium' else \
                               f"High risk with {medication}. {alternatives[medication_lower][risk_level]}"
            else:
                recommendation = f"{medication} appears to be safe for use. Monitor for any side effects." if risk_level == 'Low' else \
                               f"Use {medication} with caution. Monitor closely for side effects." if risk_level == 'Medium' else \
                               f"High risk with {medication}. Consider alternative medications."
            
            result = {
                'medication': medication,
                'risk_level': risk_level,
                'probability': probability,
                'recommendation': recommendation
            }
            
            self.logger.info(f"Formatted result: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in _format_prediction_results: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _get_alternative_recommendations(self, drug_name, risk_level, probability):
        """
        İlaç için alternatif önerileri oluştur
        """
        # İlaç kategorilerine göre alternatifler
        alternatives = {
            'aspirin': {
                'low': ['Acetaminophen (Paracetamol)', 'Naproxen'],
                'medium': ['Acetaminophen (Paracetamol)', 'Topical NSAIDs'],
                'high': ['Acetaminophen (Paracetamol)', 'Physical therapy', 'Lifestyle modifications']
            },
            'ibuprofen': {
                'low': ['Acetaminophen (Paracetamol)', 'Naproxen'],
                'medium': ['Acetaminophen (Paracetamol)', 'Topical NSAIDs'],
                'high': ['Acetaminophen (Paracetamol)', 'Physical therapy', 'Lifestyle modifications']
            },
            'metformin': {
                'low': ['Lifestyle modifications', 'Diet changes'],
                'medium': ['Sitagliptin', 'Saxagliptin'],
                'high': ['Lifestyle modifications', 'Diet changes', 'Exercise program']
            },
            'insulin': {
                'low': ['Oral diabetes medications', 'Lifestyle modifications'],
                'medium': ['Sitagliptin', 'Saxagliptin', 'Dapagliflozin'],
                'high': ['Lifestyle modifications', 'Diet changes', 'Exercise program']
            },
            'lisinopril': {
                'low': ['Amlodipine', 'Losartan'],
                'medium': ['Valsartan', 'Ramipril'],
                'high': ['Lifestyle modifications', 'Diet changes', 'Exercise program']
            },
            'atorvastatin': {
                'low': ['Rosuvastatin', 'Pravastatin'],
                'medium': ['Simvastatin', 'Lovastatin'],
                'high': ['Lifestyle modifications', 'Diet changes', 'Exercise program']
            },
            'default': {
                'low': ['Consult your doctor for alternatives'],
                'medium': ['Consult your doctor for alternatives', 'Consider lifestyle modifications'],
                'high': ['Consult your doctor for alternatives', 'Consider lifestyle modifications', 'Seek specialist advice']
            }
        }

        # İlaç adını küçük harfe çevir
        drug_name = drug_name.lower()
        
        # İlaç için alternatifleri bul
        drug_alternatives = alternatives.get(drug_name, alternatives['default'])
        
        # Risk seviyesine göre önerileri al
        recommendations = drug_alternatives.get(risk_level.lower(), drug_alternatives['medium'])
        
        # Önerileri formatla
        formatted_recommendations = []
        for rec in recommendations:
            if probability > 0.7:
                formatted_recommendations.append(f"Strongly consider: {rec}")
            elif probability > 0.5:
                formatted_recommendations.append(f"Consider: {rec}")
            else:
                formatted_recommendations.append(rec)
        
        return formatted_recommendations

    def _get_recommendation(self, probability):
        """
        Generate recommendation based on prediction probability
        """
        if probability >= 0.7:
            return "Consider alternative medication. High risk of severe side effects."
        elif probability >= 0.4:
            return "Monitor patient closely. Moderate risk of side effects."
        else:
            return "Safe to proceed with prescribed medication. Low risk of side effects." 