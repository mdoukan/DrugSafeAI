import joblib
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import logging
import traceback
import os
import time
from datetime import datetime

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
        self.feature_names = None
        self.last_update_time = None
        self.drug_list = []
        self._load_model()

    def check_model_update_needed(self):
        """Modelin güncellenmesi gerekip gerekmediğini kontrol et"""
        if not self.last_update_time:
            self.logger.info("Model needs update: No last update time available")
            return False  # Update only on demand, not automatically
            
        # 7 gün geçtiyse güncelleme zamanı
        update_interval = 7 * 24 * 60 * 60  # 7 gün (saniye olarak)
        current_time = time.time()
        
        if (current_time - self.last_update_time) > update_interval:
            days_since_update = (current_time - self.last_update_time) / (24 * 60 * 60)
            self.logger.info(f"Model needs update: Last update was {days_since_update:.1f} days ago")
            return False  # Only update manually, not during prediction
            
        return False  # Always return False to prevent automatic updates during prediction
        
    def update_model_if_needed(self):
        """Gerekirse modeli güncelle"""
        if self.check_model_update_needed():
            self.logger.info("Initiating model update process")
            try:
                from app.models.train_model import ModelTrainer
                from config import Config
                
                # ModelTrainer oluştur
                trainer = ModelTrainer(api_key=Config.OPENFDA_API_KEY)
                
                # Mevcut ilaç listesini kullan veya varsayılanı al
                drug_list = self.drug_list
                if not drug_list:
                    drug_list = [
                        'aspirin', 'ibuprofen', 'amoxicillin', 'metformin',
                        'lisinopril', 'atorvastatin', 'fluoxetine', 'omeprazole'
                    ]
                
                # Modeli güncelle
                updated, result = trainer.check_and_update_model()
                
                if updated:
                    self.logger.info("Model was updated. Reloading model...")
                    self._load_model()
                    return True, "Model successfully updated"
                else:
                    self.logger.info("No model update was needed or update failed")
                    return False, "No update required or update failed"
                    
            except Exception as e:
                self.logger.error(f"Error during model update: {str(e)}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                return False, f"Update error: {str(e)}"
        else:
            return False, "No update needed"

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
            self.feature_names = model_data.get('feature_names', None)  # Feature isimlerini yükle
            self.last_update_time = model_data.get('last_update_time', None)  # Son güncelleme zamanını yükle
            self.drug_list = model_data.get('drug_list', [])  # İlaç listesini yükle
            
            self.logger.info("Model data loaded successfully")
            self.logger.info(f"Model data keys: {model_data.keys()}")
            self.logger.info(f"Model type: {type(self.model)}")
            
            if self.last_update_time:
                last_update_str = datetime.fromtimestamp(self.last_update_time).strftime('%Y-%m-%d %H:%M:%S')
                self.logger.info(f"Model last update time: {last_update_str}")
            
            if self.feature_names:
                self.logger.info(f"Model feature names: {self.feature_names}")
                
            if self.drug_list:
                self.logger.info(f"Model drug list: {self.drug_list}")
            
            # Model parametrelerini logla
            self.logger.info(f"Model parameters: {self.model.get_params() if hasattr(self.model, 'get_params') else 'No parameters'}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def predict_side_effects(self, patient_data):
        """
        Hasta verilerine göre ilaç yan etkilerini tahmin eder
        
        Args:
            patient_data: İlaç yan etkileri tahmini için hasta verileri
            
        Returns:
            Dict: Tahmin sonuçları
        """
        try:
            # Model güncelleme kontrolünü atla - performans için
            # Bu işlem manuel olarak yapılmalı, her tahmin isteğinde değil
            
            # Veriyi önişleme
            processed_data = self._preprocess_data(patient_data)
            if processed_data is None or processed_data.empty:
                raise ValueError("Data preprocessing failed")

            # İlaç listesini al
            medications = patient_data.get('medications', [])
            if not medications:
                raise ValueError("No medications provided")

            # İlaç etkileşimlerini kontrol et
            interactions = self._check_drug_interactions(medications, patient_data)
            
            # Her bir ilaç için yan etkileri tahmin et
            results = []
            for medication in medications:
                # Vektörü oluştur
                feature_vector = self._create_feature_vector(processed_data)
                
                # Model ile tahmin yap
                prediction = self.model.predict(feature_vector)[0]
                if hasattr(self.model, "predict_proba"):
                    probability = self.model.predict_proba(feature_vector)[0][1]
                else:
                    probability = 0.5
                
                # Sonuçları formatla
                formatted_result = self._format_prediction_results(medication, prediction, probability)
                
                # Generate side effects for visualization
                formatted_result['side_effects'] = self._generate_sample_side_effects(medication, probability, formatted_result['risk_level'])
                
                results.append(formatted_result)
                
            # İlaç-koşul etkileşimlerini kontrol et
            if 'medical_history' in patient_data:
                results = self._adjust_for_medical_conditions(results, patient_data.get('medical_history', []), patient_data)
                
            # Laboratuvar değerlerine göre düzelt
            if 'laboratory_tests' in patient_data:
                results = self._adjust_for_lab_values(results, patient_data.get('laboratory_tests', []), patient_data)
                
            # Yaşa göre düzelt
            if 'age' in patient_data:
                results = self._adjust_for_age(results, patient_data.get('age', 0))
            
            # En riskli ilacı ve genel risk seviyesini belirle
            highest_risk = 0.0
            riskiest_medication = None
            
            for result in results:
                if float(result['probability']) > highest_risk:
                    highest_risk = float(result['probability'])
                    riskiest_medication = result
            
            # FAERS servisinden yan etkileri al ve katapısal olarak grupla
            all_side_effects = []
            for result in results:
                side_effects = result.get('side_effects', [])
                for effect in side_effects:
                    # İlaç bilgisini yan etkiye ekle
                    effect['medication'] = result['medication']
                    all_side_effects.append(effect)
            
            # İlaç bilgilerini oluştur
            medication_info = []
            for med in medications:
                category = self._get_medication_category(med)
                medication_info.append({
                    'name': med,
                    'category': category
                })
            
            # İlaç etkileşimlerini format
            formatted_interactions = []
            for interaction in interactions:
                if 'drugs' in interaction and len(interaction['drugs']) >= 2:
                    formatted_interactions.append({
                        'source': interaction['drugs'][0],
                        'target': interaction['drugs'][1],
                        'severity': interaction['severity'].lower(),
                        'description': interaction['description']
                    })
            
            # Sonuçları döndür
            return {
                'status': 'success',
                'data': {
                    'results': results,
                    'medications': medication_info,
                    'side_effects': all_side_effects,
                    'drug_interactions': formatted_interactions,
                    'highest_risk': str(highest_risk),
                    'riskiest_medication': riskiest_medication
                }
            }

        except Exception as e:
            self.logger.error(f"Error in predict_side_effects: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _preprocess_data(self, patient_data):
        """Hasta verilerini model için hazırla"""
        try:
            self.logger.info("Starting data preprocessing")
            
            # Veri anahtarlarını eğitimdeki ile eşleştir
            mapped_data = patient_data.copy()
            
            # Feature isimlerini eğitimdekine uygun olarak dönüştür
            if 'age' in mapped_data:
                mapped_data['patient_age'] = mapped_data.pop('age')
            if 'sex' in mapped_data:
                mapped_data['patient_sex'] = mapped_data.pop('sex')
                
            self.logger.info(f"Mapped data keys: {list(mapped_data.keys())}")
                
            # Temel özellikleri oluştur
            features = pd.DataFrame({
                'patient_age': [float(mapped_data['patient_age'])],
                'weight': [float(mapped_data['weight'])],
                'patient_sex': [int(mapped_data['patient_sex'])],
                'serious': [int(mapped_data['serious'])]
            })
            
            # Yaş grupları - ModelTrainer'dakiyle aynı
            age = float(mapped_data['patient_age'])
            age_bins = [0, 12, 18, 30, 45, 65, 80, 150]
            age_labels = ['child', 'teen', 'young_adult', 'adult', 'middle_aged', 'senior', 'elderly']
            
            # Age_group değişkenini oluştur ve sonra one-hot encoding yap
            age_group = None
            for i in range(len(age_bins)-1):
                if age_bins[i] <= age < age_bins[i+1]:
                    age_group = age_labels[i]
                    break
            
            # Manuel olarak tüm age_group_* özelliklerini oluştur ve başlangıçta 0 olarak ayarla
            for label in age_labels:
                features[f'age_group_{label}'] = [0]
            
            # Eğer bir yaş grubu belirlediyse, o sütunu 1 olarak işaretle
            if age_group:
                features[f'age_group_{age_group}'] = [1]
            
            # BMI hesaplama
            if 'height' in mapped_data and 'weight' in mapped_data:
                height = float(mapped_data['height'])
                weight = float(mapped_data['weight'])
                bmi = weight / ((height/100) ** 2)
                features['bmi'] = [bmi]
                
                # BMI kategorileri - ModelTrainer'dakiyle aynı
                bmi_bins = [0, 18.5, 25, 30, 35, 100]
                bmi_labels = ['underweight', 'normal', 'overweight', 'obese', 'extremely_obese']
                
                # BMI kategori değişkenini oluştur
                bmi_category = None
                for i in range(len(bmi_bins)-1):
                    if bmi_bins[i] <= bmi < bmi_bins[i+1]:
                        bmi_category = bmi_labels[i]
                        break
                
                # Manuel olarak tüm bmi_* özelliklerini oluştur ve başlangıçta 0 olarak ayarla
                for label in bmi_labels:
                    features[f'bmi_{label}'] = [0]
                
                # Eğer bir BMI kategorisi belirlediyse, o sütunu 1 olarak işaretle
                if bmi_category:
                    features[f'bmi_{bmi_category}'] = [1]
            
            # İlaç özelinde özellikler ekle - her ilaç için ayrı bir kolon
            medications = mapped_data.get('medications', [])
            # Eğitimde kullanılan tüm ilaçlar için sütunlar oluştur ve 0'la doldur
            all_drugs = ['aspirin', 'ibuprofen', 'amoxicillin', 'metformin', 'lisinopril', 'atorvastatin', 'fluoxetine', 'omeprazole']
            for drug in all_drugs:
                features[f'drug_{drug}'] = [1 if drug in medications else 0]
            
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
            
            medical_history = mapped_data.get('medical_history', [])
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
                'Calcium': {'normal_range': (8.5, 10.5), 'unit': 'mg/dL'},
                'Magnesium': {'normal_range': (1.7, 2.2), 'unit': 'mg/dL'},
                'Phosphorus': {'normal_range': (2.5, 4.5), 'unit': 'mg/dL'},
                'Bilirubin': {'normal_range': (0.1, 1.2), 'unit': 'mg/dL'},
                'Albumin': {'normal_range': (3.4, 5.4), 'unit': 'g/dL'},
                'UricAcid': {'normal_range': (3.5, 7.2), 'unit': 'mg/dL'},
                'TSH': {'normal_range': (0.4, 4.0), 'unit': 'mIU/L'},
                'VitaminD': {'normal_range': (20, 50), 'unit': 'ng/mL'}
            }
            
            laboratory_tests = mapped_data.get('laboratory_tests', [])
            for lab_name, lab_info in lab_tests.items():
                # Laboratuvar testi var mı?
                features[f'{lab_name}_exists'] = [0]
                # Laboratuvar test değeri normal aralıkta mı?
                features[f'{lab_name}_normal'] = [0]
                # Laboratuvar test değeri
                features[f'{lab_name}_value'] = [0.0]
                
                for lab_test in laboratory_tests:
                    if lab_name.lower() in lab_test.get('name', '').lower():
                        features[f'{lab_name}_exists'] = [1]
                        value = lab_test.get('value', 0)
                        try:
                            value = float(value)
                            features[f'{lab_name}_value'] = [value]
                            
                            # Değer normal aralıkta mı?
                            min_val, max_val = lab_info['normal_range']
                            features[f'{lab_name}_normal'] = [1 if min_val <= value <= max_val else 0]
                        except (ValueError, TypeError):
                            pass
            
            # İlaç etkileşimlerini ekle
            risky_combinations = {
                ('aspirin', 'ibuprofen'): 'blood_thinner',
                ('metformin', 'insulin'): 'blood_sugar',
            }
            
            for drug_pair, risk_type in risky_combinations.items():
                features[f'interaction_{risk_type}'] = [1 if all(drug in medications for drug in drug_pair) else 0]
            
            # Eğer model yüklendiyse ve feature_names özelliği varsa, feature_names'deki tüm özelliklerin
            # features DataFrame'inde olup olmadığını kontrol et
            if self.feature_names:
                self.logger.info("Checking if all model features exist in the processed data")
                missing_features = [feat for feat in self.feature_names if feat not in features.columns]
                extra_features = [feat for feat in features.columns if feat not in self.feature_names]
                
                if missing_features:
                    self.logger.warning(f"Missing features: {missing_features}")
                    for feat in missing_features:
                        features[feat] = [0]  # Eksik özellikleri varsayılan 0 ile ekle
                
                if extra_features:
                    self.logger.warning(f"Extra features not in model: {extra_features}")
                    features = features.drop(columns=extra_features)  # Fazla özellikleri kaldır
                    
                # Feature sıralamasını model feature_names'ine göre düzenle
                features = features[self.feature_names]
            
            # Eksik değerleri 0 ile doldur
            features = features.fillna(0)
            
            self.logger.info(f"Final processed data shape: {features.shape}")
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error during data preprocessing: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _format_prediction_results(self, medication, prediction, probability):
        """Format prediction results"""
        try:
            self.logger.info(f"Formatting results for {medication}")
            self.logger.info(f"Raw prediction: {prediction}, probability: {probability}")
            
            # Convert NumPy types to Python native types
            prediction = int(prediction)
            probability = float(probability)
            
            # Determine risk level - Updated thresholds for more balanced risk classification
            if probability < 0.40:
                risk_level = 'Low'
            elif probability < 0.60:
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
                },
                'amoxicillin': {
                    'low': 'Continue with amoxicillin as prescribed',
                    'medium': 'Consider alternative antibiotics like azithromycin or doxycycline',
                    'high': 'Switch to alternative antibiotics like cephalexin or ciprofloxacin after consulting your doctor'
                },
                'fluoxetine': {
                    'low': 'Continue with fluoxetine as prescribed',
                    'medium': 'Consider other SSRIs like sertraline or escitalopram',
                    'high': 'Discuss alternative antidepressants like bupropion or mirtazapine with your doctor'
                },
                'omeprazole': {
                    'low': 'Continue with omeprazole as prescribed',
                    'medium': 'Consider other PPIs like pantoprazole or H2 blockers like famotidine',
                    'high': 'Discuss alternative reflux treatments like H2 blockers or lifestyle modifications with your doctor'
                }
            }
            
            # Generate recommendation with alternatives
            medication_lower = medication.lower()
            
            # Determine alternative medication recommendations
            alternative_info = ""
            if medication_lower in alternatives:
                risk_level_lower = risk_level.lower()
                alternative_info = f" {alternatives[medication_lower][risk_level_lower]}"
            
            # Format the recommendation
            if risk_level == 'Low':
                recommendation = f"{medication} appears to be safe for use. Monitor for any side effects."
            elif risk_level == 'Medium':
                recommendation = f"Use {medication} with caution. Monitor closely for side effects.{alternative_info}"
            else:
                recommendation = f"High risk with {medication}.{alternative_info}"
            
            # İlaç kategorisini belirle
            medication_category = self._get_medication_category(medication)
            
            # Get alternative medications through the helper method
            alternative_meds = self._get_alternative_recommendations(medication, risk_level, probability)
            
            # Generate sample side effects for visualization
            side_effects = self._generate_sample_side_effects(medication, probability, risk_level)
            
            result = {
                'medication': medication,
                'category': medication_category,
                'risk_level': risk_level,
                'probability': probability,
                'recommendation': recommendation,
                'alternative_medications': alternative_meds,
                'side_effects': side_effects
            }
            
            self.logger.info(f"Formatted result: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in _format_prediction_results: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _generate_sample_side_effects(self, medication, probability, risk_level):
        """Generate sample side effects for visualization"""
        medication_lower = medication.lower()
        
        # Common side effects for medications
        side_effects_db = {
            'aspirin': [
                {'name': 'Stomach Upset', 'probability': 0.45, 'severity': 0.5, 
                 'description': 'Stomach irritation or discomfort', 
                 'recommendation': 'Take with food to reduce stomach upset'},
                {'name': 'Bleeding', 'probability': 0.30, 'severity': 0.8, 
                 'description': 'Increased risk of bleeding due to blood thinning properties', 
                 'recommendation': 'Stop medication and consult doctor if unusual bleeding occurs'},
                {'name': 'Heartburn', 'probability': 0.25, 'severity': 0.3, 
                 'description': 'Burning sensation in chest', 
                 'recommendation': 'Take with plenty of water'}
            ],
            'ibuprofen': [
                {'name': 'Stomach Pain', 'probability': 0.40, 'severity': 0.5,
                 'description': 'Pain or discomfort in stomach', 
                 'recommendation': 'Take with food to reduce stomach irritation'},
                {'name': 'Dizziness', 'probability': 0.20, 'severity': 0.5,
                 'description': 'Feeling lightheaded or unsteady', 
                 'recommendation': 'Avoid driving or operating machinery if affected'},
                {'name': 'Headache', 'probability': 0.15, 'severity': 0.3,
                 'description': 'Pain in the head or temples', 
                 'recommendation': 'Usually temporary; consult doctor if persistent'}
            ],
            'amoxicillin': [
                {'name': 'Diarrhea', 'probability': 0.35, 'severity': 0.5,
                 'description': 'Loose, watery stools', 
                 'recommendation': 'Stay hydrated and consult doctor if severe'},
                {'name': 'Rash', 'probability': 0.25, 'severity': 0.5,
                 'description': 'Skin eruption or discoloration', 
                 'recommendation': 'Stop medication and contact doctor immediately'},
                {'name': 'Nausea', 'probability': 0.20, 'severity': 0.3,
                 'description': 'Feeling of sickness with an inclination to vomit', 
                 'recommendation': 'Take with food to reduce nausea'}
            ],
            'metformin': [
                {'name': 'Digestive Issues', 'probability': 0.50, 'severity': 0.5,
                 'description': 'Nausea, vomiting, or diarrhea', 
                 'recommendation': 'Take with meals to minimize digestive effects'},
                {'name': 'Vitamin B12 Deficiency', 'probability': 0.15, 'severity': 0.4,
                 'description': 'Long-term use may lead to vitamin B12 deficiency', 
                 'recommendation': 'Regular blood tests recommended for long-term use'},
                {'name': 'Lactic Acidosis', 'probability': 0.05, 'severity': 0.9,
                 'description': 'Rare but serious metabolic complication', 
                 'recommendation': 'Seek immediate medical attention if experiencing unusual muscle pain, trouble breathing, or unusual tiredness'}
            ],
            'lisinopril': [
                {'name': 'Dry Cough', 'probability': 0.55, 'severity': 0.4,
                 'description': 'Persistent, tickling cough', 
                 'recommendation': 'Consult doctor about switching to an ARB if cough is bothersome'},
                {'name': 'Dizziness', 'probability': 0.25, 'severity': 0.5,
                 'description': 'Lightheadedness, especially when standing up quickly', 
                 'recommendation': 'Rise slowly from sitting or lying positions'},
                {'name': 'Hyperkalemia', 'probability': 0.15, 'severity': 0.7,
                 'description': 'Elevated potassium levels in blood', 
                 'recommendation': 'Regular blood tests and avoid high-potassium foods'}
            ]
        }
        
        # Default side effects if medication not in database
        default_side_effects = [
            {'name': f'Common Side Effect of {medication}', 'probability': 0.30, 'severity': 0.5,
             'description': f'Commonly reported side effect with {medication}', 
             'recommendation': 'Monitor and consult doctor if persistent'},
            {'name': f'Less Common Side Effect of {medication}', 'probability': 0.20, 'severity': 0.4,
             'description': f'Less frequently reported side effect with {medication}', 
             'recommendation': 'Usually resolves without intervention'},
            {'name': f'Rare Side Effect of {medication}', 'probability': 0.10, 'severity': 0.7,
             'description': f'Rare but potentially serious side effect of {medication}', 
             'recommendation': 'Seek medical attention if experienced'}
        ]
        
        # Get side effects for the medication or use defaults
        side_effects = side_effects_db.get(medication_lower, default_side_effects)
        
        # Adjust probabilities based on overall risk level
        risk_multiplier = 1.0
        if risk_level == 'High':
            risk_multiplier = 1.3
        elif risk_level == 'Medium':
            risk_multiplier = 1.1
        else:
            risk_multiplier = 0.9
            
        # Apply the risk multiplier to each side effect
        for effect in side_effects:
            effect['probability'] = min(effect['probability'] * risk_multiplier, 0.95)
            
        return side_effects

    def _get_alternative_recommendations(self, drug_name, risk_level, probability):
        """
        İlaç için kanıta dayalı, kategorize edilmiş alternatif önerileri oluşturur
        
        Args:
            drug_name: İlaç adı
            risk_level: Risk seviyesi ('Low', 'Medium', 'High')
            probability: Risk olasılığı (0-1 arası değer)
            
        Returns:
            Dict: Kategorize edilmiş alternatif ilaçlar ve ilgili bilgiler
        """
        # İlaç adını küçük harfe çevir
        drug_name = drug_name.lower()
        
        # İlaç kategorisini bul
        drug_category = self._get_medication_category(drug_name)
        
        # İlaç alternatifleri veri yapısı - kategorize edilmiş ve kanıta dayalı bilgilerle
        drug_alternatives = {
            'aspirin': {
                'same_class': [
                    {
                        'name': 'Naproxen', 
                        'dosage': '220-500mg', 
                        'risk_factor': 0.85,  # Aspirin'e göre %15 daha az riskli
                        'evidence': 'JAMA 2020;323(2):156-158',
                        'description': 'Same class (NSAID) with fewer bleeding complications'
                    },
                    {
                        'name': 'Diclofenac gel', 
                        'dosage': '1%', 
                        'risk_factor': 0.60,  # Aspirin'e göre %40 daha az riskli
                        'evidence': 'BMJ 2018;362:k3426',
                        'description': 'Topical NSAID with minimal systemic absorption'
                    }
                ],
                'different_class': [
                    {
                        'name': 'Acetaminophen (Paracetamol)', 
                        'dosage': '500-1000mg', 
                        'risk_factor': 0.50,  # Aspirin'e göre %50 daha az riskli
                        'evidence': 'Lancet 2018;391:1537-44',
                        'description': 'Non-NSAID analgesic with minimal GI bleeding risk'
                    }
                ],
                'complementary': [
                    {
                        'name': 'Physical therapy',
                        'description': 'For pain management, especially for musculoskeletal conditions',
                        'evidence': 'Annals of Internal Medicine 2017;166(7):514-530'
                    },
                    {
                        'name': 'Heat/cold therapy',
                        'description': 'Simple, non-pharmacological pain management',
                        'evidence': 'American Family Physician 2014;89(5):359-364'
                    }
                ]
            },
            'ibuprofen': {
                'same_class': [
                    {
                        'name': 'Naproxen', 
                        'dosage': '220-500mg', 
                        'risk_factor': 0.90,  # İbuprofen'e göre %10 daha az riskli
                        'evidence': 'NEJM 2016;375:2519-29',
                        'description': 'Same class (NSAID) with better CV safety profile'
                    },
                    {
                        'name': 'Diclofenac gel', 
                        'dosage': '1%', 
                        'risk_factor': 0.65,  # İbuprofen'e göre %35 daha az riskli
                        'evidence': 'BMJ 2018;362:k3426',
                        'description': 'Topical NSAID with minimal systemic absorption'
                    }
                ],
                'different_class': [
                    {
                        'name': 'Acetaminophen (Paracetamol)', 
                        'dosage': '500-1000mg', 
                        'risk_factor': 0.55,  # İbuprofen'e göre %45 daha az riskli
                        'evidence': 'Lancet 2018;391:1537-44',
                        'description': 'Non-NSAID analgesic with minimal GI bleeding risk'
                    }
                ],
                'complementary': [
                    {
                        'name': 'Physical therapy',
                        'description': 'For pain management, especially for musculoskeletal conditions',
                        'evidence': 'Annals of Internal Medicine 2017;166(7):514-530'
                    },
                    {
                        'name': 'Heat/cold therapy',
                        'description': 'Simple, non-pharmacological pain management',
                        'evidence': 'American Family Physician 2014;89(5):359-364'
                    }
                ]
            },
            'metformin': {
                'same_class': [
                    {
                        'name': 'Metformin XR (Extended Release)', 
                        'dosage': '500-1000mg', 
                        'risk_factor': 0.80,  # Standard Metformin'e göre %20 daha az GI yan etki
                        'evidence': 'Diabetes Care 2017;40(2):171-178',
                        'description': 'Same active ingredient with improved GI tolerability'
                    }
                ],
                'different_class': [
                    {
                        'name': 'Sitagliptin (Januvia)', 
                        'dosage': '100mg', 
                        'risk_factor': 0.75,  # Metformin'e göre %25 daha az riskli
                        'evidence': 'NEJM 2015;373:232-242',
                        'description': 'DPP-4 inhibitor with lower risk of GI side effects'
                    },
                    {
                        'name': 'Empagliflozin (Jardiance)', 
                        'dosage': '10-25mg', 
                        'risk_factor': 0.70,  # Metformin'e göre %30 daha az riskli
                        'evidence': 'NEJM 2015;373:2117-2128',
                        'description': 'SGLT-2 inhibitor with cardiovascular benefits'
                    }
                ],
                'complementary': [
                    {
                        'name': 'Dietary modifications',
                        'description': 'Low carbohydrate diet and portion control',
                        'evidence': 'Diabetes Care 2019;42:731-754'
                    },
                    {
                        'name': 'Regular exercise',
                        'description': '150 minutes per week of moderate activity',
                        'evidence': 'JAMA 2017;318(12):1150-1160'
                    }
                ]
            },
            'lisinopril': {
                'same_class': [
                    {
                        'name': 'Ramipril', 
                        'dosage': '2.5-10mg', 
                        'risk_factor': 0.90,  # Lisinopril'e göre %10 daha az öksürük yan etkisi
                        'evidence': 'Hypertension 2018;71:e13–e115',
                        'description': 'ACE inhibitor with slightly lower incidence of cough'
                    }
                ],
                'different_class': [
                    {
                        'name': 'Losartan', 
                        'dosage': '25-100mg', 
                        'risk_factor': 0.60,  # Lisinopril'e göre %40 daha az öksürük yan etkisi
                        'evidence': 'NEJM 2000;342:145-153',
                        'description': 'Angiotensin II receptor blocker (ARB) without cough side effect'
                    },
                    {
                        'name': 'Amlodipine', 
                        'dosage': '5-10mg', 
                        'risk_factor': 0.70,  # Lisinopril'e göre %30 daha az yan etki riski
                        'evidence': 'The Lancet 2000;356:366-372',
                        'description': 'Calcium channel blocker with different side effect profile'
                    }
                ],
                'complementary': [
                    {
                        'name': 'Dietary modifications',
                        'description': 'DASH diet and sodium restriction',
                        'evidence': 'Hypertension 2017;70:923-930'
                    },
                    {
                        'name': 'Weight management',
                        'description': 'Even moderate weight loss can reduce blood pressure',
                        'evidence': 'Circulation 2018;138:e426-e483'
                    }
                ]
            },
            'amoxicillin': {
                'same_class': [
                    {
                        'name': 'Cephalexin', 
                        'dosage': '250-500mg', 
                        'risk_factor': 0.80,  # Amoxicillin'e göre %20 daha az alerjik reaksiyon riski
                        'evidence': 'Clinical Infectious Diseases 2017;64:1-12',
                        'description': 'Different beta-lactam class with lower cross-reactivity'
                    }
                ],
                'different_class': [
                    {
                        'name': 'Azithromycin', 
                        'dosage': '250-500mg', 
                        'risk_factor': 0.75,  # Amoxicillin'e göre %25 daha az GI yan etki
                        'evidence': 'JAMA 2016;315(24):2672-2681',
                        'description': 'Macrolide antibiotic with different mechanism of action'
                    },
                    {
                        'name': 'Doxycycline', 
                        'dosage': '100mg', 
                        'risk_factor': 0.70,  # Amoxicillin'e göre %30 daha az alerjik reaksiyon riski
                        'evidence': 'NEJM 2019;380:517-528',
                        'description': 'Tetracycline class antibiotic, effective for many infections'
                    }
                ],
                'complementary': [
                    {
                        'name': 'Probiotics',
                        'description': 'May reduce antibiotic-associated diarrhea',
                        'evidence': 'Cochrane Database Syst Rev 2017;12:CD004827'
                    }
                ]
            },
            'default': {
                'same_class': [
                    {
                        'name': 'Consult specialist',
                        'description': 'For alternatives within the same medication class',
                        'evidence': 'General recommendation'
                    }
                ],
                'different_class': [
                    {
                        'name': 'Consult specialist',
                        'description': 'For alternatives from different medication classes',
                        'evidence': 'General recommendation'
                    }
                ],
                'complementary': [
                    {
                        'name': 'Consult specialist',
                        'description': 'For non-pharmacological approaches',
                        'evidence': 'General recommendation'
                    }
                ]
            }
        }
        
        # İlaç için alternatifleri bul, yoksa varsayılan önerileri kullan
        alt_recommendations = drug_alternatives.get(drug_name, drug_alternatives['default'])
        
        # Risk seviyesine göre kategori önem sıralaması
        priority_categories = {
            'Low': ['same_class', 'different_class', 'complementary'],
            'Medium': ['same_class', 'different_class', 'complementary'],
            'High': ['different_class', 'same_class', 'complementary']
        }
        
        # Risk seviyesine göre kategori önceliklerini düzenle
        risk_level_lower = risk_level.lower()
        categories_order = priority_categories.get(risk_level, priority_categories['Medium'])
        
        # Sonuç listesi
        formatted_alternatives = []
        
        # Her kategori için filtrelenmiş alternatifler ekle
        for category in categories_order:
            if category in alt_recommendations and alt_recommendations[category]:
                # Kategori bazında alternatiflerin risk hesaplamasını ve sıralamasını yap
                alternatives_with_risk = []
                for alt in alt_recommendations[category]:
                    # Risk hesaplaması
                    if 'risk_factor' in alt:
                        # Alternatif ilacın hesaplanmış risk değeri
                        calculated_risk = probability * alt['risk_factor']
                        risk_level_alt = self._determine_risk_level(calculated_risk)
                        
                        # Alternatif veri yapısına risk bilgisini ekle
                        alt_with_risk = alt.copy()
                        alt_with_risk['calculated_risk'] = calculated_risk
                        alt_with_risk['risk_level'] = risk_level_alt
                        
                        # Alternatif ilaç daha düşük riskli mi kontrolü
                        is_lower_risk = calculated_risk < probability
                        alt_with_risk['is_lower_risk'] = is_lower_risk
                        
                        # Risk seviyesi iyileşme yüzdesi
                        if is_lower_risk:
                            risk_improvement = ((probability - calculated_risk) / probability) * 100
                            alt_with_risk['risk_improvement'] = f"{risk_improvement:.1f}%"
                        
                        alternatives_with_risk.append(alt_with_risk)
            else:
                        # Tamamlayıcı tedaviler için risk hesaplaması yapmıyoruz
                        alternatives_with_risk.append(alt)
                
                # Riski düşük olan alternatifleri öne çıkar
                if category != 'complementary':
                    alternatives_with_risk.sort(key=lambda x: x.get('calculated_risk', 1.0))
                
                # Formatlanmış ve kategorilenmiş alternatifleri ekle
                formatted_category = {
                    'category': category,
                    'category_name': self._get_category_display_name(category),
                    'alternatives': alternatives_with_risk
                }
                
                formatted_alternatives.append(formatted_category)
        
        return formatted_alternatives
        
    def _get_category_display_name(self, category_key):
        """Kategori kodunu kullanıcı dostu başlığa dönüştür"""
        category_names = {
            'same_class': 'Same Medication Class (Similar Alternatives)',
            'different_class': 'Different Medication Class (Alternative Approaches)',
            'complementary': 'Complementary Approaches (Non-Pharmacological)'
        }
        return category_names.get(category_key, category_key.replace('_', ' ').title())

    def _get_recommendation(self, probability):
        """
        Generate recommendation based on prediction probability
        """
        if probability >= 0.60:
            return "Consider discussing alternative medication options with your doctor. Higher risk of significant side effects."
        elif probability >= 0.50:
            return "Monitor for side effects carefully. Moderately elevated risk that may require attention."
        elif probability >= 0.40:
            return "Use medication as prescribed, but stay alert for any unusual reactions. Moderate risk of side effects."
        elif probability >= 0.30:
            return "This medication appears generally safe for you. Lower risk of side effects, but still monitor your symptoms."
        else:
            return "Safe to proceed with prescribed medication. Low risk of side effects based on your profile." 

    def _check_drug_interactions(self, medications, patient_data):
        """
        İlaç etkileşimlerini kontrol eder ve olası etkileşimleri döndürür
        
        Bu fonksiyon gerçek tıbbi veritabanlarında yer alan ilaç etkileşim bilgilerini kullanmalı
        Şu anda sadece yaygın bilinen bazı etkileşimleri içeren bir örnek
        
        Args:
            medications: İlaç listesi
            patient_data: Hasta verileri
            
        Returns:
            List: Tespit edilen ilaç etkileşimleri listesi
        """
        # Etkileşimlerin saklanacağı liste
        interactions = []
        
        # İlaç etkileşim veritabanı - gerçekte daha kapsamlı olmalı
        # Bu, yaygın ilaç etkileşimlerini içeren basit bir örnek
        interaction_db = {
            ('aspirin', 'ibuprofen'): {
                'type': 'increased_bleeding_risk',
                'severity': 'moderate',
                'description': 'Combining NSAIDs may increase risk of bleeding and GI side effects',
                'risk_factor': 0.15  # %15 risk artışı
            },
            ('aspirin', 'warfarin'): {
                'type': 'increased_bleeding_risk',
                'severity': 'severe',
                'description': 'Combining aspirin with warfarin significantly increases bleeding risk',
                'risk_factor': 0.40  # %40 risk artışı
            },
            ('metformin', 'atorvastatin'): {
                'type': 'increased_myopathy_risk',
                'severity': 'mild',
                'description': 'May slightly increase risk of muscle pain or weakness',
                'risk_factor': 0.08  # %8 risk artışı
            },
            ('fluoxetine', 'ibuprofen'): {
                'type': 'increased_bleeding_risk',
                'severity': 'moderate',
                'description': 'SSRIs with NSAIDs can increase bleeding risk',
                'risk_factor': 0.12  # %12 risk artışı
            },
            ('lisinopril', 'ibuprofen'): {
                'type': 'reduced_efficacy',
                'severity': 'moderate',
                'description': 'NSAIDs may reduce blood pressure lowering effects of ACE inhibitors',
                'risk_factor': 0.10  # %10 risk artışı
            },
            ('omeprazole', 'clopidogrel'): {
                'type': 'reduced_efficacy',
                'severity': 'moderate',
                'description': 'Proton pump inhibitors may reduce efficacy of clopidogrel',
                'risk_factor': 0.20  # %20 risk artışı
            },
            ('fluoxetine', 'tramadol'): {
                'type': 'increased_serotonin_syndrome_risk',
                'severity': 'severe',
                'description': 'Increased risk of serotonin syndrome',
                'risk_factor': 0.35  # %35 risk artışı
            }
        }
        
        # Her olası ilaç çifti için etkileşimleri kontrol et
        for i in range(len(medications)):
            for j in range(i+1, len(medications)):
                med1 = medications[i].lower()
                med2 = medications[j].lower()
                
                # Etkileşimi her iki yönde kontrol et (sıra önemli değil)
                interaction_data = interaction_db.get((med1, med2)) or interaction_db.get((med2, med1))
                
                if interaction_data:
                    self.logger.info(f"Detected interaction between {med1} and {med2}: {interaction_data['type']}")
                    interactions.append({
                        'drugs': [med1, med2],
                        'type': interaction_data['type'],
                        'severity': interaction_data['severity'],
                        'description': interaction_data['description'],
                        'risk_factor': interaction_data['risk_factor']
                    })
        
        # Özel koşulları kontrol et
        self._check_conditional_interactions(interactions, medications, patient_data)
        
        return interactions

    def _check_conditional_interactions(self, interactions, medications, patient_data):
        """
        Hasta durumuna bağlı koşullu ilaç etkileşimlerini kontrol eder
        
        Örneğin, böbrek yetmezliğinde bazı ilaç kombinasyonları daha risklidir
        Ya da yaşlı hastalarda bazı kombinasyonlar daha fazla risk oluşturur
        
        Args:
            interactions: Mevcut etkileşim listesi (bu fonksiyon tarafından güncellenecek)
            medications: İlaç listesi
            patient_data: Hasta verileri
        """
        # Hasta yaşını al
        age = patient_data.get('age', 0)
        
        # Tıbbi geçmişi al
        medical_history = patient_data.get('medical_history', [])
        
        # Yaşlı hastalar için özel etkileşimler (65 yaş üstü)
        if age >= 65:
            # İlaçlar arasında mevcut etkileşim varsa, yaşlı hastalarda risk faktörünü artır
            for interaction in interactions:
                interaction['risk_factor'] *= 1.2  # %20 daha fazla risk
                interaction['description'] += " Risk is higher in elderly patients."
            
            # Yaşlı hastalarda özel dikkat gerektiren kombinasyonlar
            if 'ibuprofen' in medications and 'lisinopril' in medications:
                # Eğer bu etkileşim zaten listelenmediyse
                if not any(set(i['drugs']) == set(['ibuprofen', 'lisinopril']) for i in interactions):
                    interactions.append({
                        'drugs': ['ibuprofen', 'lisinopril'],
                        'type': 'increased_kidney_risk',
                        'severity': 'severe',
                        'description': 'NSAIDs with ACE inhibitors can cause kidney damage, especially in elderly',
                        'risk_factor': 0.25  # %25 risk artışı
                    })
        
        # Böbrek yetmezliği olan hastalarda özel etkileşimler
        if 'Kidney Disease' in medical_history:
            # Mevcut etkileşimlerde risk artışı
            for interaction in interactions:
                interaction['risk_factor'] *= 1.3  # %30 daha fazla risk
                interaction['description'] += " Risk is significantly higher with kidney disease."
            
            # Böbrek hastalarında ek etkileşimler
            if 'metformin' in medications:
                # Metformin'i içeren tüm ilaç kombinasyonları için
                other_meds = [m for m in medications if m != 'metformin']
                for other_med in other_meds:
                    # Eğer bu etkileşim zaten listelenmediyse
                    if not any(set(i['drugs']) == set(['metformin', other_med]) for i in interactions):
                        interactions.append({
                            'drugs': ['metformin', other_med],
                            'type': 'increased_lactic_acidosis_risk',
                            'severity': 'severe',
                            'description': 'Metformin with other medications increases lactic acidosis risk in kidney disease',
                            'risk_factor': 0.30  # %30 risk artışı
                        })
        
        # Karaciğer hastalığı olan hastalarda özel etkileşimler
        if 'Liver Disease' in medical_history:
            # Atorvastatin ile herhangi bir ilaç kombinasyonu kontrol et
            if 'atorvastatin' in medications:
                other_meds = [m for m in medications if m != 'atorvastatin']
                for other_med in other_meds:
                    # Eğer bu etkileşim zaten listelenmediyse
                    if not any(set(i['drugs']) == set(['atorvastatin', other_med]) for i in interactions):
                        interactions.append({
                            'drugs': ['atorvastatin', other_med],
                            'type': 'increased_liver_toxicity',
                            'severity': 'severe',
                            'description': 'Statins with other medications may increase liver toxicity in patients with liver disease',
                            'risk_factor': 0.35  # %35 risk artışı
                        })

    def _calculate_interaction_risk(self, interactions):
        """
        İlaç etkileşimlerinin genel risk faktörünü hesaplar
        
        Args:
            interactions: Tespit edilen ilaç etkileşimleri listesi
            
        Returns:
            float: Genel etkileşim risk faktörü (0-1 arası)
        """
        if not interactions:
            return 0.0
        
        # Tüm etkileşimlerden maksimum risk faktörünü hesapla
        # Bu yaklaşım, en yüksek riskli etkileşimi temel alır
        max_risk_factor = max(interaction['risk_factor'] for interaction in interactions)
        
        # Etkileşim sayısına göre ek risk faktörü
        # Çoklu etkileşimler riski daha da artırır
        additional_risk = min(len(interactions) * 0.05, 0.25)  # En fazla %25 ek risk
        
        # Toplam riski hesapla (maksimum 0.8 - %80 risk artışı ile sınırla)
        total_risk = min(max_risk_factor + additional_risk, 0.8)
        
        return total_risk

    def _adjust_for_medical_conditions(self, results, medical_history, patient_data):
        """
        Hasta tıbbi geçmişine göre risk tahminlerini ayarlar
        
        Args:
            results: Tahmin sonuçları listesi
            medical_history: Hasta tıbbi geçmişi
            patient_data: Tüm hasta verileri
            
        Returns:
            List: Güncellenen sonuçlar
        """
        # Tıbbi durumlara göre risk faktörleri
        condition_risk_factors = {
            'Hypertension': {
                'general_factor': 0.10,  # Genel risk artışı
                'specific_meds': {
                    'ibuprofen': 0.15,   # İbuprofen hipertansiyonu kötüleştirir
                    'lisinopril': -0.05  # Lisinopril hipertansiyon ilacı, riski azaltır
                }
            },
            'Diabetes': {
                'general_factor': 0.08,
                'specific_meds': {
                    'metformin': -0.10,  # Metformin diyabet ilacı, riski azaltır
                    'ibuprofen': 0.05    # NSAIDler kan şekerini etkileyebilir
                }
            },
            'HeartDisease': {
                'general_factor': 0.15,
                'specific_meds': {
                    'aspirin': -0.10,    # Aspirin kalp hastalığında koruyucu 
                    'ibuprofen': 0.20,   # NSAIDler kalp hastalığı riskini artırır
                    'atorvastatin': -0.12 # Statinler kalp hastalığını iyileştirir
                }
            },
            'Asthma': {
                'general_factor': 0.05,
                'specific_meds': {
                    'ibuprofen': 0.20   # Bazı astım hastalarında NSAIDler astımı tetikleyebilir
                }
            },
            'Cancer': {
                'general_factor': 0.12,
                'specific_meds': {
                    'metformin': 0.05,
                    'omeprazole': 0.08
                }
            }
        }
        
        # Her sonucu hasta tıbbi durumlarına göre ayarla
        for result in results:
            medication = result['medication']
            probability = float(result['probability'])
            
            # Toplam risk faktörü
            total_risk_factor = 0.0
            
            for condition in medical_history:
                if condition in condition_risk_factors:
                    # Genel faktörü ekle
                    general_factor = condition_risk_factors[condition]['general_factor']
                    
                    # İlaç özel faktörü ekle (varsa)
                    specific_factor = condition_risk_factors[condition]['specific_meds'].get(medication, 0.0)
                    
                    # Toplam faktörü güncelle
                    total_risk_factor += general_factor + specific_factor
                    
                    # Öneriyi güncelle
                    condition_name = condition
                    if specific_factor > 0:
                        result['recommendation'] += f" Use caution with {condition_name} as it may increase risk."
                    elif specific_factor < 0:
                        result['recommendation'] += f" Medication is commonly used for {condition_name} treatment."
            
            # Olasılığı güncelle (riski artırarak veya azaltarak)
            adjusted_probability = probability * (1 + total_risk_factor)
            
            # Mantıklı sınırlar içinde tut (0.05-0.95)
            adjusted_probability = max(min(adjusted_probability, 0.95), 0.05)
            
            # Sonuçları güncelle
            result['probability'] = str(adjusted_probability)
            result['risk_level'] = self._determine_risk_level(adjusted_probability)
        
        return results

    def _generate_recommendation(self, medication, probability, patient_data):
        """Risk seviyesine göre ilaç önerileri üretir"""
        # DataFrame'den scalar değerler çıkar
        age = patient_data.get('patient_age', 0)
        # DataFrame ise ilk değeri al
        if hasattr(age, 'iloc'):
            age = age.iloc[0]
        else:
            age = float(age)
            
        sex = patient_data.get('patient_sex', 0)
        if hasattr(sex, 'iloc'):
            sex = sex.iloc[0]
        else:
            sex = int(sex)
            
        serious = patient_data.get('serious', 0)
        if hasattr(serious, 'iloc'):
            serious = serious.iloc[0]
        else:
            serious = int(serious)
        
        # Yüksek risk
        if risk_level == 'High':
            if serious == 1:
                return "Consider alternative medication due to high risk. Only use if benefits clearly outweigh risks."
            elif float(age) > 65:
                return "Higher risk of side effects, especially in elderly patients. Careful monitoring required."
            else:
                return "Higher risk of side effects. Carefully consider benefit-risk ratio and monitor closely."
        # Orta risk
        elif risk_level == 'Medium':
            if serious == 1:
                return "Moderate risk of side effects. Monitor regularly and report any adverse events immediately."
            else:
                return "Moderate risk of side effects. Take as directed and be aware of potential adverse reactions."
        # Düşük risk
        else:
            return "Safe to proceed with prescribed medication. Low risk of side effects based on your profile."
    
    def _adjust_for_lab_values(self, results, lab_tests, patient_data):
        """
        Laboratuvar test sonuçlarına göre risk tahminlerini ayarlar
        
        Args:
            results: Tahmin sonuçları listesi
            lab_tests: Laboratuvar test sonuçları
            patient_data: Tüm hasta verileri
            
        Returns:
            List: Güncellenen sonuçlar
        """
        # Laboratuvar testleri normal aralıkları
        lab_normal_ranges = {
            'BloodPressure': (90, 140),      # Sistolik, mmHg
            'BloodSugar': (70, 130),         # mg/dL
            'Cholesterol': (125, 200),       # mg/dL
            'Hemoglobin': (12, 17),          # g/dL
            'WhiteBloodCell': (4000, 11000), # hücre/mikroL
            'Platelets': (150000, 450000),   # hücre/mikroL
            'Creatinine': (0.6, 1.3),        # mg/dL
            'ALT': (7, 55),                  # U/L
            'AST': (8, 48)                   # U/L
        }
        
        # İlaç bazlı laboratuvar risk faktörleri
        med_lab_risk_factors = {
            'aspirin': {
                'Platelets': {'low': 0.30, 'high': 0.05},  # Düşük trombosit sayısında kanama riski artar
                'Hemoglobin': {'low': 0.25, 'high': 0.0}   # Düşük hemoglobin kanama riski artışı gösterir
            },
            'ibuprofen': {
                'BloodPressure': {'low': 0.0, 'high': 0.15},  # Yüksek tansiyon NSAIDlerle daha da artabilir
                'Creatinine': {'low': 0.0, 'high': 0.20}     # Yüksek kreatinin böbrek fonksiyon bozukluğunu gösterir
            },
            'metformin': {
                'Creatinine': {'low': 0.0, 'high': 0.40},    # Böbrek fonksiyon bozukluğunda laktik asidoz riski
                'BloodSugar': {'low': 0.10, 'high': 0.10}    # Düşük kan şekeri hipoglisemi, yüksekse etkisizlik
            },
            'atorvastatin': {
                'ALT': {'low': 0.0, 'high': 0.30},          # Karaciğer enzim yüksekliği karaciğer hasarı gösterebilir
                'AST': {'low': 0.0, 'high': 0.30}           # Karaciğer enzim yüksekliği karaciğer hasarı gösterebilir
            },
            'lisinopril': {
                'BloodPressure': {'low': 0.20, 'high': -0.10}, # Düşük tansiyon aşırı düşük tansiyona, yüksekse tedavi eder
                'Creatinine': {'low': 0.0, 'high': 0.25}      # Böbrek fonksiyon bozukluğu riski
            }
        }
        
        # Her sonucu laboratuvar değerlerine göre ayarla
        for result in results:
            medication = result['medication']
            probability = float(result['probability'])
            
            # Eğer bu ilaç için tanımlanmış lab risk faktörleri yoksa atla
            if medication not in med_lab_risk_factors:
                continue
            
            # Toplam risk faktörü
            total_lab_risk_factor = 0.0
            
            # Bu ilaç için laboratuvar risk faktörlerini kontrol et
            med_risks = med_lab_risk_factors[medication]
            
            # Her laboratuvar testi için
            for lab_test in lab_tests:
                test_name = lab_test.get('testname')
                test_result = lab_test.get('testresult')
                
                # Eğer bu test için tanımlanmış normal aralıklar ve ilaç risk faktörleri yoksa atla
                if test_name not in lab_normal_ranges or test_name not in med_risks:
                    continue
                
                try:
                    # Test sonucunu sayıya çevir
                    test_value = float(test_result)
                    
                    # Normal aralıkları al
                    normal_min, normal_max = lab_normal_ranges[test_name]
                    
                    # Test değerinin normal aralığın dışında olup olmadığını kontrol et
                    if test_value < normal_min:
                        # Düşük değer için risk faktörünü ekle
                        risk_factor = med_risks[test_name].get('low', 0.0)
                        total_lab_risk_factor += risk_factor
                        
                        if risk_factor > 0:
                            result['recommendation'] += f" Caution: Low {test_name} increases risk."
                            
                    elif test_value > normal_max:
                        # Yüksek değer için risk faktörünü ekle
                        risk_factor = med_risks[test_name].get('high', 0.0)
                        total_lab_risk_factor += risk_factor
                        
                        if risk_factor > 0:
                            result['recommendation'] += f" Caution: High {test_name} increases risk."
                    
                except (ValueError, TypeError):
                    # Sayısal olmayan değerler için hata kontrolü
                    continue
            
            # Olasılığı güncelle (riski artırarak veya azaltarak)
            adjusted_probability = probability * (1 + total_lab_risk_factor)
            
            # Mantıklı sınırlar içinde tut (0.05-0.95)
            adjusted_probability = max(min(adjusted_probability, 0.95), 0.05)
            
            # Sonuçları güncelle
            result['probability'] = str(adjusted_probability)
            result['risk_level'] = self._determine_risk_level(adjusted_probability)
        
        return results
    
    def _adjust_for_age(self, results, age):
        """
        Hasta yaşına göre risk tahminlerini ayarlar
        
        Args:
            results: Tahmin sonuçları listesi
            age: Hasta yaşı
            
        Returns:
            List: Güncellenen sonuçlar
        """
        try:
            # Yaşı sayıya çevir
            age = float(age)
            
            # Yaş risk faktörü - yaşlı hastalarda yan etki riski genellikle artar
            # Özellikle 65 yaş üstü
            age_factor = 0.0
            
            if age >= 65:
                # 65-74 arası
                age_factor = 0.15
            elif age >= 75:
                # 75 ve üstü
                age_factor = 0.25
            elif age < 18:
                # 18 yaş altı (pediatrik)
                age_factor = 0.20
            
            # Yaş faktörü sıfırdan büyükse, her ilacın riskini ayarla
            if age_factor > 0:
                for result in results:
                    medication = result['medication']
                    probability = float(result['probability'])
                    
                    # İlaç bazlı yaş faktörleri - bazı ilaçlar yaşlı hastalarda daha riskli
                    med_age_factors = {
                        'aspirin': 0.10,     # Özellikle kanama riski
                        'ibuprofen': 0.15,   # Böbrek/GI riski
                        'metformin': 0.05,   # Böbrek fonksiyonu genelde yaşla azalır
                        'lisinopril': 0.10,  # Tansiyon çok düşebilir
                        'atorvastatin': 0.08 # Kas ağrısı riski
                    }
                    
                    # Bu ilaç için ek yaş faktörü
                    med_factor = med_age_factors.get(medication, 0.05)
                    
                    # Toplam yaş faktörü
                    total_age_factor = age_factor * med_factor
                    
                    # Olasılığı güncelle
                    adjusted_probability = probability * (1 + total_age_factor)
                    
                    # Mantıklı sınırlar içinde tut (0.05-0.95)
                    adjusted_probability = max(min(adjusted_probability, 0.95), 0.05)
                    
                    # Sonuçları güncelle
                    result['probability'] = str(adjusted_probability)
                    result['risk_level'] = self._determine_risk_level(adjusted_probability)
                    
                    # Yaş uyarısı ekle
                    age_group = "elderly" if age >= 65 else ("pediatric" if age < 18 else "")
                    if age_group:
                        result['recommendation'] += f" Use with caution in {age_group} patients."
        
        except (ValueError, TypeError):
            # Yaş değeri geçersizse hiçbir değişiklik yapma
            pass
            
        return results 

    def _create_feature_vector(self, prediction_data):
        """
        Convert prediction data to feature vector compatible with the model
        """
        try:
            # If prediction_data is a DataFrame, use it directly, otherwise create a DataFrame
            if isinstance(prediction_data, pd.DataFrame):
                features = prediction_data
            else:
                # Convert to DataFrame
                features = pd.DataFrame(prediction_data, index=[0])
            
            # Ensure all required features are present
            if self.feature_names:
                missing_features = [feat for feat in self.feature_names if feat not in features.columns]
                if missing_features:
                    self.logger.warning(f"Missing features in _create_feature_vector: {missing_features}")
                    for feat in missing_features:
                        features[feat] = 0  # Add missing features with default value
                
                # Reorder columns to match the feature names order
                features = features[self.feature_names]
            
            # Fill any remaining NaN values
            features = features.fillna(0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error in _create_feature_vector: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Return empty DataFrame with correct columns as fallback
            if self.feature_names:
                return pd.DataFrame(columns=self.feature_names)
            else:
                return pd.DataFrame()
                
    def _determine_risk_level(self, probability):
        """
        Determine risk level based on probability
        """
        if probability < 0.40:
            return 'Low'
        elif probability < 0.60:
            return 'Medium'
        else:
            return 'High'

    def _suggest_alternatives(self, medication, probability, patient_data):
        """
        İlaç için alternatif önerileri sunar
        
        Args:
            medication: İlaç adı
            probability: Yan etki olasılığı
            patient_data: Hasta verileri
            
        Returns:
            List: Alternatif ilaç önerileri
        """
        # İlaç kategorilerine göre alternatifler
        alternatives = {
            'aspirin': ['Acetaminophen (Tylenol)', 'Naproxen'],
            'ibuprofen': ['Acetaminophen', 'Naproxen', 'Diclofenac'],
            'amoxicillin': ['Azithromycin', 'Cephalexin', 'Doxycycline'],
            'metformin': ['Sitagliptin', 'Pioglitazone', 'Lifestyle changes'],
            'lisinopril': ['Losartan', 'Valsartan', 'Amlodipine'],
            'atorvastatin': ['Rosuvastatin', 'Pravastatin', 'Simvastatin'],
            'fluoxetine': ['Sertraline', 'Escitalopram', 'Bupropion'],
            'omeprazole': ['Pantoprazole', 'Famotidine', 'Lifestyle changes']
        }
        
        # Risk seviyesine göre önerme şekli
        if probability >= 0.7:
            message = "Strongly consider alternative: "
        elif probability >= 0.5:
            message = "Consider alternative: "
        else:
            message = "Possible alternative if needed: "
        
        # Önerileri formatla
        medication_lower = medication.lower()
        if medication_lower in alternatives:
            return [f"{message}{alt}" for alt in alternatives[medication_lower]]
        else:
            return ["Consult your healthcare provider for alternative options"]

    def _get_medication_category(self, medication_name):
        """İlaç kategorisini belirle"""
        categories = {
            'aspirin': 'NSAID',
            'ibuprofen': 'NSAID',
            'acetaminophen': 'Analgesic',
            'amoxicillin': 'Antibiotic',
            'metformin': 'Antidiabetic',
            'lisinopril': 'ACE Inhibitor',
            'atorvastatin': 'Statin',
            'fluoxetine': 'SSRI',
            'omeprazole': 'PPI',
            'warfarin': 'Anticoagulant',
            'simvastatin': 'Statin',
            'clopidogrel': 'Antiplatelet',
            'losartan': 'ARB',
            'amlodipine': 'Calcium Channel Blocker',
            'levothyroxine': 'Thyroid Hormone'
        }
        
        return categories.get(medication_name.lower(), 'Unknown') 