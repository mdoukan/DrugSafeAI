import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import sys
from pathlib import Path
import logging
import asyncio
import json
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from config import Config
from app.services.openfda_service import OpenFDAService

class ModelTrainer:
    def __init__(self, api_key: str):
        self.data_service = OpenFDAService(api_key)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []  # Feature isimlerini tutacak liste
        self.logger = logging.getLogger(__name__)
        
        # XGBoost parametreleri
        self.model = XGBClassifier(
            n_estimators=100,          # Azaltıldı
            learning_rate=0.1,         # Artırıldı
            max_depth=5,               # Azaltıldı
            min_child_weight=1,        # Azaltıldı
            gamma=0,                   # Azaltıldı
            subsample=0.8,             # Azaltıldı
            colsample_bytree=0.8,      # Azaltıldı
            objective='binary:logistic',
            random_state=42,
            eval_metric='auc',
            scale_pos_weight=1,
            early_stopping_rounds=10    # Azaltıldı
        )

    async def collect_training_data(self, drug_list: List[str]) -> pd.DataFrame:
        """OpenFDA API'den eğitim verilerini topla"""
        all_data = []
        max_events_per_drug = 500  # Her ilaç için max 500 olay (1000'den 500'e düşürüldü)
        batch_size = 500  # Her sorguda 500 olay çek
        
        for drug in drug_list:
            try:
                self.logger.info(f"Fetching data for drug: {drug}")
                total_events = 0
                offset = 0
                
                # Her ilaç için en fazla 2 batch çek (max 1000 olay)
                max_batches = 2
                batch_count = 0
                
                # Her ilaç için istenilen sayıda veri çekene kadar birden fazla sorgu yap
                while total_events < max_events_per_drug and batch_count < max_batches:
                    batch_count += 1
                    self.logger.info(f"Fetching batch for {drug}: offset={offset}, limit={batch_size}")
                    events = await self.data_service.fetch_adverse_events(drug, limit=batch_size, offset=offset)
                    
                    if not events:
                        self.logger.warning(f"No more events found for {drug} after {total_events} events")
                        break
                    
                    self.logger.info(f"Received {len(events)} events for {drug} (batch {offset//batch_size + 1})")
                    processed_events = self._process_events(events, drug)
                    
                    if processed_events:
                        self.logger.info(f"Processed {len(processed_events)} valid events for {drug} in current batch")
                        all_data.extend(processed_events)
                        total_events += len(processed_events)
                    
                    # Tüm sonuçları aldıysak döngüyü sonlandır
                    if len(events) < batch_size:
                        self.logger.info(f"Reached end of results for {drug}")
                        break
                    
                    # Sonraki sayfaya geç
                    offset += batch_size
                    
                    # API rate limiting - her sorgu sonrası bekleme (2 saniyeden 1 saniyeye düşürüldü)
                    await asyncio.sleep(1)
                
                self.logger.info(f"Total processed events for {drug}: {total_events}")
                
                # İlaçlar arası geçişte 1 saniye bekle (3 saniyeden 1 saniyeye düşürüldü)
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error processing drug {drug}: {str(e)}")
                continue
        
        if not all_data:
            self.logger.error("No valid data collected")
            return pd.DataFrame()
            
        self.logger.info(f"Total collected data: {len(all_data)} records")
        return pd.DataFrame(all_data)

    def _process_events(self, events: List[Dict[str, Any]], drug_name: str) -> List[Dict[str, Any]]:
        """Process OpenFDA events into training data"""
        processed_data = []
        
        for event in events:
            try:
                patient = event.get('patient', {})
                
                # Temel hasta bilgilerini al
                record = {
                    'drug_name': drug_name,
                    'patient_age': self._safe_convert_to_float(patient.get('patientonsetage')),
                    'patient_sex': self._safe_convert_to_int(patient.get('patientsex')),
                    'height': self._safe_convert_to_float(patient.get('patientheight')),
                    'weight': self._safe_convert_to_float(patient.get('patientweight')),
                    'serious': 1 if event.get('serious') else 0
                }
                
                # Reaksiyon bilgisini al
                reactions = patient.get('reaction', [])
                if reactions:
                    record['outcome'] = self._safe_convert_to_int(reactions[0].get('reactionoutcome', 0))
                else:
                    record['outcome'] = 0
                
                # Tıbbi geçmiş ve laboratuvar testlerini ekle
                record['medical_history'] = patient.get('medical_history', [])
                record['laboratory_tests'] = patient.get('laboratory_tests', [])
                
                # İlaç listesini al
                record['drugs'] = [
                    drug.get('medicinalproduct', '').lower()
                    for drug in patient.get('drug', [])
                ]
                
                # Zorunlu alanları kontrol et
                required_fields = ['patient_age', 'patient_sex', 'weight', 'serious', 'outcome']
                if all(record[field] is not None for field in required_fields):
                    processed_data.append(record)
                
            except Exception as e:
                self.logger.error(f"Error processing event: {str(e)}")
                continue
                
        return processed_data

    def _safe_convert_to_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float"""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _safe_convert_to_int(self, value: Any) -> Optional[int]:
        """Safely convert value to int"""
        if value is None:
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None

    def prepare_features(self, df):
        """Özellik hazırlama ve mühendisliği"""
        if df.empty:
            raise ValueError("No valid data available for training")

        print("DataFrame columns:", df.columns.tolist())  # Debug için
        print("Number of records:", len(df))  # Debug için

        # Eksik değerleri doldur
        df = self._handle_missing_values(df)
        
        # Temel özellikler
        features = pd.DataFrame()
        
        # Demografik özellikler - güvenli erişim
        required_columns = ['patient_age', 'weight', 'patient_sex', 'serious']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        features['patient_age'] = df['patient_age']
        features['weight'] = df['weight']
        features['patient_sex'] = df['patient_sex']
        features['serious'] = df['serious']

        # Yaş grupları - yaşa bağlı yan etkileri daha iyi modelleyebilmek için
        age_bins = [0, 12, 18, 30, 45, 65, 80, 150]
        age_labels = ['child', 'teen', 'young_adult', 'adult', 'middle_aged', 'senior', 'elderly']
        features['age_group'] = pd.cut(df['patient_age'], bins=age_bins, labels=age_labels, right=False)
        
        # One-hot encoding yaş grupları için
        age_dummies = pd.get_dummies(features['age_group'], prefix='age_group')
        features = pd.concat([features, age_dummies], axis=1)
        features.drop('age_group', axis=1, inplace=True)

        # BMI hesaplama (eğer height ve weight varsa)
        if 'height' in df.columns and 'weight' in df.columns:
            mask = (df['height'] > 0) & (df['weight'] > 0)
            features['bmi'] = 0.0  # Varsayılan değer
            features.loc[mask, 'bmi'] = df.loc[mask, 'weight'] / ((df.loc[mask, 'height']/100) ** 2)
            
            # BMI kategorileri
            bmi_bins = [0, 18.5, 25, 30, 35, 100]
            bmi_labels = ['underweight', 'normal', 'overweight', 'obese', 'extremely_obese']
            features['bmi_category'] = pd.cut(features['bmi'], bins=bmi_bins, labels=bmi_labels, right=False)
            
            # One-hot encoding BMI kategorileri için
            bmi_dummies = pd.get_dummies(features['bmi_category'], prefix='bmi')
            features = pd.concat([features, bmi_dummies], axis=1)
            features.drop('bmi_category', axis=1, inplace=True)

        # İlaç özelinde özellikler ekle - her ilaç için ayrı bir kolon
        if 'drug_name' in df.columns:
            # İlaç adına göre one-hot encoding
            drug_dummies = pd.get_dummies(df['drug_name'], prefix='drug')
            features = pd.concat([features, drug_dummies], axis=1)
        
        # Tıbbi geçmiş özellikleri
        medical_condition_columns = [col for col in df.columns if col.startswith('has_')]
        for col in medical_condition_columns:
            features[col] = df[col]

        # Laboratuvar test özellikleri
        lab_test_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in ['BloodPressure', 'BloodSugar', 'Cholesterol', 'Hemoglobin', 'WhiteBloodCell', 'Platelets', 'Creatinine', 'ALT', 'AST', 'Potassium', 'Sodium', 'Calcium', 'Magnesium', 'Phosphorus', 'Bilirubin', 'Albumin', 'UricAcid', 'TSH', 'VitaminD'])]
        for col in lab_test_columns:
            features[col] = df[col]

        # İlaç etkileşimleri
        features = self._process_drug_interactions(df, features)
        
        # Çıktı değişkeni ile korelasyonu düşük olan sütunları kaldır
        if 'outcome' in df.columns and len(features.columns) > 10:
            correlations = pd.DataFrame({
                'feature': features.columns,
                'correlation': [abs(df['outcome'].corr(features[col])) if features[col].dtype != 'object' else 0 
                               for col in features.columns]
            })
            
            # En yüksek korelasyonlu özellikleri göster
            top_features = correlations.sort_values('correlation', ascending=False).head(20)
            print("Top correlated features:", top_features['feature'].tolist())
        
        # Kategorik değişkenleri kodla
        categorical_columns = ['patient_sex']
        for col in categorical_columns:
            if col in features.columns:  # Sadece hala mevcut olan sütunları dönüştür
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                features[col] = self.label_encoders[col].fit_transform(features[col].astype(str))

        # Eksik değerleri doldur
        features = features.fillna(0)
        
        # Özellikleri ölçeklendir
        numeric_cols = features.select_dtypes(include=['float64', 'int64']).columns
        if not numeric_cols.empty:
            features[numeric_cols] = self.scaler.fit_transform(features[numeric_cols])

        # Hedef değişken
        y = (df['outcome'] >= 2).astype(int)
        
        print(f"Final feature shape: {features.shape}")
        return features, y

    def _handle_missing_values(self, df):
        """Eksik değerleri akıllıca doldur"""
        # Sayısal değerler için medyan
        numeric_cols = ['patient_age', 'weight', 'height']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        # Kategorik değerler için mod
        categorical_cols = ['patient_sex']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])

        # Tıbbi geçmiş özellikleri için 0
        medical_condition_cols = [col for col in df.columns if col.startswith('has_')]
        for col in medical_condition_cols:
            df[col] = df[col].fillna(0)

        # Laboratuvar test özellikleri için
        lab_test_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in ['BloodPressure', 'BloodSugar', 'Cholesterol', 'Hemoglobin', 'WhiteBloodCell', 'Platelets', 'Creatinine', 'ALT', 'AST', 'Potassium', 'Sodium', 'Calcium', 'Magnesium', 'Phosphorus', 'Bilirubin', 'Albumin', 'UricAcid', 'TSH', 'VitaminD'])]
        
        for col in lab_test_cols:
            if col.endswith('_exists'):
                df[col] = df[col].fillna(0)
            elif col.endswith('_value'):
                df[col] = df[col].fillna(df[col].median())
            elif col.endswith('_normal'):
                df[col] = df[col].fillna(0)

        # Listeler için boş liste
        if 'medical_history' in df.columns:
            df['medical_history'] = df['medical_history'].fillna('[]')
        
        if 'laboratory_tests' in df.columns:
            df['laboratory_tests'] = df['laboratory_tests'].fillna('[]')
        
        if 'drugs' in df.columns:
            df['drugs'] = df['drugs'].fillna('[]')
        
        return df

    def _process_drug_interactions(self, df, features):
        """İlaç etkileşimlerini işle"""
        # İlaç kombinasyonlarını kontrol et
        risky_combinations = {
            ('aspirin', 'ibuprofen'): 'blood_thinner',
            ('metformin', 'insulin'): 'blood_sugar',
            # Daha fazla kombinasyon eklenebilir
        }

        # Varsayılan olarak tüm etkileşimleri 0 yap
        for _, risk_type in risky_combinations.items():
            features[f'interaction_{risk_type}'] = 0

        # Eğer drugs sütunu varsa etkileşimleri kontrol et
        if 'drugs' in df.columns:
            for drugs, risk_type in risky_combinations.items():
                features[f'interaction_{risk_type}'] = df['drugs'].apply(
                    lambda drug_list: 1 if all(drug in drug_list for drug in drugs) else 0
                )

        return features

    async def train_model(self, drug_list: List[str]) -> Optional[Dict[str, Any]]:
        """Modeli eğit"""
        try:
            # Veriyi topla
            df = await self.collect_training_data(drug_list)
            
            if df.empty:
                self.logger.error("No data available for training")
                return None
            
            # Özellikleri hazırla
            X, y = self.prepare_features(df)
            
            # Feature isimlerini kaydet
            self.feature_names = X.columns.tolist()
            self.logger.info(f"Feature names: {self.feature_names}")
            
            # Veriyi böl
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Modeli eğit
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            # Modeli değerlendir
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Modeli kaydet
            self._save_model()
            
            return {
                'accuracy': accuracy,
                'best_params': self.model.get_params(),
                'n_samples': len(df)
            }
            
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            return None

    def _save_model(self):
        """Modeli kaydet"""
        try:
            model_path = Path(__file__).parent.parent.parent / 'app' / 'models'
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Feature isimlerini kaydet
            self.logger.info(f"Saving model with feature names: {self.feature_names}")
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names  # Feature isimlerini ekle
            }
            
            joblib.dump(model_data, model_path / 'xgboost_model.pkl')
            self.logger.info("Model saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise 