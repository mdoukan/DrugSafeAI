import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model():
    try:
        # Model dosyasının yolunu belirle
        model_path = Path(__file__).parent / 'app' / 'models' / 'xgboost_model.pkl'
        logger.info(f"Loading model from: {model_path}")
        
        # Modeli yükle
        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data['scaler']
        label_encoders = model_data['label_encoders']
        
        logger.info("Model loaded successfully")
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Model parameters: {model.get_params()}")
        logger.info(f"Model feature names: {model.feature_names_in_ if hasattr(model, 'feature_names_in_') else 'Not available'}")
        
        # Test verisi oluştur
        test_data = {
            'age': 75,
            'weight': 90,
            'height': 165,
            'sex': 2,
            'serious': 1,
            'medical_history': ['heart_disease', 'diabetes', 'hypertension', 'kidney_disease'],
            'laboratory_tests': []
        }
        
        # Veriyi DataFrame'e dönüştür
        df = pd.DataFrame([{
            'age': float(test_data['age']),
            'weight': float(test_data['weight']),
            'sex': int(test_data['sex']),
            'serious': int(test_data['serious']),
            'height': float(test_data['height'])
        }])
        
        # BMI hesapla
        mask = (df['height'] > 0) & (df['weight'] > 0)
        df['bmi'] = 0.0
        df.loc[mask, 'bmi'] = df.loc[mask, 'weight'] / ((df.loc[mask, 'height']/100) ** 2)
        
        # Tıbbi geçmiş özelliklerini ekle
        medical_conditions = [
            'heart_disease', 'asthma', 'cancer', 'diabetes', 'hypertension',
            'kidney_disease', 'liver_disease', 'thyroid_disease', 'arthritis',
            'depression', 'allergies', 'gastrointestinal', 'neurological',
            'respiratory', 'immune_system'
        ]
        
        for condition in medical_conditions:
            df[f'has_{condition}'] = 0
            if condition in test_data['medical_history']:
                df[f'has_{condition}'] = 1
        
        # Laboratuvar test özelliklerini ekle
        lab_tests = [
            'BloodPressure', 'BloodSugar', 'Cholesterol', 'Hemoglobin',
            'WhiteBloodCell', 'Platelets', 'Creatinine', 'ALT', 'AST',
            'Potassium', 'Sodium', 'Calcium', 'Magnesium', 'Phosphorus',
            'Bilirubin', 'Albumin', 'UricAcid', 'TSH', 'VitaminD'
        ]
        
        for test in lab_tests:
            df[f'{test}_exists'] = 0
            df[f'{test}_value'] = 0
            df[f'{test}_normal'] = 0
        
        # İlaç etkileşim özelliklerini ekle
        df['interaction_blood_thinner'] = 0
        df['interaction_blood_sugar'] = 0
        
        # Eksik sütunları ekle
        expected_columns = [
            'age', 'weight', 'sex', 'serious', 'bmi'
        ] + [f'has_{condition}' for condition in medical_conditions] + \
        [f'{test}_{suffix}' for test in lab_tests for suffix in ['exists', 'value', 'normal']] + \
        ['interaction_blood_thinner', 'interaction_blood_sugar']
        
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Gereksiz sütunları kaldır
        df = df[expected_columns]
        
        logger.info("Test data prepared successfully")
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"DataFrame sample: {df.iloc[0].to_dict()}")
        
        # Kategorik değişkenleri kodla
        if 'sex' in df.columns:
            df['sex'] = label_encoders['sex'].transform(df['sex'].astype(str))
        
        # Veriyi ölçeklendir
        scaled_data = scaler.transform(df)
        scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
        
        logger.info("Data scaled successfully")
        logger.info(f"Scaled DataFrame shape: {scaled_df.shape}")
        
        # Tahmin yap
        prediction_prob = model.predict_proba(scaled_df)
        prediction = model.predict(scaled_df)
        
        logger.info(f"Raw prediction: {prediction}")
        logger.info(f"Raw prediction probability: {prediction_prob}")
        
        # Sonuçları formatla
        risk_level = "High" if prediction[0] == 1 else "Low"
        probability = float(prediction_prob[0][1])
        
        result = {
            'risk_level': risk_level,
            'probability': probability,
            'recommendation': "Consider alternative medication. High risk of severe side effects." if probability >= 0.7
                            else "Monitor patient closely. Moderate risk of side effects." if probability >= 0.4
                            else "Safe to proceed with prescribed medication. Low risk of side effects."
        }
        
        logger.info(f"Final prediction result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    result = test_model()
    if result:
        print("\nTest Results:")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Probability: {result['probability']*100:.1f}%")
        print(f"Recommendation: {result['recommendation']}")
    else:
        print("\nTest failed. Check the logs for details.") 