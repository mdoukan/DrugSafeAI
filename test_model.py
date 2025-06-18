import sys
from pathlib import Path
import logging
import csv
import random

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from app.services.prediction_service import PredictionService

def test_model():
    try:
        logger.info("Initializing PredictionService for testing")
        prediction_service = PredictionService()
        
        # Test verisi oluştur
        test_data = {
            'age': 75,
            'weight': 90,
            'height': 165,
            'sex': 1,  # 1 for male, 2 for female
            'serious': 1,
            'medical_history': ['heart_disease', 'diabetes', 'hypertension'],
            'laboratory_tests': [],
            'medications': ['aspirin', 'metformin']  # Test two medications
        }
        
        logger.info(f"Test data: {test_data}")
        
        # Tahmin yap
        results = prediction_service.predict_side_effects(test_data)
        
        logger.info(f"Prediction results: {results}")
        
        # First medication results
        first_med = results['results'][0]
        
        return {
            'risk_level': first_med['risk_level'],
            'probability': first_med['probability'],
            'recommendation': first_med['recommendation']
        }
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def generate_sample_data(num_samples=100):
    """Rastgele örnek hasta verileri üretir."""
    meds = ['aspirin', 'metformin', 'lisinopril', 'ibuprofen', 'atorvastatin', 'amoxicillin', 'fluoxetine', 'omeprazole']
    histories = ['heart_disease', 'diabetes', 'hypertension', 'asthma', 'cancer', 'kidney_disease', 'liver_disease']
    data = []
    for _ in range(num_samples):
        age = random.randint(18, 90)
        weight = random.randint(50, 120)
        height = random.randint(150, 200)
        sex = random.choice([1, 2])
        serious = random.choice([0, 1])
        medical_history = random.sample(histories, k=random.randint(0, 3))
        medications = random.sample(meds, k=random.randint(1, 3))
        data.append({
            'age': age,
            'weight': weight,
            'height': height,
            'sex': sex,
            'serious': serious,
            'medical_history': medical_history,
            'laboratory_tests': [],
            'medications': medications
        })
    return data

def batch_test_and_save_probs(num_samples=100, out_file='probs.csv'):
    logger.info("Initializing PredictionService for batch testing")
    prediction_service = PredictionService()
    test_data_list = generate_sample_data(num_samples)
    all_probs = []
    with open(out_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sample_idx', 'medication', 'probability'])
        for idx, test_data in enumerate(test_data_list):
            if idx < 3:  # İlk 3 örneği debug için logla
                logger.info(f"Sample {idx} test data: {test_data}")
            
            results = prediction_service.predict_side_effects(test_data)
            
            if idx < 3:  # İlk 3 örneğin çıktısını logla
                logger.info(f"Sample {idx} results type: {type(results)}")
                logger.info(f"Sample {idx} results: {results}")
            
            # predict_side_effects fonksiyonu nested bir yapı döndürüyor
            # results['data']['results'] şeklinde erişmemiz gerekiyor
            if isinstance(results, dict) and 'data' in results and 'results' in results['data']:
                results_list = results['data']['results']
                for res in results_list:
                    if 'probability' in res and 'medication' in res:
                        prob = float(res['probability'])
                        med = res['medication']
                        writer.writerow([idx, med, prob])
                        all_probs.append(prob)
            # Eğer çıktı doğrudan bir liste ise (eski format)
            elif isinstance(results, list):
                for res in results:
                    if 'probability' in res and 'medication' in res:
                        prob = float(res['probability'])
                        med = res['medication']
                        writer.writerow([idx, med, prob])
                        all_probs.append(prob)
            # Eğer çıktı doğrudan bir sözlük ise (tek ilaç)
            elif isinstance(results, dict):
                if 'probability' in results and 'medication' in results:
                    prob = float(results['probability'])
                    med = results['medication']
                    writer.writerow([idx, med, prob])
                    all_probs.append(prob)
            else:
                logger.warning(f"Sample {idx} için beklenmeyen çıktı formatı: {type(results)}")
    logger.info(f"Saved {len(all_probs)} probabilities to {out_file}")
    if len(all_probs) == 0:
        logger.error("Hiç probability kaydedilemedi! Model veya veri formatı kontrol edilmeli.")
    return all_probs

if __name__ == "__main__":
    # Tekli test yerine toplu test ve histogram için dosya kaydı
    batch_test_and_save_probs(num_samples=200, out_file='probs.csv')
    print("Olasılıklar probs.csv dosyasına kaydedildi.") 