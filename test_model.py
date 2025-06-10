import sys
from pathlib import Path
import logging

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

if __name__ == "__main__":
    result = test_model()
    if result:
        print("\nTest Results:")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Probability: {result['probability']*100:.1f}%")
        print(f"Recommendation: {result['recommendation']}")
    else:
        print("\nTest failed. Check the logs for details.") 