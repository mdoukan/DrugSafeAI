import asyncio
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from app.models.train_model import ModelTrainer
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting model training...")
    
    # API key'i config'den al
    api_key = Config.OPENFDA_API_KEY
    
    # Eğitilecek ilaç listesi
    drug_list = [
        'aspirin',
        'ibuprofen',
        'amoxicillin', 
        'metformin',
        'lisinopril',
        'atorvastatin',
        'fluoxetine',
        'omeprazole'
    ]
    
    # Model eğitici oluştur
    trainer = ModelTrainer(api_key)
    
    # Modeli eğit
    results = await trainer.train_model(drug_list)
    
    if results:
        logger.info("Model training completed successfully!")
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Number of samples: {results['n_samples']}")
        logger.info(f"Best parameters: {results['best_params']}")
        
        # İlaç bazında tahminleri göster
        logger.info("Predictions by medication:")
        for drug, prob in results.get('predictions_by_drug', {}).items():
            risk_level = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
            logger.info(f"  {drug}: {prob:.4f} ({risk_level} risk)")
    else:
        logger.error("Model training failed!")

if __name__ == "__main__":
    asyncio.run(main()) 