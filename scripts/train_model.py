import asyncio
import logging
from pathlib import Path
import sys

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from config import Config
from app.models.train_model import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    try:
        # İlaç listesi - örnek olarak en yaygın ilaçlar
        drug_list = [
            'aspirin',
            'ibuprofen',
            'metformin',
            'lisinopril',
            'simvastatin',
            'amiodarone',
            'warfarin',
            'insulin',
            'spironolactone'
        ]
        
        # Model trainer'ı başlat
        trainer = ModelTrainer(Config.OPENFDA_API_KEY)
        
        # Modeli eğit
        logger.info("Starting model training...")
        results = await trainer.train_model(drug_list)
        
        if results:
            logger.info(f"Model training completed successfully!")
            logger.info(f"Accuracy: {results['accuracy']:.4f}")
            logger.info(f"Number of samples: {results['n_samples']}")
            logger.info(f"Best parameters: {results['best_params']}")
        else:
            logger.error("Model training failed!")
            
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 