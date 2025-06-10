#!/usr/bin/env python
"""
DrugSafeAI Otomatik Model Güncelleme Betiği
-------------------------------------------
Bu betik, DrugSafeAI modelini otomatik olarak güncellemek için kullanılır.
Periyodik olarak (örn. günlük) çalıştırılabilir ve gerekirse modeli yeni
verilerle günceller.

Kullanım:
    python update_model.py [--force]

Parametreler:
    --force    Zorunlu güncelleme yap (güncelleme gerekli olmazsa bile)
"""

import sys
import asyncio
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_update.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from app.services.prediction_service import PredictionService
from app.models.train_model import ModelTrainer
from config import Config

async def update_model(force=False):
    """Modeli güncelle"""
    logger.info("Starting model update check")
    
    try:
        # PredictionService başlat
        prediction_service = PredictionService()
        
        # Eğer zorunlu güncelleme istenmişse veya güncelleme gerekiyorsa
        if force or prediction_service.check_model_update_needed():
            logger.info("Model update required. Initializing update process.")
            
            # ModelTrainer oluştur ve güncelle
            trainer = ModelTrainer(api_key=Config.OPENFDA_API_KEY)
            
            # Mevcut ilaç listesini kullan
            drug_list = prediction_service.drug_list
            
            # Eğer ilaç listesi boşsa varsayılanı kullan
            if not drug_list:
                drug_list = [
                    'aspirin', 'ibuprofen', 'amoxicillin', 'metformin',
                    'lisinopril', 'atorvastatin', 'fluoxetine', 'omeprazole'
                ]
                logger.info(f"Using default drug list: {drug_list}")
            
            # Modeli eğit
            logger.info("Training model with literature-based risk calculation")
            result = await trainer.train_model(drug_list)
            
            if result:
                logger.info(f"Model successfully updated with accuracy: {result['accuracy']:.4f}")
                return True, result
            else:
                logger.error("Model update failed")
                return False, None
        else:
            logger.info("No model update needed at this time")
            return False, None
            
    except Exception as e:
        logger.error(f"Error during model update: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False, None

async def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description='Update DrugSafeAI model automatically')
    parser.add_argument('--force', action='store_true', help='Force model update regardless of update interval')
    args = parser.parse_args()
    
    logger.info(f"Starting model update script (force={args.force})")
    
    success, result = await update_model(force=args.force)
    
    if success:
        logger.info("Model update completed successfully")
        if result:
            logger.info(f"Training set size: {result.get('n_samples', 0)}")
            logger.info(f"Predictions by drug: {result.get('predictions_by_drug', {})}")
    else:
        logger.warning("Model update did not complete or was not needed")
    
    logger.info("Update script finished")

if __name__ == "__main__":
    asyncio.run(main()) 