#!/usr/bin/env python
"""
DrugSafeAI Otomatik Model Güncelleme Ana Betiği
-----------------------------------------------
Bu betik, DrugSafeAI modelini otomatik olarak güncellemek için kullanılır.
Günlük, haftalık veya aylık olarak ayarlanabilen aralıklarda çalıştırılabilir.

Kullanım:
    python update_model_script.py [--interval daily|weekly|monthly] [--force]

Parametreler:
    --interval     Güncelleme aralığı: daily, weekly, monthly (varsayılan: weekly)
    --force        Zorunlu güncelleme yap (güncelleme gerekli olmazsa bile)
"""

import sys
import asyncio
import logging
import argparse
import time
import datetime
import os
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Günlük dosyası oluştur
log_dir = Path(project_root) / 'logs'
if not log_dir.exists():
    os.makedirs(log_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "model_update.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def perform_update(force=False):
    """Modeli güncelle"""
    try:
        logger.info("Starting model update process")
        
        # Model güncelleme betiğini çalıştır
        from scripts.update_model import update_model
        success, result = await update_model(force=force)
        
        if success:
            logger.info("Model successfully updated")
            if result:
                logger.info(f"Model accuracy: {result.get('accuracy', 0):.4f}")
                logger.info(f"Sample size: {result.get('n_samples', 0)}")
                
                # Son güncelleme zamanını kaydet
                with open(log_dir / "last_update.txt", "w") as f:
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"Last update: {now}\n")
                    f.write(f"Accuracy: {result.get('accuracy', 0):.4f}\n")
                    f.write(f"Samples: {result.get('n_samples', 0)}\n")
            
            return True
        else:
            logger.warning("Model update failed or was not needed")
            return False
            
    except Exception as e:
        logger.error(f"Error during model update: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def calculate_next_run(interval):
    """Bir sonraki çalışma zamanını hesapla"""
    now = datetime.datetime.now()
    
    if interval == 'daily':
        # Her gün saat 02:00'de çalıştır
        next_run = now.replace(hour=2, minute=0, second=0, microsecond=0)
        if next_run <= now:
            next_run = next_run + datetime.timedelta(days=1)
    
    elif interval == 'weekly':
        # Her Pazartesi saat 02:00'de çalıştır
        days_ahead = 0 - now.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        next_run = now.replace(hour=2, minute=0, second=0, microsecond=0) + datetime.timedelta(days=days_ahead)
    
    elif interval == 'monthly':
        # Her ayın 1'i saat 02:00'de çalıştır
        if now.day == 1 and now.hour < 2:
            next_run = now.replace(hour=2, minute=0, second=0, microsecond=0)
        else:
            # Bir sonraki ayın 1'ine git
            if now.month == 12:
                next_run = now.replace(year=now.year+1, month=1, day=1, hour=2, minute=0, second=0, microsecond=0)
            else:
                next_run = now.replace(month=now.month+1, day=1, hour=2, minute=0, second=0, microsecond=0)
    
    return next_run

async def scheduled_updates(interval='weekly', force=False):
    """Belirli bir aralıkta modeli güncelle"""
    logger.info(f"Starting scheduled updates with interval: {interval}")
    
    while True:
        try:
            # Şimdiki zaman
            now = datetime.datetime.now()
            
            # Bir sonraki çalışma zamanını hesapla
            next_run = calculate_next_run(interval)
            seconds_until_next_run = (next_run - now).total_seconds()
            
            # Log bilgisi
            logger.info(f"Next update scheduled for: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Waiting {seconds_until_next_run/3600:.1f} hours until next update")
            
            # Bir sonraki çalışma zamanına kadar bekle
            await asyncio.sleep(seconds_until_next_run)
            
            # Güncelleme yap
            await perform_update(force=force)
            
        except KeyboardInterrupt:
            logger.info("Scheduled updates stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in scheduled updates: {str(e)}")
            logger.error(traceback.format_exc())
            # Hata durumunda 1 saat bekle ve tekrar dene
            await asyncio.sleep(3600)

async def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description='DrugSafeAI Model Update Scheduler')
    parser.add_argument('--interval', choices=['daily', 'weekly', 'monthly'], default='weekly',
                        help='Update interval: daily, weekly, or monthly')
    parser.add_argument('--force', action='store_true', 
                        help='Force model update regardless of update interval')
    parser.add_argument('--now', action='store_true',
                        help='Perform an update immediately and then continue with scheduled updates')
    args = parser.parse_args()
    
    logger.info(f"Starting DrugSafeAI model update scheduler")
    logger.info(f"Update interval: {args.interval}")
    logger.info(f"Force update: {args.force}")
    
    # Eğer --now parametresi verilmişse, hemen bir güncelleme yap
    if args.now:
        logger.info("Performing immediate update before scheduling")
        await perform_update(force=args.force)
    
    # Planlanmış güncellemeleri başlat
    await scheduled_updates(interval=args.interval, force=args.force)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Script terminated by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        import traceback
        logger.error(traceback.format_exc()) 