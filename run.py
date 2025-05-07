from app import create_app
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting DrugSafeAI application")
        app = create_app()
        
        # Check if model file exists
        model_path = Path(__file__).parent / 'app' / 'models' / 'xgboost_model.pkl'
        if not model_path.exists():
            logger.error(f"Model file not found at: {model_path}")
            sys.exit(1)
            
        logger.info(f"Model file found at: {model_path}")
        logger.info(f"Model file size: {model_path.stat().st_size} bytes")
        
        # Start the server
        logger.info("Starting Flask server")
        app.run(host='0.0.0.0', port=5000, debug=True)
        
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 