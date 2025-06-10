from flask import Flask
from config import Config
import logging
from app.models.database import db

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app(config_class=Config):
    try:
        logger.info("Creating Flask application")
        app = Flask(__name__)
        app.config.from_object(config_class)
        
        # Initialize database
        logger.info("Initializing database")
        db.init_app(app)
        
        # Create database tables
        with app.app_context():
            db.create_all()
            logger.info("Database tables created")
        
        logger.info("Initializing services")
        # Initialize services
        from app.services import faers_service, prediction_service
        
        logger.info("Registering blueprints")
        # Register blueprints
        from app.routes import main
        app.register_blueprint(main)
        
        # Register user routes blueprint
        from app.routes_user import user_bp
        app.register_blueprint(user_bp)
        
        logger.info("Flask application created successfully")
        return app
        
    except Exception as e:
        logger.error(f"Error creating Flask application: {str(e)}")
        raise 