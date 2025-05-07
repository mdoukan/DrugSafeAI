import asyncio
from app.models.train_model import ModelTrainer
from config import Config

async def main():
    # Test için örnek ilaç listesi
    drug_list = [
        'aspirin',
        'ibuprofen',
        'paracetamol',
        'metformin',
        'amoxicillin'
    ]

    trainer = ModelTrainer(api_key=Config.OPENFDA_API_KEY)
    results = await trainer.train_model(drug_list)
    
    if results:
        print("Model training completed successfully!")
        print(f"Training accuracy: {results['accuracy']:.4f}")
        print(f"Best parameters: {results['best_params']}")
        print(f"Number of samples: {results['n_samples']}")
    else:
        print("Model training failed!")

if __name__ == "__main__":
    asyncio.run(main()) 