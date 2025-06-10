import asyncio
from app.models.train_model import ModelTrainer
from config import Config

async def main():
    # Kısaltılmış ilaç listesi - her kategoriden sadece 1-2 ilaç
    drug_list = [
        # Ağrı kesiciler / Antiinflamatuarlar
        'aspirin',
        'ibuprofen',
        
        # Antibiyotikler
        'amoxicillin',
        
        # Diyabet ilaçları
        'metformin',
        
        # Kan basıncı ilaçları
        'lisinopril',
        
        # Kolesterol ilaçları
        'atorvastatin',
        
        # Antidepresanlar
        'fluoxetine',
        
        # Diğer yaygın ilaçlar
        'omeprazole'
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