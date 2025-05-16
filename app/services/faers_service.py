import requests
import sys
import logging
from pathlib import Path
import random
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from config import Config

class FaersService:
    def __init__(self):
        self.api_key = Config.FAERS_API_KEY or Config.OPENFDA_API_KEY
        self.base_url = Config.FAERS_API_BASE_URL
        self.logger = logging.getLogger(__name__)
        self.cache = {}  # Sonuçları önbelleğe al
        self.cache_expiry = {}  # Önbellek süre sonu
        self.cache_duration = 86400  # Saniye cinsinden (24 saat)

    def fetch_adverse_events(self, drug_name):
        """
        Fetch adverse events for a specific drug from FAERS
        """
        params = {
            'api_key': self.api_key,
            'search': f'patient.drug.medicinalproduct:{drug_name}',
            'limit': 100
        }
        
        try:
            self.logger.info(f"Fetching adverse events for {drug_name}")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'error' in data:
                self.logger.error(f"API Error for {drug_name}: {data['error']}")
                return None
            
            if 'results' not in data or not data['results']:
                self.logger.warning(f"No results found for {drug_name}")
                return None
            
            return data
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching data from FAERS for {drug_name}: {e}")
            return None
        except ValueError as e:
            self.logger.error(f"Error parsing JSON response for {drug_name}: {e}")
            return None

    def get_drug_side_effects(self, drug_name):
        """
        İlaç için OpenFDA API'den yan etkileri getirir ve formatlayarak döndürür
        """
        # Önbellekte varsa ve süresi dolmamışsa kullan
        cache_key = f"side_effects_{drug_name}"
        if cache_key in self.cache and time.time() < self.cache_expiry.get(cache_key, 0):
            self.logger.info(f"Using cached side effects for {drug_name}")
            return self.cache[cache_key]
            
        try:
            # OpenFDA API URL
            url = self.base_url
            
            # API parametreleri
            params = {
                'api_key': self.api_key,
                'search': f'patient.drug.medicinalproduct:"{drug_name}" AND serious:1',
                'count': 'patient.reaction.reactionmeddrapt.exact',
                'limit': 15  # En sık görülen 15 yan etki
            }
            
            self.logger.info(f"Fetching side effects for {drug_name}")
            response = requests.get(url, params=params)
            
            # API yanıtını kontrol et
            if response.status_code == 200:
                data = response.json()
                side_effects = []
                
                # Yan etkileri formatla
                if 'results' in data:
                    for result in data['results']:
                        effect_name = result.get('term', '')
                        count = result.get('count', 0)
                        
                        # Yan etki şiddetini belirle
                        severity = self._determine_severity(effect_name)
                        
                        # Olasılık hesapla
                        probability = min(count / 1000, 0.9)  # 1000 vakaya oranla, max 0.9
                        
                        side_effects.append({
                            'name': effect_name,
                            'probability': probability,
                            'severity': severity,
                            'description': self._get_effect_description(effect_name),
                            'recommendation': self._get_recommendation(effect_name, severity)
                        })
                
                # Sonuçları önbelleğe al
                self.cache[cache_key] = side_effects
                self.cache_expiry[cache_key] = time.time() + self.cache_duration
                
                return side_effects
            else:
                self.logger.error(f"API error: {response.status_code} - {response.text}")
                return self._get_fallback_side_effects(drug_name)
                
        except Exception as e:
            self.logger.error(f"Error fetching side effects for {drug_name}: {str(e)}")
            return self._get_fallback_side_effects(drug_name)
    
    def _determine_severity(self, effect_name):
        """
        Yan etki adına göre şiddet seviyesini belirler
        Gerçek bir sistem, Medra kod sistemini ve listeleri kullanmalıdır
        """
        # Yüksek riskli yan etkiler listesi
        high_severity = [
            'death', 'fatal', 'stroke', 'heart attack', 'cardiac arrest', 'seizure', 
            'anaphylaxis', 'anaphylactic', 'hemorrhage', 'bleeding', 
            'liver failure', 'kidney failure', 'renal failure', 'respiratory failure',
            'suicide', 'coma', 'blindness', 'heart failure', 'hepatic failure',
            'stevens-johnson', 'toxic epidermal', 'thrombosis', 'embolism',
            'sepsis', 'septic', 'shock', 'paralysis'
        ]
        
        # Orta riskli yan etkiler
        medium_severity = [
            'infection', 'diarrhea', 'rash', 'nausea', 'vomiting', 'headache', 'dizziness',
            'fatigue', 'pain', 'allergy', 'allergic', 'weakness', 'inflammation', 'edema',
            'swelling', 'dysfunction', 'insufficiency', 'hypertension', 'hypotension',
            'tachycardia', 'bradycardia', 'insomnia', 'depression', 'anxiety', 'confusion'
        ]
        
        # Küçük harfe çevir
        effect_lower = effect_name.lower()
        
        # Tıbbi terminoloji sözlüğüne göre şiddet belirle
        for term in high_severity:
            if term in effect_lower:
                return 'High'
                
        for term in medium_severity:
            if term in effect_lower:
                return 'Medium'
                
        return 'Low'
    
    def _get_effect_description(self, effect_name):
        """
        Yan etki için tıbbi açıklama döndürür
        Gerçek bir sistem, resmi tıbbi açıklamalarla bir veritabanı kullanmalı
        """
        # Yan etki açıklamaları veritabanı benzeri
        descriptions = {
            'Rash': 'Skin irritation that may appear as redness, spots, or itchy areas',
            'Nausea': 'A feeling of sickness with an inclination to vomit',
            'Headache': 'Pain in any region of the head',
            'Dizziness': 'Feeling lightheaded or unsteady',
            'Diarrhea': 'Loose, watery stools occurring more frequently than usual',
            'Vomiting': 'Forceful expulsion of stomach contents through the mouth',
            'Abdominal pain': 'Pain that occurs between the chest and pelvic regions',
            'Fatigue': 'Extreme tiredness resulting from mental or physical exertion',
            'Insomnia': 'Persistent problems falling and staying asleep',
            'Anaphylaxis': 'Severe, potentially life-threatening allergic reaction',
            'Hepatic failure': 'Deterioration of liver function',
            'Renal failure': 'Kidneys can no longer filter waste products from the blood'
        }
        
        # Yan etki açıklamasını döndür
        for key, desc in descriptions.items():
            if key.lower() in effect_name.lower():
                return desc
        
        # Eşleşme bulunamazsa genel açıklama döndür
        return f"A reported adverse reaction to medication"

    def _get_recommendation(self, effect_name, severity):
        """
        Yan etki ve şiddetine göre öneriler döndürür
        Gerçek bir sistem, klinik veya FDA önerilerine dayalı bir veritabanı kullanmalı
        """
        # Yüksek şiddetli yan etkiler için öneriler
        if severity == 'High':
            return "Stop medication immediately and seek medical attention"
        
        # Özel yan etki önerileri
        if 'rash' in effect_name.lower():
            return "Discontinue medication and consult healthcare provider; may indicate allergic reaction"
        elif 'nausea' in effect_name.lower() or 'vomiting' in effect_name.lower():
            return "Take medication with food; consult doctor if persistent or severe"
        elif 'headache' in effect_name.lower():
            return "Usually temporary; consult doctor if severe or persistent"
        elif 'dizziness' in effect_name.lower():
            return "Change positions slowly; avoid driving or operating machinery"
        elif 'diarrhea' in effect_name.lower():
            return "Stay hydrated; consult doctor if severe or persistent for more than 2 days"
        elif 'pain' in effect_name.lower():
            return "Consult healthcare provider if pain is severe or persistent"
        elif 'insomnia' in effect_name.lower() or 'sleep' in effect_name.lower():
            return "Take medication in the morning; avoid caffeine; consult doctor if persistent"
        
        # Orta şiddetli yan etkiler için genel öneriler
        if severity == 'Medium':
            return "Monitor symptoms; consult healthcare provider if symptoms worsen or persist"
        
        # Düşük şiddetli yan etkiler için genel öneriler
        return "Usually resolves without intervention; consult healthcare provider if concerning"
    
    def _get_fallback_side_effects(self, drug_name):
        """
        API çağrısı başarısız olduğunda fallback yan etki verileri
        """
        # İlaç adına göre genel yan etki verileri
        drug_name_lower = drug_name.lower()
        
        # Yaygın ilaçlar için özel yan etkiler
        if 'aspirin' in drug_name_lower:
            return [
                {'name': 'Stomach Upset', 'probability': 0.45, 'severity': 'Medium', 
                 'description': 'Stomach irritation or discomfort', 
                 'recommendation': 'Take with food to reduce stomach upset'},
                {'name': 'Bleeding', 'probability': 0.30, 'severity': 'High', 
                 'description': 'Increased risk of bleeding due to blood thinning properties', 
                 'recommendation': 'Stop medication and consult doctor if unusual bleeding occurs'},
                {'name': 'Heartburn', 'probability': 0.25, 'severity': 'Low', 
                 'description': 'Burning sensation in chest', 
                 'recommendation': 'Take with plenty of water'}
            ]
        elif 'ibuprofen' in drug_name_lower:
            return [
                {'name': 'Stomach Pain', 'probability': 0.40, 'severity': 'Medium', 
                 'description': 'Pain or discomfort in stomach', 
                 'recommendation': 'Take with food to reduce stomach irritation'},
                {'name': 'Dizziness', 'probability': 0.20, 'severity': 'Medium', 
                 'description': 'Feeling lightheaded or unsteady', 
                 'recommendation': 'Avoid driving or operating machinery if affected'},
                {'name': 'Headache', 'probability': 0.15, 'severity': 'Low', 
                 'description': 'Pain in the head or temples', 
                 'recommendation': 'Usually temporary; consult doctor if persistent'}
            ]
        elif 'amoxicillin' in drug_name_lower:
            return [
                {'name': 'Diarrhea', 'probability': 0.35, 'severity': 'Medium', 
                 'description': 'Loose, watery stools', 
                 'recommendation': 'Stay hydrated and consult doctor if severe'},
                {'name': 'Rash', 'probability': 0.25, 'severity': 'Medium', 
                 'description': 'Skin eruption or discoloration', 
                 'recommendation': 'Stop medication and contact doctor immediately'},
                {'name': 'Nausea', 'probability': 0.20, 'severity': 'Low', 
                 'description': 'Feeling of sickness with an inclination to vomit', 
                 'recommendation': 'Take with food to reduce nausea'}
            ]
        # Diğer yaygın ilaçlar için yan etkiler eklenebilir
        
        # Jenerik fallback
        return [
            {'name': 'Common Side Effect', 'probability': 0.30, 'severity': 'Medium', 
             'description': f'A frequently reported adverse reaction to {drug_name}', 
             'recommendation': 'Monitor symptoms and consult doctor if they persist'},
            {'name': 'Mild Reaction', 'probability': 0.20, 'severity': 'Low', 
             'description': f'A less common reaction to {drug_name}', 
             'recommendation': 'Usually resolves without intervention'},
            {'name': 'Rare Side Effect', 'probability': 0.10, 'severity': 'High', 
             'description': f'An uncommon but serious reaction to {drug_name}', 
             'recommendation': 'Seek medical attention immediately if this occurs'}
        ] 