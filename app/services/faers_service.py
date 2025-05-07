import requests
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from config import Config

class FAERSService:
    def __init__(self):
        self.base_url = Config.FAERS_API_BASE_URL
        self.api_key = Config.FAERS_API_KEY

    async def fetch_adverse_events(self, drug_name):
        """
        Fetch adverse events for a specific drug from FAERS
        """
        params = {
            'api_key': self.api_key,
            'search': f'patient.drug.medicinalproduct:{drug_name}',
            'limit': 100
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'error' in data:
                print(f"API Error for {drug_name}: {data['error']}")
                return None
            
            if 'results' not in data or not data['results']:
                print(f"No results found for {drug_name}")
                return None
            
            return data
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from FAERS for {drug_name}: {e}")
            return None
        except ValueError as e:
            print(f"Error parsing JSON response for {drug_name}: {e}")
            return None 