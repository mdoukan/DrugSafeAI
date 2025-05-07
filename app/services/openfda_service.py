import aiohttp
import asyncio
from typing import List, Dict, Any
import json
from datetime import datetime, timedelta
import logging
from urllib.parse import urlencode

class OpenFDAService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.fda.gov/drug/event.json"
        self.logger = logging.getLogger(__name__)
        self.timeout = aiohttp.ClientTimeout(total=30)  # 30 saniye timeout

    async def fetch_adverse_events(self, drug_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Belirli bir ilaç için yan etki raporlarını OpenFDA API'den çeker
        
        Args:
            drug_name: İlaç adı
            limit: Çekilecek maksimum rapor sayısı
            
        Returns:
            List[Dict]: Yan etki raporları listesi
        """
        try:
            # Son 1 yıllık verileri al
            one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
            
            # URL parametrelerini hazırla
            search_query = f'patient.drug.medicinalproduct:"{drug_name}"'
            params = {
                'api_key': self.api_key,
                'search': search_query,
                'limit': limit
            }
            
            # URL'yi oluştur
            url = f"{self.base_url}?{urlencode(params)}"
            
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('results', [])
                    else:
                        self.logger.error(f"API error: {response.status} - {await response.text()}")
                        return []
                        
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout while fetching data for {drug_name}")
            return []
        except Exception as e:
            self.logger.error(f"Error fetching data for {drug_name}: {str(e)}")
            return []

    async def fetch_drug_info(self, drug_name: str) -> Dict[str, Any]:
        """
        İlaç hakkında detaylı bilgi çeker
        
        Args:
            drug_name: İlaç adı
            
        Returns:
            Dict: İlaç bilgileri
        """
        try:
            # URL parametrelerini hazırla
            search_query = f'medicinalproduct:"{drug_name}"'
            params = {
                'api_key': self.api_key,
                'search': search_query,
                'limit': 1
            }
            
            # URL'yi oluştur
            url = f"https://api.fda.gov/drug/drugsfda.json?{urlencode(params)}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('results', [{}])[0]
                    else:
                        self.logger.error(f"API error: {response.status}")
                        self.logger.error(f"Response: {await response.text()}")
                        return {}
                        
        except Exception as e:
            self.logger.error(f"Error fetching drug info: {str(e)}")
            return {}

    async def fetch_drug_interactions(self, drug_name: str) -> List[Dict[str, Any]]:
        """
        İlaç etkileşimlerini çeker
        
        Args:
            drug_name: İlaç adı
            
        Returns:
            List[Dict]: Etkileşim bilgileri
        """
        try:
            # URL parametrelerini hazırla
            search_query = f'patient.drug.medicinalproduct:"{drug_name}"'
            params = {
                'api_key': self.api_key,
                'search': search_query,
                'limit': 100
            }
            
            # URL'yi oluştur
            url = f"{self.base_url}?{urlencode(params)}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        interactions = []
                        
                        for result in data.get('results', []):
                            drugs = result.get('patient', {}).get('drug', [])
                            if len(drugs) > 1:  # Birden fazla ilaç varsa etkileşim olabilir
                                interactions.append({
                                    'drugs': [drug.get('medicinalproduct') for drug in drugs],
                                    'reactions': result.get('patient', {}).get('reaction', [])
                                })
                        
                        return interactions
                    else:
                        self.logger.error(f"API error: {response.status}")
                        self.logger.error(f"Response: {await response.text()}")
                        return []
                        
        except Exception as e:
            self.logger.error(f"Error fetching drug interactions: {str(e)}")
            return [] 