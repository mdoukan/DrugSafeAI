import requests
import json
import time
from typing import Dict, List, Optional

class DrugDataAPI:
    """
    Drug information API integration for more accurate and up-to-date data
    """
    
    def __init__(self):
        self.fda_base_url = "https://api.fda.gov/drug"
        self.pubmed_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.drugbank_base_url = "https://go.drugbank.com/api/v1"
        
    def get_fda_drug_info(self, drug_name: str) -> Optional[Dict]:
        """
        Fetch drug information from FDA API
        """
        try:
            url = f"{self.fda_base_url}/label.json"
            params = {
                'search': f'openfda.generic_name:"{drug_name}"',
                'limit': 1
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    return data['results'][0]
            return None
        except Exception as e:
            print(f"FDA API error: {e}")
            return None
    
    def search_pubmed_articles(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search PubMed for recent articles about drug interactions
        """
        try:
            # Search for articles
            search_url = f"{self.pubmed_base_url}/esearch.fcgi"
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'json',
                'sort': 'relevance'
            }
            
            response = requests.get(search_url, params=search_params)
            if response.status_code == 200:
                data = response.json()
                article_ids = data.get('esearchresult', {}).get('idlist', [])
                
                # Get article details
                articles = []
                for pmid in article_ids:
                    fetch_url = f"{self.pubmed_base_url}/efetch.fcgi"
                    fetch_params = {
                        'db': 'pubmed',
                        'id': pmid,
                        'retmode': 'xml'
                    }
                    
                    article_response = requests.get(fetch_url, params=fetch_params)
                    if article_response.status_code == 200:
                        # Parse XML and extract relevant info
                        articles.append({
                            'pmid': pmid,
                            'title': self._extract_title(article_response.text),
                            'abstract': self._extract_abstract(article_response.text),
                            'date': self._extract_date(article_response.text)
                        })
                    
                    time.sleep(0.1)  # Be respectful to API
                
                return articles
            return []
        except Exception as e:
            print(f"PubMed API error: {e}")
            return []
    
    def _extract_title(self, xml_content: str) -> str:
        """Extract title from PubMed XML"""
        # Simple XML parsing - in production use proper XML parser
        if '<ArticleTitle>' in xml_content:
            start = xml_content.find('<ArticleTitle>') + len('<ArticleTitle>')
            end = xml_content.find('</ArticleTitle>')
            return xml_content[start:end].strip()
        return ""
    
    def _extract_abstract(self, xml_content: str) -> str:
        """Extract abstract from PubMed XML"""
        if '<AbstractText>' in xml_content:
            start = xml_content.find('<AbstractText>') + len('<AbstractText>')
            end = xml_content.find('</AbstractText>')
            return xml_content[start:end].strip()
        return ""
    
    def _extract_date(self, xml_content: str) -> str:
        """Extract publication date from PubMed XML"""
        if '<PubDate>' in xml_content:
            # Extract year from PubDate
            year_start = xml_content.find('<Year>') + len('<Year>')
            year_end = xml_content.find('</Year>')
            if year_start > len('<Year>') and year_end > year_start:
                return xml_content[year_start:year_end]
        return ""
    
    def get_drug_interactions(self, drug_name: str) -> List[Dict]:
        """
        Get drug interactions from multiple sources
        """
        interactions = []
        
        # FDA data
        fda_data = self.get_fda_drug_info(drug_name)
        if fda_data:
            # Extract drug interactions from FDA label
            drug_interactions = fda_data.get('drug_interactions', [])
            for interaction in drug_interactions:
                interactions.append({
                    'source': 'FDA',
                    'drug': drug_name,
                    'interaction': interaction,
                    'severity': 'Unknown'  # FDA doesn't always provide severity
                })
        
        # PubMed articles about interactions
        query = f'"{drug_name}" AND "drug interaction" AND "risk"'
        articles = self.search_pubmed_articles(query, max_results=3)
        
        for article in articles:
            interactions.append({
                'source': 'PubMed',
                'drug': drug_name,
                'pmid': article['pmid'],
                'title': article['title'],
                'abstract': article['abstract'][:200] + "..." if len(article['abstract']) > 200 else article['abstract'],
                'date': article['date']
            })
        
        return interactions
    
    def get_alternative_drugs(self, drug_name: str, condition: str = None) -> List[Dict]:
        """
        Get alternative drugs based on recent literature
        """
        alternatives = []
        
        # Search for alternative treatments
        query = f'"{drug_name}" AND "alternative" AND "treatment"'
        if condition:
            query += f' AND "{condition}"'
        
        articles = self.search_pubmed_articles(query, max_results=5)
        
        for article in articles:
            alternatives.append({
                'source': 'PubMed',
                'original_drug': drug_name,
                'pmid': article['pmid'],
                'title': article['title'],
                'abstract': article['abstract'][:300] + "..." if len(article['abstract']) > 300 else article['abstract'],
                'date': article['date'],
                'evidence_level': 'Literature Review'
            })
        
        return alternatives

# Usage example
if __name__ == "__main__":
    api = DrugDataAPI()
    
    # Test with aspirin
    print("=== Testing Drug Data API ===")
    
    # Get FDA info
    fda_info = api.get_fda_drug_info("aspirin")
    if fda_info:
        print("FDA Info found for aspirin")
    
    # Get interactions
    interactions = api.get_drug_interactions("aspirin")
    print(f"Found {len(interactions)} interactions for aspirin")
    
    # Get alternatives
    alternatives = api.get_alternative_drugs("aspirin", "pain")
    print(f"Found {len(alternatives)} alternative articles for aspirin")
    
    # Save to JSON for review
    with open('drug_data_validation.json', 'w') as f:
        json.dump({
            'fda_info': fda_info,
            'interactions': interactions,
            'alternatives': alternatives
        }, f, indent=2)
    
    print("Data saved to drug_data_validation.json") 