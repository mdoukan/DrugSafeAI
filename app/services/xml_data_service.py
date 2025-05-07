import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

class XMLDataService:
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent.parent / 'datasets'

    def read_adverse_events(self, drug_name):
        """
        XML dosyalarından yan etki verilerini oku
        """
        all_events = []
        
        try:
            # datasets klasöründeki tüm XML dosyalarını tara
            for xml_file in self.data_dir.glob('*.xml'):
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # İlgili ilaca ait kayıtları bul
                for event in root.findall('.//safetyreport'):
                    drugs = event.findall('.//drug/medicinalproduct')
                    if any(d.text and d.text.lower() == drug_name.lower() for d in drugs):
                        event_data = self._parse_event(event)
                        if event_data:
                            all_events.append(event_data)
        
        except Exception as e:
            print(f"Error reading XML files for {drug_name}: {e}")
        
        return {'results': all_events}

    def _parse_event(self, event):
        """
        XML'den detaylı yan etki olayını parse et
        """
        try:
            # Temel rapor bilgileri
            report_data = {
                'safetyreportid': self._safe_get_value(event, './/safetyreportid'),
                'primarysourcecountry': self._safe_get_value(event, './/primarysourcecountry'),
                'serious': 1 if event.find('.//serious').text == '1' else 0,
                'seriousnesshospitalization': self._safe_get_value(event, './/seriousnesshospitalization'),
                'seriousnessdeath': self._safe_get_value(event, './/seriousnessdeath'),
                'receivedate': self._parse_date(self._safe_get_value(event, './/receivedate'))
            }

            # Hasta bilgileri
            patient = event.find('.//patient')
            if patient is None:
                return None

            patient_data = {
                'patientonsetage': self._safe_convert_float(self._safe_get_value(patient, './/patientonsetage')),
                'patientweight': self._safe_convert_float(self._safe_get_value(patient, './/patientweight')),
                'patientheight': self._safe_convert_float(self._safe_get_value(patient, './/patientheight')),
                'patientsex': self._safe_convert_int(self._safe_get_value(patient, './/patientsex')),
                'medical_history': [c.text for c in patient.findall('.//patientmedicalhistory/condition') if c.text]
            }

            # Yan etki bilgileri
            reactions = []
            for reaction in patient.findall('.//reaction'):
                reaction_data = {
                    'reactionmeddrapt': self._safe_get_value(reaction, './/reactionmeddrapt'),
                    'reactionoutcome': self._safe_convert_int(self._safe_get_value(reaction, './/reactionoutcome')),
                    'reactionstartdate': self._parse_date(self._safe_get_value(reaction, './/reactionstartdate')),
                    'reactionenddate': self._parse_date(self._safe_get_value(reaction, './/reactionenddate')),
                    'reactionseverity': self._safe_get_value(reaction, './/reactionseverity')
                }
                reactions.append(reaction_data)

            # İlaç bilgileri
            drugs = []
            for drug in patient.findall('.//drug'):
                drug_data = {
                    'medicinalproduct': self._safe_get_value(drug, './/medicinalproduct'),
                    'drugindication': self._safe_get_value(drug, './/drugindication'),
                    'drugstartdate': self._parse_date(self._safe_get_value(drug, './/drugstartdate')),
                    'drugenddate': self._parse_date(self._safe_get_value(drug, './/drugenddate')),
                    'drugdosagetext': self._safe_get_value(drug, './/drugdosagetext'),
                    'drugadministrationroute': self._safe_get_value(drug, './/drugadministrationroute'),
                    'drugcharacterization': self._safe_convert_int(self._safe_get_value(drug, './/drugcharacterization')),
                    'activesubstance': self._safe_get_value(drug, './/activesubstance')
                }
                drugs.append(drug_data)

            # Laboratuvar test sonuçları
            labs = []
            for lab in patient.findall('.//laboratory'):
                lab_data = {
                    'testname': self._safe_get_value(lab, './/testname'),
                    'testresult': self._safe_get_value(lab, './/testresult'),
                    'testunit': self._safe_get_value(lab, './/testunit'),
                    'testdate': self._parse_date(self._safe_get_value(lab, './/testdate'))
                }
                labs.append(lab_data)

            return {
                'report': report_data,
                'patient': patient_data,
                'reactions': reactions,
                'drugs': drugs,
                'laboratory_tests': labs
            }

        except Exception as e:
            print(f"Error parsing event: {e}")
            return None

    def _safe_get_value(self, element, xpath):
        """
        XML elementinden güvenli bir şekilde değer al
        """
        try:
            found = element.find(xpath)
            return found.text.strip() if found is not None and found.text else None
        except:
            return None

    def _safe_convert_float(self, value):
        """
        Güvenli float dönüşümü
        """
        try:
            return float(value) if value is not None else None
        except (ValueError, TypeError):
            return None

    def _safe_convert_int(self, value):
        """
        Güvenli integer dönüşümü
        """
        try:
            return int(float(value)) if value is not None else None
        except (ValueError, TypeError):
            return None

    def _parse_date(self, date_str):
        """
        Tarih string'ini datetime objesine çevir
        """
        try:
            if date_str and len(date_str) == 8:
                return datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
            return None
        except ValueError:
            return None 