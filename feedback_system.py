import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
import logging

class DrugFeedbackSystem:
    """
    System to collect and manage user feedback on drug information accuracy
    """
    
    def __init__(self, db_path: str = "drug_feedback.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """
        Initialize SQLite database for storing feedback
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drug_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                drug_name TEXT NOT NULL,
                alternative_name TEXT,
                feedback_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT NOT NULL,
                user_evidence TEXT,
                user_credentials TEXT,
                status TEXT DEFAULT 'pending',
                reviewed_by TEXT,
                review_notes TEXT,
                review_date TEXT
            )
        ''')
        
        # Create feedback_analytics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                drug_name TEXT NOT NULL,
                total_feedback INTEGER DEFAULT 0,
                critical_feedback INTEGER DEFAULT 0,
                resolved_feedback INTEGER DEFAULT 0,
                last_updated TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def submit_feedback(self, 
                       drug_name: str,
                       feedback_type: str,
                       severity: str,
                       description: str,
                       alternative_name: str = None,
                       user_evidence: str = None,
                       user_credentials: str = None) -> int:
        """
        Submit user feedback about drug information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO drug_feedback 
                (timestamp, drug_name, alternative_name, feedback_type, severity, 
                 description, user_evidence, user_credentials, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                drug_name,
                alternative_name,
                feedback_type,
                severity,
                description,
                user_evidence,
                user_credentials,
                'pending'
            ))
            
            feedback_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            self.logger.info(f"Feedback submitted: ID {feedback_id} for {drug_name}")
            return feedback_id
            
        except Exception as e:
            self.logger.error(f"Error submitting feedback: {e}")
            return None
    
    def get_feedback_by_drug(self, drug_name: str, status: str = None) -> List[Dict]:
        """
        Get all feedback for a specific drug
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT * FROM drug_feedback 
                WHERE drug_name = ?
            '''
            params = [drug_name]
            
            if status:
                query += ' AND status = ?'
                params.append(status)
            
            query += ' ORDER BY timestamp DESC'
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            feedback_list = []
            for row in rows:
                feedback_list.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'drug_name': row[2],
                    'alternative_name': row[3],
                    'feedback_type': row[4],
                    'severity': row[5],
                    'description': row[6],
                    'user_evidence': row[7],
                    'user_credentials': row[8],
                    'status': row[9],
                    'reviewed_by': row[10],
                    'review_notes': row[11],
                    'review_date': row[12]
                })
            
            conn.close()
            return feedback_list
            
        except Exception as e:
            self.logger.error(f"Error getting feedback: {e}")
            return []
    
    def get_critical_feedback(self) -> List[Dict]:
        """
        Get all critical feedback that needs immediate attention
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM drug_feedback 
                WHERE severity = 'critical' AND status = 'pending'
                ORDER BY timestamp DESC
            ''')
            
            rows = cursor.fetchall()
            feedback_list = []
            for row in rows:
                feedback_list.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'drug_name': row[2],
                    'alternative_name': row[3],
                    'feedback_type': row[4],
                    'severity': row[5],
                    'description': row[6],
                    'user_evidence': row[7],
                    'user_credentials': row[8],
                    'status': row[9]
                })
            
            conn.close()
            return feedback_list
            
        except Exception as e:
            self.logger.error(f"Error getting critical feedback: {e}")
            return []
    
    def get_feedback_by_severity(self, severity: str) -> List[Dict]:
        """
        Get feedback by severity level
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM drug_feedback 
                WHERE severity = ? AND status = 'pending'
                ORDER BY timestamp DESC
            ''', (severity,))
            
            rows = cursor.fetchall()
            feedback_list = []
            for row in rows:
                feedback_list.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'drug_name': row[2],
                    'alternative_name': row[3],
                    'feedback_type': row[4],
                    'severity': row[5],
                    'description': row[6],
                    'user_evidence': row[7],
                    'user_credentials': row[8],
                    'status': row[9],
                    'reviewed_by': row[10],
                    'review_notes': row[11],
                    'review_date': row[12]
                })
            
            conn.close()
            return feedback_list
            
        except Exception as e:
            self.logger.error(f"Error getting feedback by severity: {e}")
            return []
    
    def review_feedback(self, feedback_id: int, status: str, review_notes: str, reviewed_by: str):
        """
        Review and update feedback status
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE drug_feedback 
                SET status = ?, review_notes = ?, reviewed_by = ?, review_date = ?
                WHERE id = ?
            ''', (status, review_notes, reviewed_by, datetime.now().isoformat(), feedback_id))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Feedback {feedback_id} reviewed by {reviewed_by}")
            
        except Exception as e:
            self.logger.error(f"Error reviewing feedback: {e}")
    
    def get_feedback_analytics(self) -> Dict:
        """
        Get analytics about feedback patterns
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get overall statistics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical,
                    COUNT(CASE WHEN status = 'resolved' THEN 1 END) as resolved,
                    COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending
                FROM drug_feedback
            ''')
            
            overall_stats = cursor.fetchone()
            
            # Get feedback by drug
            cursor.execute('''
                SELECT drug_name, COUNT(*) as count
                FROM drug_feedback
                GROUP BY drug_name
                ORDER BY count DESC
                LIMIT 10
            ''')
            
            drug_stats = cursor.fetchall()
            
            # Get feedback by type
            cursor.execute('''
                SELECT feedback_type, COUNT(*) as count
                FROM drug_feedback
                GROUP BY feedback_type
                ORDER BY count DESC
            ''')
            
            type_stats = cursor.fetchall()
            
            conn.close()
            
            return {
                'overall': {
                    'total': overall_stats[0],
                    'critical': overall_stats[1],
                    'resolved': overall_stats[2],
                    'pending': overall_stats[3]
                },
                'by_drug': [{'drug': drug, 'count': count} for drug, count in drug_stats],
                'by_type': [{'type': ftype, 'count': count} for ftype, count in type_stats]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting analytics: {e}")
            return {}
    
    def export_feedback_report(self, filename: str = None) -> str:
        """
        Export feedback data to JSON report
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drug_feedback_report_{timestamp}.json"
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all feedback
            cursor.execute('SELECT * FROM drug_feedback ORDER BY timestamp DESC')
            rows = cursor.fetchall()
            
            feedback_data = []
            for row in rows:
                feedback_data.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'drug_name': row[2],
                    'alternative_name': row[3],
                    'feedback_type': row[4],
                    'severity': row[5],
                    'description': row[6],
                    'user_evidence': row[7],
                    'user_credentials': row[8],
                    'status': row[9],
                    'reviewed_by': row[10],
                    'review_notes': row[11],
                    'review_date': row[12]
                })
            
            # Get analytics
            analytics = self.get_feedback_analytics()
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'analytics': analytics,
                'feedback': feedback_data
            }
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            conn.close()
            self.logger.info(f"Feedback report exported to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error exporting report: {e}")
            return None

# Web interface for feedback submission
class FeedbackWebInterface:
    """
    Web interface for users to submit feedback
    """
    
    def __init__(self, feedback_system: DrugFeedbackSystem):
        self.feedback_system = feedback_system
    
    def create_feedback_form_html(self) -> str:
        """
        Create HTML form for feedback submission
        """
        return '''
        <div class="feedback-form">
            <h3>Report Drug Information Issue</h3>
            <form id="drugFeedbackForm">
                <div class="form-group">
                    <label for="drugName">Drug Name:</label>
                    <input type="text" id="drugName" name="drugName" required>
                </div>
                
                <div class="form-group">
                    <label for="alternativeName">Alternative Drug (if applicable):</label>
                    <input type="text" id="alternativeName" name="alternativeName">
                </div>
                
                <div class="form-group">
                    <label for="feedbackType">Issue Type:</label>
                    <select id="feedbackType" name="feedbackType" required>
                        <option value="">Select issue type</option>
                        <option value="incorrect_risk_factor">Incorrect Risk Factor</option>
                        <option value="outdated_evidence">Outdated Evidence</option>
                        <option value="missing_alternative">Missing Alternative</option>
                        <option value="incorrect_interaction">Incorrect Drug Interaction</option>
                        <option value="other">Other</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="severity">Severity:</label>
                    <select id="severity" name="severity" required>
                        <option value="">Select severity</option>
                        <option value="critical">Critical (Patient Safety Risk)</option>
                        <option value="high">High (Significant Error)</option>
                        <option value="medium">Medium (Minor Error)</option>
                        <option value="low">Low (Suggestion)</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="description">Description:</label>
                    <textarea id="description" name="description" rows="4" required 
                              placeholder="Please describe the issue in detail..."></textarea>
                </div>
                
                <div class="form-group">
                    <label for="userEvidence">Supporting Evidence (optional):</label>
                    <textarea id="userEvidence" name="userEvidence" rows="3" 
                              placeholder="Please provide any supporting evidence, references, or links..."></textarea>
                </div>
                
                <div class="form-group">
                    <label for="userCredentials">Your Credentials (optional):</label>
                    <input type="text" id="userCredentials" name="userCredentials" 
                           placeholder="e.g., MD, PharmD, RN, etc.">
                </div>
                
                <button type="submit" class="btn btn-primary">Submit Feedback</button>
            </form>
        </div>
        '''
    
    def create_feedback_javascript(self) -> str:
        """
        Create JavaScript for handling feedback form submission
        """
        return '''
        <script>
        document.getElementById('drugFeedbackForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const feedbackData = {
                drug_name: formData.get('drugName'),
                alternative_name: formData.get('alternativeName') || null,
                feedback_type: formData.get('feedbackType'),
                severity: formData.get('severity'),
                description: formData.get('description'),
                user_evidence: formData.get('userEvidence') || null,
                user_credentials: formData.get('userCredentials') || null
            };
            
            // Submit feedback via AJAX
            fetch('/api/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(feedbackData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Thank you for your feedback! We will review it shortly.');
                    e.target.reset();
                } else {
                    alert('Error submitting feedback: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error submitting feedback. Please try again.');
            });
        });
        </script>
        '''

# Usage example
if __name__ == "__main__":
    # Initialize feedback system
    feedback_system = DrugFeedbackSystem()
    
    # Submit sample feedback
    feedback_id = feedback_system.submit_feedback(
        drug_name="aspirin",
        feedback_type="incorrect_risk_factor",
        severity="high",
        description="The risk factor for Naproxen (0.85) seems too low based on recent studies.",
        alternative_name="Naproxen",
        user_evidence="Recent JAMA study 2023 shows higher bleeding risk",
        user_credentials="MD"
    )
    
    print(f"Sample feedback submitted with ID: {feedback_id}")
    
    # Get critical feedback
    critical_feedback = feedback_system.get_critical_feedback()
    print(f"Critical feedback count: {len(critical_feedback)}")
    
    # Get analytics
    analytics = feedback_system.get_feedback_analytics()
    print("Feedback Analytics:", analytics)
    
    # Export report
    report_file = feedback_system.export_feedback_report()
    print(f"Feedback report exported to: {report_file}") 