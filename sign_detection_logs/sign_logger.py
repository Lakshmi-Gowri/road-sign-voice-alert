import csv
import os
import datetime
from pathlib import Path

class SignDetectionLogger:
    def __init__(self, log_dir="sign_detection_logs"):
        """Initialize the logger with specified directory"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create log file with current date
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        self.log_file = self.log_dir / f"sign_detections_{today}.csv"
        
        # Create CSV file with headers if it doesn't exist
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Create CSV file with headers if it doesn't exist"""
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'Timestamp',
                    'Date',
                    'Time',
                    'Detected_Sign',
                    'Confidence',
                    'Detection_Source',
                    'Language',
                    'Location_Info',
                    'Session_ID'
                ])
    
    def log_detection(self, sign_name, confidence, source="Unknown", language="English", location="Not Available"):
        """
        Log a detected sign with all relevant information
        
        Args:
            sign_name (str): Name of the detected road sign
            confidence (float): Detection confidence (0-1)
            source (str): Source of detection (Camera, Image, Video)
            language (str): Current language setting
            location (str): Location information if available
        """
        try:
            now = datetime.datetime.now()
            session_id = now.strftime("%Y%m%d_%H%M%S")
            
            log_entry = [
                now.isoformat(),  # Full timestamp
                now.strftime("%Y-%m-%d"),  # Date
                now.strftime("%H:%M:%S"),  # Time
                sign_name,
                f"{confidence:.4f}",
                source,
                language,
                location,
                session_id
            ]
            
            with open(self.log_file, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(log_entry)
            
            print(f"Logged detection: {sign_name} ({confidence:.2%}) from {source}")
            
        except Exception as e:
            print(f"Error logging detection: {e}")
    
    def get_log_file_path(self):
        """Return the current log file path"""
        return str(self.log_file)
    
    def get_all_log_files(self):
        """Return list of all log files in the directory"""
        return list(self.log_dir.glob("sign_detections_*.csv"))
    
    def get_detection_stats(self):
        """Get statistics from current log file"""
        try:
            if not self.log_file.exists():
                return {"total_detections": 0, "unique_signs": 0, "sources": {}}
            
            stats = {
                "total_detections": 0,
                "unique_signs": set(),
                "sources": {},
                "languages": {},
                "avg_confidence": 0
            }
            
            confidences = []
            
            with open(self.log_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    stats["total_detections"] += 1
                    stats["unique_signs"].add(row["Detected_Sign"])
                    
                    source = row["Detection_Source"]
                    stats["sources"][source] = stats["sources"].get(source, 0) + 1
                    
                    language = row["Language"]
                    stats["languages"][language] = stats["languages"].get(language, 0) + 1
                    
                    try:
                        confidences.append(float(row["Confidence"]))
                    except:
                        pass
            
            stats["unique_signs"] = len(stats["unique_signs"])
            stats["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0
            
            return stats
            
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"total_detections": 0, "unique_signs": 0, "sources": {}}