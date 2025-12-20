"""
Attendance Management Module
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.metrics.pairwise import cosine_similarity
from .database import EmbeddingDatabase


class AttendanceManager:
    def __init__(self, attendance_csv="data/attendance.csv", threshold=0.5):
        """
        Initialize attendance manager
        
        Args:
            attendance_csv: Path to attendance CSV file
            threshold: Cosine distance threshold for face matching (default: 0.5)
        """
        self.attendance_csv = attendance_csv
        self.threshold = threshold
        self.db = EmbeddingDatabase()
        self._ensure_csv_exists()
        self.marked_today = set()  # Track marked attendance in current session
    
    def _ensure_csv_exists(self):
        """Create attendance CSV file if it doesn't exist"""
        os.makedirs(os.path.dirname(self.attendance_csv), exist_ok=True)
        if not os.path.exists(self.attendance_csv):
            df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
            df.to_csv(self.attendance_csv, index=False)
    
    def compare_embeddings(self, embedding1, embedding2):
        """
        Compare two embeddings using cosine similarity
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            similarity: Cosine similarity score (0-1, higher is more similar)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Reshape for sklearn cosine_similarity
        emb1 = embedding1.reshape(1, -1)
        emb2 = embedding2.reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(emb1, emb2)[0][0]
        
        # Cosine distance = 1 - similarity
        distance = 1 - similarity
        
        return distance, similarity
    
    def find_match(self, query_embedding, threshold=None):
        """
        Find matching face in registered embeddings
        
        Args:
            query_embedding: Embedding vector to match
            threshold: Optional threshold override
            
        Returns:
            matched_name: Name of matched person or None
            distance: Cosine distance to matched person
        """
        if threshold is None:
            threshold = self.threshold
        
        # Load all registered embeddings
        registered_embeddings = self.db.load_embeddings()
        
        if len(registered_embeddings) == 0:
            return None, None
        
        best_match = None
        best_distance = float('inf')
        
        # Compare with all registered faces
        for name, embedding in registered_embeddings.items():
            distance, similarity = self.compare_embeddings(query_embedding, embedding)
            
            if distance < best_distance:
                best_distance = distance
                best_match = name
        
        # Check if best match is within threshold
        if best_match and best_distance < threshold:
            return best_match, best_distance
        
        return None, None
    
    def mark_attendance(self, name):
        """
        Mark attendance for a person
        
        Args:
            name: Person's name
            
        Returns:
            bool: True if attendance was marked, False if already marked today
        """
        if name is None or name.strip() == "":
            return False
        
        name = name.strip()
        today = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Check if already marked today in this session
        if name in self.marked_today:
            return False
        
        # Check if already marked today in CSV
        if os.path.exists(self.attendance_csv):
            df = pd.read_csv(self.attendance_csv)
            if len(df) > 0:
                today_records = df[(df['Name'] == name) & (df['Date'] == today)]
                if len(today_records) > 0:
                    return False
        
        # Add new attendance record
        new_record = {
            'Name': name,
            'Date': today,
            'Time': current_time
        }
        
        df = pd.DataFrame([new_record])
        df.to_csv(self.attendance_csv, mode='a', header=not os.path.exists(self.attendance_csv) or os.path.getsize(self.attendance_csv) == 0, index=False)
        
        # Track in session
        self.marked_today.add(name)
        
        return True
    
    def get_attendance_log(self, date_filter=None, name_filter=None):
        """
        Get attendance log as DataFrame
        
        Args:
            date_filter: Optional date filter (YYYY-MM-DD)
            name_filter: Optional name filter
            
        Returns:
            DataFrame: Attendance log
        """
        if not os.path.exists(self.attendance_csv):
            return pd.DataFrame(columns=['Name', 'Date', 'Time'])
        
        df = pd.read_csv(self.attendance_csv)
        
        if len(df) == 0:
            return df
        
        # Apply filters
        if date_filter:
            df = df[df['Date'] == date_filter]
        
        if name_filter:
            df = df[df['Name'].str.contains(name_filter, case=False, na=False)]
        
        return df.sort_values(by=['Date', 'Time'], ascending=False)
    
    def reset_session(self):
        """Reset session tracking (clear marked_today set)"""
        self.marked_today.clear()

