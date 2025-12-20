"""
Face Detection Module using Haar Cascade
"""
import cv2
import numpy as np


class FaceDetector:
    def __init__(self):
        """Initialize Haar Cascade classifier for face detection"""
        try:
            # Try to load the cascade file
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                raise FileNotFoundError("Haar Cascade file not found")
        except Exception as e:
            raise Exception(f"Failed to load Haar Cascade: {str(e)}")
    
    def detect_face(self, frame):
        """
        Detect face in the given frame
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            cropped_face: Cropped face image (160x160) or None if no face detected
            bbox: Bounding box coordinates (x, y, w, h) or None
        """
        if frame is None:
            return None, None
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None, None
        
        # Get the largest face (assuming it's the main subject)
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        # Crop the face region
        face_roi = frame[y:y+h, x:x+w]
        
        # Resize to 160x160 for embedding
        face_resized = cv2.resize(face_roi, (160, 160))
        
        return face_resized, (x, y, w, h)
    
    def draw_bbox(self, frame, bbox):
        """
        Draw bounding box on frame
        
        Args:
            frame: Input frame
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            frame with bounding box drawn
        """
        if bbox is None:
            return frame
        
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame

