"""
Face Embedding Module using ONNX Facenet Model
"""
import onnxruntime as ort
import numpy as np
import cv2


class FaceEmbedder:
    def __init__(self, model_path="models/facenet.onnx"):
        """
        Initialize ONNX model for face embedding
        
        Args:
            model_path: Path to facenet.onnx model file
        """
        self.model_path = model_path
        try:
            # Create inference session
            self.session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            
            # Get input/output details
            input_info = self.session.get_inputs()[0]
            self.input_name = input_info.name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = input_info.shape
            
            # Determine expected input format
            # Common formats: (batch, channels, height, width) or (batch, height, width, channels)
            if len(self.input_shape) == 4:
                # Check if CHW or HWC format
                if self.input_shape[1] == 3 or self.input_shape[1] == 1:
                    # CHW format: (batch, channels, height, width)
                    self.input_format = 'CHW'
                    self.expected_height = self.input_shape[2] if self.input_shape[2] > 0 else 368
                    self.expected_width = self.input_shape[3] if self.input_shape[3] > 0 else 368
                else:
                    # HWC format: (batch, height, width, channels)
                    self.input_format = 'HWC'
                    self.expected_height = self.input_shape[1] if self.input_shape[1] > 0 else 160
                    self.expected_width = self.input_shape[2] if self.input_shape[2] > 0 else 160
            else:
                # Default to CHW format with 368x368
                self.input_format = 'CHW'
                self.expected_height = 368
                self.expected_width = 368
            
        except Exception as e:
            raise Exception(f"Failed to load ONNX model from {model_path}: {str(e)}")
    
    def preprocess_face(self, face_image):
        """
        Preprocess face image for model input
        
        Args:
            face_image: Face image (BGR format from OpenCV)
            
        Returns:
            preprocessed_image: Normalized image ready for inference
        """
        # Convert BGR to RGB
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        else:
            face_rgb = face_image
        
        # Resize to expected dimensions
        if face_rgb.shape[:2] != (self.expected_height, self.expected_width):
            face_rgb = cv2.resize(face_rgb, (self.expected_width, self.expected_height))
        
        # Normalize pixel values to [0, 1] range
        face_normalized = face_rgb.astype(np.float32) / 255.0
        
        # Convert to CHW format if needed (channels first)
        if self.input_format == 'CHW':
            # Transpose from (H, W, C) to (C, H, W)
            face_normalized = np.transpose(face_normalized, (2, 0, 1))
        
        # Add batch dimension
        # CHW: (1, 3, H, W) or HWC: (1, H, W, 3)
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        return face_batch
    
    def get_embedding(self, face_image):
        """
        Generate 128-dimensional embedding for face image
        
        Args:
            face_image: Face image (BGR format, 160x160)
            
        Returns:
            embedding: 128-dimensional numpy array
        """
        if face_image is None:
            raise ValueError("Face image is None")
        
        # Preprocess the face
        preprocessed = self.preprocess_face(face_image)
        
        # Run inference
        try:
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: preprocessed}
            )
            
            embedding = outputs[0][0]  # Remove batch dimension
            
            # Normalize embedding (L2 normalization)
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            raise Exception(f"Failed to generate embedding: {str(e)}")

