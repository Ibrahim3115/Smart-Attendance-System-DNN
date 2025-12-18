# Smart Attendance System

A complete face recognition-based attendance system using CNN face detection and ONNX Facenet embeddings.

## Features

- 👤 **Face Registration**: Register faces with names using webcam
- ✅ **Real-Time Attendance**: Automatically mark attendance using face recognition
- 📊 **Attendance Log**: View and filter attendance records
- 🔒 **Duplicate Prevention**: Prevents marking attendance multiple times per day

## Requirements

- Python 3.10+
- Webcam access
- Facenet ONNX model file (`facenet.onnx`)

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add the Facenet ONNX model:
   - Download or obtain `facenet.onnx` model file
   - Place it in the `models/` directory
   - The file should be at: `models/facenet.onnx`

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. The application will open in your default web browser

3. **Register Faces**:
   - Navigate to "Register Face" page
   - Enter a name
   - Capture face using the camera
   - Click "Register Face" button

4. **Mark Attendance**:
   - Navigate to "Mark Attendance" page
   - Click "Start Attendance"
   - Position your face in front of the camera
   - The system will automatically detect and mark attendance

5. **View Attendance Log**:
   - Navigate to "Attendance Log" page
   - Filter by date or name
   - Download CSV if needed

## Project Structure

```
smart_attendance/
│── app.py                          # Main Streamlit application
│── requirements.txt                # Python dependencies
│── README.md                      # This file
│── models/
│     └── facenet.onnx            # Facenet ONNX model (user-provided)
│── utils/
│     ├── __init__.py
│     ├── face_detector.py        # Haar Cascade face detection
│     ├── embedder.py             # ONNX embedding generation
│     ├── database.py             # Embedding storage/retrieval
│     └── attendance_manager.py   # Attendance marking logic
│── data/
│     ├── registered_faces/
│     │     └── embeddings.pkl   # Stored face embeddings
│     └── attendance.csv          # Attendance records
│── assets/                       # UI resources (optional)
```

## Technical Details

- **Face Detection**: Haar Cascade Classifier (OpenCV)
- **Face Recognition**: Facenet ONNX model (128-dimensional embeddings)
- **Matching**: Cosine similarity with configurable threshold (default: 0.5)
- **Storage**: Pickle for embeddings, CSV for attendance logs

## Notes

- The system prevents duplicate attendance marking for the same person on the same day
- Face embeddings are stored locally in `data/registered_faces/embeddings.pkl`
- Attendance records are stored in `data/attendance.csv`
- Make sure you have proper lighting and face the camera directly for best results

## Troubleshooting

- **Model not found**: Ensure `facenet.onnx` is placed in the `models/` directory
- **Webcam not working**: Check camera permissions and ensure no other application is using it
- **No face detected**: Ensure good lighting and face the camera directly
- **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`

