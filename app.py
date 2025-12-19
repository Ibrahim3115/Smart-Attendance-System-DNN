"""
Smart Attendance System - Main Streamlit Application
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from datetime import datetime
import os

from utils.face_detector import FaceDetector
from utils.embedder import FaceEmbedder
from utils.database import EmbeddingDatabase
from utils.attendance_manager import AttendanceManager

# Page configuration
st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="📸",
    layout="wide"
)

# Initialize session state
if 'face_detector' not in st.session_state:
    st.session_state.face_detector = None
if 'face_embedder' not in st.session_state:
    st.session_state.face_embedder = None
if 'attendance_manager' not in st.session_state:
    st.session_state.attendance_manager = None
if 'db' not in st.session_state:
    st.session_state.db = None
if 'attendance_active' not in st.session_state:
    st.session_state.attendance_active = False
if 'video_capture' not in st.session_state:
    st.session_state.video_capture = None
if 'last_match' not in st.session_state:
    st.session_state.last_match = None
if 'last_match_time' not in st.session_state:
    st.session_state.last_match_time = None

def initialize_components():
    """Initialize all components"""
    try:
        if st.session_state.face_detector is None:
            st.session_state.face_detector = FaceDetector()
        
        if st.session_state.face_embedder is None:
            model_path = "models/facenet.onnx"
            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}. Please add facenet.onnx to the models/ directory.")
                return False
            st.session_state.face_embedder = FaceEmbedder(model_path)
        
        if st.session_state.attendance_manager is None:
            st.session_state.attendance_manager = AttendanceManager()
        
        if st.session_state.db is None:
            st.session_state.db = EmbeddingDatabase()
        
        return True
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        return False

def main():
    """Main application"""
    st.title("📸 Smart Attendance System")
    st.markdown("---")
    
    # Initialize components
    if not initialize_components():
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Home", "👤 Register Face", "✅ Mark Attendance", "📊 Attendance Log"]
    )
    
    # Display registered count
    registered_count = st.session_state.db.get_count()
    st.sidebar.info(f"Registered Faces: {registered_count}")
    
    # Route to appropriate page
    if page == "🏠 Home":
        show_home()
    elif page == "👤 Register Face":
        show_registration()
    elif page == "✅ Mark Attendance":
        show_attendance()
    elif page == "📊 Attendance Log":
        show_attendance_log()

def show_home():
    """Home page"""
    st.header("Welcome to Smart Attendance System")
    st.markdown("""
    This system uses face recognition to automatically mark attendance.
    
    **Features:**
    - 👤 Register faces with names
    - ✅ Real-time attendance marking
    - 📊 View attendance logs
    
    **How to use:**
    1. Go to "Register Face" to add new faces
    2. Go to "Mark Attendance" to start attendance scanning
    3. Go to "Attendance Log" to view records
    """)
    
    # Check model file
    model_path = "models/facenet.onnx"
    if os.path.exists(model_path):
        st.success("✓ Model file found")
    else:
        st.warning(f"⚠ Model file not found: {model_path}. Please add facenet.onnx to the models/ directory.")

def show_registration():
    """Face registration page"""
    st.header("👤 Register New Face")
    
    # Get name input
    name = st.text_input("Enter Name", placeholder="e.g., John Doe")
    
    # Webcam capture
    st.subheader("Capture Face")
    img_file_buffer = st.camera_input("Take a picture")
    
    if img_file_buffer is not None:
        # Convert to OpenCV format
        bytes_data = img_file_buffer.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Detect face
        face_image, bbox = st.session_state.face_detector.detect_face(cv_img)
        
        if face_image is not None:
            # Display detected face
            st.success("✓ Face detected!")
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            st.image(face_rgb, caption="Detected Face", width=200)
            
            # Register button
            if st.button("Register Face", type="primary"):
                if name and name.strip():
                    try:
                        # Generate embedding
                        embedding = st.session_state.face_embedder.get_embedding(face_image)
                        
                        # Save embedding
                        st.session_state.db.save_embedding(name.strip(), embedding)
                        
                        st.success(f"✓ Successfully registered: {name.strip()}")
                        st.balloons()
                        
                        # Clear camera input
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to register face: {str(e)}")
                else:
                    st.warning("Please enter a name")
        else:
            st.warning("⚠ No face detected. Please ensure your face is clearly visible.")

def show_attendance():
    """Real-time attendance marking page"""
    st.header("✅ Mark Attendance")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("Start Attendance", disabled=st.session_state.attendance_active):
            st.session_state.attendance_active = True
            st.session_state.attendance_manager.reset_session()
            st.rerun()
        
        if st.button("Stop Attendance", disabled=not st.session_state.attendance_active):
            st.session_state.attendance_active = False
            if st.session_state.video_capture is not None:
                st.session_state.video_capture.release()
                st.session_state.video_capture = None
            st.rerun()
    
    if st.session_state.attendance_active:
        # Use Streamlit's camera input for better compatibility
        st.subheader("Live Camera Feed")
        img_file_buffer = st.camera_input("Scan your face", key="attendance_camera")
        
        status_placeholder = st.empty()
        
        if img_file_buffer is not None:
            # Convert to OpenCV format
            bytes_data = img_file_buffer.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Detect face
            face_image, bbox = st.session_state.face_detector.detect_face(cv_img)
            
            if face_image is not None:
                # Draw bounding box
                frame_with_bbox = st.session_state.face_detector.draw_bbox(cv_img.copy(), bbox)
                
                try:
                    # Generate embedding
                    embedding = st.session_state.face_embedder.get_embedding(face_image)
                    
                    # Find match
                    matched_name, distance = st.session_state.attendance_manager.find_match(embedding)
                    
                    if matched_name:
                        # Display matched name on frame
                        cv2.putText(frame_with_bbox, f"Match: {matched_name}", 
                                  (bbox[0], bbox[1] - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Display frame with bounding box
                        frame_rgb = cv2.cvtColor(frame_with_bbox, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, caption=f"Matched: {matched_name} (Distance: {distance:.3f})", use_container_width=True)
                        
                        # Mark attendance if not already marked
                        if matched_name != st.session_state.last_match or \
                           (st.session_state.last_match_time and 
                            (time.time() - st.session_state.last_match_time) > 5):
                            if st.session_state.attendance_manager.mark_attendance(matched_name):
                                st.session_state.last_match = matched_name
                                st.session_state.last_match_time = time.time()
                                status_placeholder.success(f"✓ Attendance marked for: {matched_name}")
                                st.balloons()
                            else:
                                status_placeholder.info(f"Attendance already marked for {matched_name} today")
                        else:
                            status_placeholder.info(f"Matched: {matched_name} (already marked in this session)")
                    else:
                        frame_rgb = cv2.cvtColor(frame_with_bbox, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, caption="Face detected - No match found", use_container_width=True)
                        status_placeholder.warning("No match found. Please register your face first.")
                        st.session_state.last_match = None
                except Exception as e:
                    status_placeholder.error(f"Error processing face: {str(e)}")
            else:
                status_placeholder.warning("⚠ No face detected. Please ensure your face is clearly visible.")
        else:
            status_placeholder.info("Camera is active. Position your face in front of the camera.")
    else:
        st.info("Click 'Start Attendance' to begin scanning")
        if st.session_state.video_capture is not None:
            st.session_state.video_capture.release()
            st.session_state.video_capture = None

def show_attendance_log():
    """Attendance log viewer page"""
    st.header("📊 Attendance Log")
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        date_filter = st.date_input("Filter by Date", value=None)
    with col2:
        name_filter = st.text_input("Filter by Name", placeholder="Enter name to search")
    
    # Apply filters
    date_str = date_filter.strftime("%Y-%m-%d") if date_filter else None
    name_str = name_filter if name_filter else None
    
    # Get attendance log
    df = st.session_state.attendance_manager.get_attendance_log(
        date_filter=date_str,
        name_filter=name_str
    )
    
    if len(df) > 0:
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"attendance_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Statistics
        st.subheader("Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            unique_names = df['Name'].nunique()
            st.metric("Unique People", unique_names)
        with col3:
            if date_filter:
                st.metric("Records Today", len(df))
            else:
                today = datetime.now().strftime("%Y-%m-%d")
                today_records = len(df[df['Date'] == today])
                st.metric("Records Today", today_records)
    else:
        st.info("No attendance records found")

if __name__ == "__main__":
    main()

