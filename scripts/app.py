import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile

# .\.venv\Scripts\activate
# streamlit run scripts/app.py
st.set_page_config(page_title="Airport Safety PPE Detection", layout="wide")
st.title("✈️ Airport Safety: Privacy & Compliance Layer")

st.sidebar.header("System Settings")
model_path = 'models/detect/Airport_Safety_FYP/final_run_v1/weights/best.pt'

try:
    model = YOLO(model_path)
    st.sidebar.success("System Online")
except:
    st.sidebar.error("Model not found! Check path.")

conf = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
enable_privacy = st.sidebar.checkbox("Enable Privacy Mode (GDPR)", value=True)

def apply_privacy_blur(img_array, results):
    img_copy = img_array.copy()
    
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        
        if label in ['Person', 'Worker', 'person']: 
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            box_height = y2 - y1
            box_width = x2 - x1
            
            head_y2 = int(y1 + (box_height * 0.20))
            
            if head_y2 > y1 and box_width > 0:
                roi = img_copy[y1:head_y2, x1:x2]
                h_roi, w_roi = roi.shape[:2]
                
                if h_roi > 0 and w_roi > 0:
                    small = cv2.resize(roi, (max(1, w_roi//10), max(1, h_roi//10)), interpolation=cv2.INTER_LINEAR)
                    pixelated = cv2.resize(small, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)
                    
                    img_copy[y1:head_y2, x1:x2] = pixelated
                
    return img_copy

mode = st.sidebar.radio("Select Source", ["Image Inspection", "CCTV Video Audit"])

if mode == "Image Inspection":
    uploaded_file = st.file_uploader("Upload Site Photo", type=['jpg', 'png', 'webp'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Raw Input", use_container_width=True)
            
        if st.button("Analyze Compliance"):
            results = model.predict(image, conf=conf)
            
            orig_bgr_img = results[0].orig_img 
            
            if enable_privacy:
                blurred_bgr = apply_privacy_blur(orig_bgr_img, results)
                res_plotted_bgr = results[0].plot(img=blurred_bgr)
                caption_text = "Processed Output (Privacy Active)"
            else:
                res_plotted_bgr = results[0].plot()
                caption_text = "Processed Output (Raw)"
                
            final_img_rgb = cv2.cvtColor(res_plotted_bgr, cv2.COLOR_BGR2RGB)
                
            with col2:
                st.image(final_img_rgb, caption=caption_text, use_container_width=True)

elif mode == "CCTV Video Audit":
    video_file = st.file_uploader("Upload MP4", type=['mp4'])
    
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            results = model.predict(frame, conf=conf, verbose=False)
            
            if enable_privacy:
                blurred_frame = apply_privacy_blur(frame, results)
                frame_plotted = results[0].plot(img=blurred_frame)
            else:
                frame_plotted = results[0].plot()
            
            st_frame.image(frame_plotted, channels="BGR")
            
        cap.release()