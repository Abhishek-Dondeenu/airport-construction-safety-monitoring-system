import cv2

def apply_privacy_blur(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        roi = cv2.GaussianBlur(roi, (99, 99), 30) 
        frame[y:y+h, x:x+w] = roi
        
    return frame
cap = cv2.VideoCapture(0)
print("Privacy Layer Prototype Active. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    blurred_frame = apply_privacy_blur(frame)
    cv2.imshow('GDPR-Compliant PPE Stream', blurred_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
