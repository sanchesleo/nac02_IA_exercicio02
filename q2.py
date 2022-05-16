import cv2
import numpy as np

cap = cv2.VideoCapture("admiravelmundonovo.mp4")

counter = 0

def  blurVideo(image):
    face_detect = cv2.CascadeClassifier('cascade/haarcascade_frontalface_alt.xml')
    face_data = face_detect.detectMultiScale(image, 1.3, 5)
    if len(face_data) > 0:
        for (x, y, w, h) in face_data:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = image[y:y+h, x:x+w]
        
            roi = cv2.GaussianBlur(roi, (23, 23), 30)
            
            image[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
            cv2.imshow("Feed", image)
    else:
        cv2.imshow("Feed", image)

def  sobelVideo(image):
    face_detect = cv2.CascadeClassifier('cascade/haarcascade_frontalface_alt.xml')
    face_data = face_detect.detectMultiScale(image, 1.3, 5)
      
    
    if len(face_data) > 0:
        for (x, y, w, h) in face_data:
            
            roi = image[y:y+h, x:x+w]
            
            roi = cv2.Sobel(roi,cv2.CV_64F,1,0,ksize=5)
            
            image[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
            cv2.imshow("Feed", image)
    else:
        cv2.imshow("Feed", image)

def  lenaVideo(image):
    alpha = 0.8
    beta = (1.0 - alpha)

    src1 = image
    src2 = cv2.imread("lena.png")
    src2= cv2.cvtColor(src2, cv2.COLOR_BGR2RGB )

    src2 = cv2.resize(src2, src1.shape[1::-1])


    dst = cv2.addWeighted(src1, alpha, src2, beta, 0.0)
    cv2.imshow("Feed", dst)
        
while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    
    cv2.imshow("Feed", frame)
    
    def mouse_click(event, x, y, flags, param):
            global counter, frame
            if event == cv2.EVENT_LBUTTONDOWN:
                counter += 1

                if counter == 4:
                     counter = 0
    cv2.setMouseCallback('Feed', mouse_click)

    if counter == 1:
        blurVideo(frame)
    elif counter == 2:
        sobelVideo(frame)
    elif counter == 3:
        lenaVideo(frame)
    else:
        cv2.imshow("Feed", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()