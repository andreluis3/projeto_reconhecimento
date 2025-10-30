import os
import cv2
import face_recognition
import numpy as np
import matplotlib.pyplot as plt

dataset_path = os.path.join(os.path.dirname(__file__), "dataset_teste")
imagePath1 = 'andre.png'
imagePath2 = 'duda.jpg'

img1 = cv2.imread(imagePath1)
img2 = cv2.imread(imagePath2)

img1.shape
img2.shape

#examinar imagem em tons de cinza 
gray_image = cv2.cvtColor(img1, img2, cv2.COLOR_BGR2GRAY)

gray_image.shape

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)
#Desenho da caixa delimmitadora do rosto1
for (x, y, w, h) in face:
    cv2.rectangle(img1, img2, (x, y), (x + w, y + h), (0, 255, 0), 4)  

img_rgb = cv2.cvtColor(img1, img2, cv2.COLOR_BGR2RGB)


#acessar webcam
video_capture = cv2.VideoCapture(0)
ret, frame = video_capture.read()

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces


#detectar faces em tempo real 
while True:

    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    faces = detect_bounding_box(
        video_frame
    )  # apply the function we created to the video frame

    cv2.imshow(
        "My Face Detection Project", video_frame
    )  # display the processed frame in a window named "My Face Detection Project"

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()

plt.figure(figsize=(20,10))
plt.imshow(img_rgb)
plt.axis('off')