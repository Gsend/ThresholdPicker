import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector
# from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1200)
cap.set(4, 640)
# hand detector
detector = FaceDetector(minDetectionCon=.8)
while True:
    success, img = cap.read()
    print(img.shape)

    img, faces = detector.findFaces(img)
    cv2.imshow('video', img)
    cv2.waitKey(1)