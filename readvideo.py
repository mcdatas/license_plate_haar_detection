import cv2
import numpy as np

plat_detector =  cv2.CascadeClassifier(cv2.data.haarcascades + "DL_license_plate.xml")
video = cv2.VideoCapture('/Data/8.mp4')

if(video.isOpened()==False):
    print('Error Reading Video')

while True:
    ret,frame = video.read()
    if ret == True:
        gray_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        plate = plat_detector.detectMultiScale(gray_video,scaleFactor=1.2,minNeighbors=5,minSize=(25,25))

        for (x,y,w,h) in plate:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
            cv2.putText(frame,text='License Plate',org=(x-3,y-3),fontFace=cv2.FONT_HERSHEY_COMPLEX,color=(0,0,255),thickness=1,fontScale=0.6)

        print(plate)
        cv2.imshow('Video', frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    else:
        break
    print(ret)
video.release()
cv2.destroyAllWindows()            
