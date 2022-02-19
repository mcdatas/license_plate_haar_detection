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
        #gray_video = cv2.resize(gray_video, (500, 500))

        plate = plat_detector.detectMultiScale(gray_video,scaleFactor=1.2,minNeighbors=5,minSize=(25,25))

        for (x,y,w,h) in plate:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
            #frame[y:y+h,x:x+w] = cv2.blur(frame[y:y+h,x:x+w],ksize=(10,10))
            #text1 = text.readtext(np.array(gray_video * 255).astype('uint8'), paragraph="False")
            plateNumber = pytesseract.image_to_string(gray_video)
            text1 = print(plateNumber)
            cv2.putText(frame,text=text1,org=(x-3,y-3),fontFace=cv2.FONT_HERSHEY_COMPLEX,color=(0,0,255),thickness=1,fontScale=0.6)

        print(plate)
         
    #if ret == True:
        cv2.imshow('Video', frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    else:
        break
    print(ret)
video.release()
cv2.destroyAllWindows()            
