#importing the required libraries
import cv2
import dlib
import numpy as np
from keras.models import model_from_json

emotion_dict={0:'Angry',1:'Disgusted',2:'Fearful',3:'Happy',4:'Neutral',5:'Sad',6:'Surprised'}
json_file=open('model/emotion_model.json','r')
loaded_model_json=json_file.read()
json_file.close()
emotion_model=model_from_json(loaded_model_json)
emotion_model.load_weights('model/emotion_model.h5')
print("model loaded")
#capturing live video feed or giving a recorded video by copying video path
cap=cv2.VideoCapture("VID-20220517-WA0003.mp4")
#using print statements to debug the errors caused due to the delay in system camera response
print(4)
detector=dlib.get_frontal_face_detector()
print(6)
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
print(8)
#running an infinite loop until stopped by the user manually
while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(1280,720))
    if not ret:
        print("inactive")
        continue
    #converting a coloured frame into gray frame
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    print(13)
    faces=detector(gray)
    print(faces)
    for face in faces:

        x1=face.left()
        y1=face.top()
        x2=face.right()
        y2=face.bottom()
        #drawing a rectangle frame across the face detected
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),3)
        roi_gray_frame=gray[y1:y2,x1:x2]
        print(roi_gray_frame)
        if roi_gray_frame.size>0:
            cropped_img=np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame,(48,48)),-1),0)
            emotion_prediction=emotion_model.predict(cropped_img)
            maxindex=int(np.argmax(emotion_prediction))
            cv2.putText(frame,emotion_dict[maxindex],(x1+5,y2-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
    cv2.imshow('image',frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
