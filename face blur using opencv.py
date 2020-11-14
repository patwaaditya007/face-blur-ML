#!/usr/bin/env python
# coding: utf-8

# In[7]:



import cv2
import numpy


# In[10]:


#download the file and upload it to your local environment to use it
#loading the face detector xml file

face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
test = face_detect.load('haarcascade_frontalface_default.xml')

print(test)


# In[11]:


#standard method to start video input via webcam using opencv
video= cv2.VideoCapture(0)
while True:
    ret,img = video.read()

#using .detectMultiScale to extract the (x,y,width,height) of the rectangle
    faces = face_detect.detectMultiScale(img, 1.3, 5)
    
    
    
#getting the region of interest,which is the face of the person in the image
    for (x,y,width,height) in faces:
        cv2.rectangle(img,(x,y),(x+width,y+height),(0,0,255),2)
        region = img[y:y+height, x:x+width]


    #getting dimensions
    (h,w) =  region.shape[:2]
    kW = int(w/3.0)
    kH = int(h/3.0)
    #to ensure the kernel has odd weight and height which is neccessary.
    if(kH%2==0):
      kH=kH-1
    if(kW%2==0):
      kW=kW-1
    #print(kH,kW)


    #blurring the region of interest in the image 
    blur=cv2.GaussianBlur(region,(kH,kW),0)


    #overlapping the blurred ROI on the original image by image arithmetic and 
    #then using cv2_imshow to see the final blurred image. 
    img[y:y+h,x:x+w] = blur
    cv2.imshow('img',img)


#press q key on keyboard to close the webcam videocapture
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
    


# In[ ]:





# In[ ]:




