#!/usr/bin/env python
# coding: utf-8

# # COMPUTER VISION AND IOT INTERN
# 
# # SPARKS FOUNDATION
# 
# ## TASK 1 : Object Detection
# 
# ## Detect the object in a photo/video
# 
# ## By: Swati Namdev

# In[1]:


import numpy as np
import cv2


# In[2]:


net=cv2.dnn.readNet("yolov3.weights","yolov3.cfg")


# In[3]:


classes=[]
with open("coco.names","r") as f:
    classes=[line.strip()for line in f.readlines()]
print(classes)


# In[4]:


layer_names=net.getLayerNames()
output_layers=[layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]


# In[5]:


colors=np.random.uniform(0,255, size=(len(classes),3))


# In[6]:


img=cv2.imread("room_ser.jpg")
img=cv2.resize(img,None,fx=0.4,fy=0.4)
height,width,channels=img.shape
blob=cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=False)


# In[7]:


''''for b in blob:
    for n,img_blob in enumerate(b):
        cv2.imshow(str(n),img_blob)'''


# In[8]:


net.setInput(blob)
outs=net.forward(output_layers)


# In[9]:


class_ids=[]
confidences=[]
boxes=[]


# In[10]:


for out in outs:
    for detection in out:
        scores=detection[5:]
        class_id=np.argmax(scores)
        confidence=scores[class_id]
        if confidence>0.5:
            center_x=int(detection[0]*width)
            center_y=int(detection[1]*height)
            w=int(detection[2]*width)
            h=int(detection[1]*height)
            
            x=int(center_x-w/2)
            y=int(center_y-h/2)
            
            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)


# In[11]:


print(len(boxes))


# In[12]:


#number_objects_detected=len(boxes)
indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
#print(indexes)
font=cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x,y,w,h=boxes[i]
        label=str(classes[class_ids[i]])
        color=colors[i]
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.putText(img,label,(x,y+30),font,3,color,3)
        #print(label)


# In[ ]:





# In[13]:


cv2.imshow("Image",img)
cv2.waitKey(0)


# In[14]:


cv2.destroyAllWindows()


# In[ ]:




