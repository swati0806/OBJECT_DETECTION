#!/usr/bin/env python
# coding: utf-8

# In[66]:


import cv2
import numpy as np


# In[67]:


net=cv2.dnn.readNet('yolov3.weights','yolov3.cfg')


# In[68]:


with open('coco.names','r') as f:
    classes=f.read().splitlines()
    print(classes)


# In[69]:


cap=cv2.VideoCapture("video.mp4")


# In[70]:

while True:
	_,img=cap.read()
	height,width,_=img.shape


	# In[71]:


	blob=cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),swapRB=True)
	net.setInput(blob)


	# In[72]:


	output_layers_names=net.getUnconnectedOutLayers()
	layerOutputs=net.forward(output_layers_names)


	# In[73]:


	boxes=[]
	confidences=[]
	class_ids=[]


	# In[74]:


	for output in layerOutputs:
	    for detection in output:
	        scores=detection[5:]
	        class_id=np.argmax(scores)
	        confidence=scores[class_id]
	        if confidence>0.5:
	            center_x=int(detection[0]*width)
	            center_y=int(detection[1]*height)
	            w=int(detection[2]*width)
	            h=int(detection[3]*height)
	            
	            x=int(center_x-w/2)
	            y=int(center_y-h/2)
	            
	            boxes.append([x,y,w,h])
	            confidences.append((float(confidence)))
	            class_ids.append(class_id)


	# In[75]:


	indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
	font=cv2.FONT_HERSHEY_PLAIN
	colors=np.random.uniform(9,255,size=(len(boxes),3))


	# In[76]:


	if len(indexes)>0:
	    for i in indexes.flatten():
	        x,y,w,h=boxes[i]
	        label=str(classes[class_ids[i]])
	        confidence=str(round(confidences[i],2))
	        color=colors[i]
	        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
	        cv2.putText(img,label+ " "*100+confidence,(x,y+30),font,2,color,2)
	    cv2.imshow("Image",img)
	    cv2.waitKey(1)
	#cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




