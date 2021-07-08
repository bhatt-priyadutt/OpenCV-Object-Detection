import cv2
import tkinter as tk
import numpy as np
from tkinter import *
from tkinter import filedialog as fd
classesNames = []
confidenceThresholds = 0.7
nmsThreshold=0.3
wh =320 #width height
with open("Resources/coco.names.txt","rt") as f:
    classesNames = f.read().rstrip('\n').split('\n') #names stored in list
print(classesNames)
modelConfig = 'Resources/yolov3.cfg.txt'
modelWeights = 'Resources/yolov3.weights'
net = cv2.dnn.readNetFromDarknet(modelConfig,modelWeights) #creating a network
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) #tell the network which package is used
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) #tell net to use cpu
def findObj(outputs,img):  #used to get the bounding box values which has the highest confidence levels
    height,width,channel = img.shape
    boundingBox = []
    classIds = []
    confidence = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confi = scores[classId]
            if confi > confidenceThresholds:
                w,h = int(det[2]*width),int(det[3]*height)
                x,y = int((det[0]*width)-w/2),int((det[1]*height)-h/2)
                boundingBox.append([x,y,w,h])
                classIds.append(classId)
                confidence.append(float(confi))
                indices = cv2.dnn.NMSBoxes(boundingBox,confidence,confidenceThresholds,nmsThreshold) #used to remove unwanted overlapping boxes
                for i in indices:
                    i=i[0]
                    box = boundingBox[i]
                    x,y,w,h = box[0],box[1],box[2],box[3]
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
                    cv2.putText(img,f'{classesNames[classIds[i]].upper()}{int(confidence[i]*100)}%' ,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
def convert():
    fname = fd.askopenfilename(title='select a file',initialdir='/')
    detect(fname)

def gui():
    wid = tk.Tk()
    wid.title('Traffic Management')
    wid.geometry("500x200")
    Label(wid, text='Enter video file')
    btn = tk.Button(wid,text='Open', width=25, command=convert)
    btn.pack()
    wid.mainloop()
def detect(fname):
    vid = cv2.VideoCapture(fname)
    while True:
        success, img = vid.read()
        blob = cv2.dnn.blobFromImage(img,1/255,(wh,wh),[0,0,0],1,crop=False) #convert image into blob format
        net.setInput(blob)
        layerNames = net.getLayerNames()
        outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()] # extracting output layers
        outputs = net.forward(outputNames) #bounding boxes are there
        print(outputs[0].shape)
        findObj(outputs,img)
        cv2.imshow("video1", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    exit()
gui()