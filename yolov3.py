import cv2
import numpy as np

classPath = 'obj.names'
config = 'yolo-obj.cfg'
model = 'yolo-obj_best.weights'
whT = (320, 320)
confThreshold = 0.3
nmsThreshold = 0.2

cap = cv2.VideoCapture(0)
cv2.resizeWindow('Image',200,200)

classNames = []
with open(classPath,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
# print(classNames)

# create the network
net = cv2.dnn.readNetFromDarknet(config,model)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    #  Get nthe shape of image
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    # Get the individual output layes
    for output in outputs:
        # loop through each list in the array
        for det in output:
            # elements in list cx,cy,w,h,conf,person,car,...
            scores = det[5:]
            # get the index of max value
            classId = np.argmax(scores)
            # get the conf of max value
            confidence = scores[classId]
            # check if conf > thres
            if confidence > confThreshold:
                # convert w, h from % to values
                w,h = int(det[2]*wT) , int(det[3]*hT)
                # convert cx,cy to origin x and y
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    # Remove noise/ multi detection 
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    # print(indices)
 
    for i in indices:
        # To remove extra bracket
        i = i[0]
        # get bbox at i
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv2.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)
        # get the index value store in classIds and get the names using that index
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

while True:
    _,frame = cap.read()
    # frame = cv2.flip(frame,1)
    cv2.resize(frame,(200,200))

    # convert image into blob to input into network
    blob = cv2.dnn.blobFromImage(frame, 1/255, whT, [0, 0, 0],1,False)
    # set the input of the network / connect the input layer to network
    net.setInput(blob)

    # find the layer names of the network
    # print(net.getLayerNames())
    layers = net.getLayerNames()

    # To get output layers
    # print(net.getUnconnectedOutLayers())
    outputLayer = [(layers[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    # print(outputLayer)

    # connect output layer to network
    outputs = net.forward(outputLayer)
    # print(len(outputs))
    # print(outputs[0].shape)
    findObjects(outputs,frame)

    cv2.imshow('Image', frame)
    if cv2.waitKey(1) == 27 &0xff:
        break