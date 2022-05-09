import cv2
import numpy as np

capture = cv2.VideoCapture(0)
whT = 320
confidenceThresh = 0.5
nms_threshold = 0.3

classesFile = 'coco.names'

classNames = []
with open(classesFile, 'rt') as fi:
    classNames = fi.read().rstrip('\n').split('\n')

print(classNames)
print(len(classNames))

modelConfig = 'yolov3-tiny.cfg'
modelWeights = 'yolov3-tiny.weights'

net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs, image):
    hT, wT, ct = image.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confidenceThresh:
                w, h = int(detection[2] * wT), int(detection[3] * hT)
                x, y = int((detection[0] * wT) - w / 2), int((detection[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confidenceThresh, nms_threshold)

    for i in indices:
        i = 0  # i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(image, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2)


while True:
    success, image = capture.read()
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
    print(outputNames)
    print(net.getUnconnectedOutLayers())

    '''
    print(net.getUnconnectedOutLayers())
    # determine the output layer
    # outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)
    # print(net.getUnconnectedOutLayers())
    '''

    outputs = net.forward(outputNames)
    '''
    print(outputs[0].shape)
    print(outputs[1].shape)
    print(outputs[2].shape)
    '''

    findObjects(outputs, image)

    print(outputs[0][0])
    cv2.imshow('Image', image)
    cv2.waitKey(1)
