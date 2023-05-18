import cv2
import mediapipe as mp
import time
import torch
from PIL import Image
from torchvision.transforms import functional as F

# Load the YOLOv5 model with custom weights
model = torch.hub.load('yolov5', 'custom', path='/home/srv/perception673/project 5/last.pt', source='local')
model.eval()

mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

path_model = "models/"


model_name = "model-f6b98070.onnx"

model_dnn = cv2.dnn.readNet(path_model + model_name)

if(model_dnn.empty()):
    print("Could not load the neural net - check path")

model_dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model_dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

def depth_to_distance(depth):
    return -1.7 * depth + 2

cap = cv2.VideoCapture(0)
#with mp_facedetector.FaceDetection(min_detection_confidence=0.6) as face_detection:
while cap.isOpened():
        
    success, img = cap.read()
        
    imgHeight, imgWidth, channels = img.shape
        
    start = time.time()

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #results = face_detection.process(img)


    # Convert the frame to PIL Image format
    pil_image = Image.fromarray(img)

    # Perform object detection
    detections = model(pil_image)

    # Process the detections
    for detection in detections.xyxy[0]:
        x1, y1, x2, y2, conf, cls = detection.tolist()
        center_pt = (((x1+x2)/2),((y1+y2)/2))
        x,y = center_pt
        

        if conf > 0.5:
            label = model.names[int(cls)]

            # Draw bounding box and label on the frame
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(img, f'{label}: {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Display the frame
            #cv2.imshow('Webcam', img)
            ### Depth calculation
            blob = cv2.dnn.blobFromImage(img, 1/255., (384,384), (123.675, 116.28, 103.53), True, False)
            #blob = cv2.dnn.blobFromImage(img, 1/255., (256,256), (123.675, 116.28, 103.53), True, False)
            model_dnn.setInput(blob)

            depth_map = model_dnn.forward()

            depth_map = depth_map[0,:,:]
            depth_map = cv2.resize(depth_map, (imgWidth, imgHeight))
                
            depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            list1=[]
            for i in range(int(x-5),int(x+5)):
                for j in range(int(y-5),int(y+5)):
                    
                    depth_face = depth_map[int(j),int(i)]

                    depth_face = depth_to_distance(depth_face)
                    list1.append(depth_face)
            avg = sum(list1)/len(list1)
            print("average is : ")
            print(avg)
            print("Depth to face: ", depth_face)
            cv2.putText(img, "Depth in cm: " + str(round(depth_face,2)*100),(50,400), cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,0,0),2)


            end = time.time()
            totalTime = end - start

            fps = 1/totalTime

            cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,255,0),2)
            cv2.imshow("Face Detection", img)
            cv2.imshow("Depth map", depth_map)





    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        """ if results.detections:
            for id, detection in enumerate(results.detections):
                mp_draw.draw_detection(img, detection)

                bBox = detection.location_data.relative_bounding_box

                h,w,c = img.shape

                boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width *w), int(bBox.height * h)
                center_point = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)

                cv2.putText(img, f'{int(detection.score[0] * 100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1.5, (0,255,0),2)


                blob = cv2.dnn.blobFromImage(img, 1/255., (256,256), (123.675, 116.28, 103.53), True, False)
                model.setInput(blob)

                depth_map = model.forward()

                depth_map = depth_map[0,:,:]
                depth_map = cv2.resize(depth_map, (imgWidth, imgHeight))
                    
                depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


                depth_face = depth_map[int(center_point[1]),int(center_point[0])]

                depth_face = depth_to_distance(depth_face)
                print("Depth to face: ", depth_face)
                cv2.putText(img, "Depth in cm: " + str(round(depth_face,2)*100),(50,400), cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,0,0),2)


                end = time.time()
                totalTime = end - start

                fps = 1/totalTime

                cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,255,0),2)
                cv2.imshow("Face Detection", img)
                cv2.imshow("Depth map", depth_map)

                if cv2.waitKey(5) & 0xFF == 27:
                    break """

cap.release()