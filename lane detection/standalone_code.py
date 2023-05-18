import RPi.GPIO as GPIO
from time import sleep

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2 
import numpy as np

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

camera= PiCamera()
camera.resolution=(480,240)
camera.framerate=30
# Capture from Camera
rawCapture=PiRGBArray(camera,size=(480, 240))
time.sleep(0.1)

def masking(img):

    # Define the region of interest (ROI) as the bottom 2/3rd of the image
    # img = img[int(height/3):height, 0:width]
    # img = cv.flip(img, -1)
    lower_hsv = np.array([0, 0, 124])
    upper_hsv = np.array([255, 15, 255])

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def warp_image(img): 

    h, w, _ = img.shape   
    # pts1 = np.float32([[73, 115], [345, 120], [1, 213], [478, 200]])
    pts1 = np.float32([[80, 150], [350, 150], [1, 230], [478, 230]])
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    # if inv:
    #     matrix = cv2.getPerspectiveTransform(pts2, pts1)
    # else:
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    return imgWarp

def getHistogram(img, minPer=0.1, display=False, region=1):
    if region == 1:
        histValues = np.sum(img, axis=0)
    else:
        histValues = np.sum(img[img.shape[0] // region:, :], axis=0)

    maxValue = np.max(histValues)
    minValue = minPer * maxValue

    indexArray = np.where(histValues >= minValue)
    basePoint = int(np.average(indexArray))

    if display:
        imgHist = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        for x, intensity in enumerate(histValues):
            pt1 = (x, img.shape[0])
            pt2 = (x, img.shape[0] - int(intensity // (255 // region)))
            cv2.line(imgHist, pt1, pt2, (255, 0, 255), 1)
            cv2.circle(imgHist, (basePoint, img.shape[0]), 20, (0, 255, 255), cv2.FILLED)
        return basePoint, imgHist

    return basePoint

def move(speed=0.5,turn=0,t=0):
    speed *=100
    turn *=70
    leftSpeed = speed-turn
    rightSpeed = speed+turn

    if leftSpeed>100: leftSpeed =100
    elif leftSpeed<-100: leftSpeed = -100
    if rightSpeed>100: rightSpeed =100
    elif rightSpeed<-100: rightSpeed = -100
    #print(leftSpeed,rightSpeed)
    pwmA.ChangeDutyCycle(abs(leftSpeed))
    pwmB.ChangeDutyCycle(abs(rightSpeed))
    if leftSpeed>0:GPIO.output(In1A,GPIO.HIGH);GPIO.output(In2A,GPIO.LOW)
    else:GPIO.output(In1A,GPIO.LOW);GPIO.output(In2A,GPIO.HIGH)
    if rightSpeed>0:GPIO.output(In1B,GPIO.HIGH);GPIO.output(In2B,GPIO.LOW)
    else:GPIO.output(In1B,GPIO.LOW);GPIO.output(In2B,GPIO.HIGH)
    sleep(t)

curveList = []
avgVal=10

EnaA= 3
In1A = 33
In2A = 31
EnaB= 5
In1B = 35
In2B = 37
GPIO.setup(EnaA,GPIO.OUT);GPIO.setup(In1A,GPIO.OUT);GPIO.setup(In2A,GPIO.OUT)
GPIO.setup(EnaB,GPIO.OUT);GPIO.setup(In1B,GPIO.OUT);GPIO.setup(In2B,GPIO.OUT)
pwmA = GPIO.PWM(EnaA, 100)
pwmB = GPIO.PWM(EnaB, 100)
pwmA.start(0)
pwmB.start(0)
mySpeed=0

while True:
    camera.capture(rawCapture, format="bgr")
    image = rawCapture.array
    image = cv2.flip(image, -1)
    imgResult = image.copy()
    hT, wT, _ = image.shape
    # height, width, _ = image.shape
    masked_image = masking(image)
    # cv2.imshow("org", masked_image)
    
    warped_image = warp_image(image)
    # cv2.imshow("warp", warped_image)

    # basePoint,imgHist = getHistogram(warped_image,display=True,minPer=0.5,region=4)
    # curveAveragePoint, imgHist = getHistogram(warped_image, display=True, minPer=0.9)

    basePoint = getHistogram(warped_image,display=False,minPer=0.5,region=4)
    curveAveragePoint = getHistogram(warped_image, display=False, minPer=0.9)

    curveRaw = curveAveragePoint - basePoint
    curveList.append(curveRaw)
    if len(curveList)>avgVal:
        curveList.pop(0)
    curve = int(sum(curveList)/len(curveList))

    #-----------------
    # imgInvWarp = utils.warpImg(imgWarp, points, wT, hT, inv=True)
    # imgInvWarp = cv2.cvtColor(warped_image, cv2.COLOR_GRAY2BGR)
    warped_image[0:hT // 3, 0:wT] = 0, 0, 0
    imgLaneColor = np.zeros_like(image)
    imgLaneColor[:] = 0, 255, 0
    imgLaneColor = cv2.bitwise_and(warped_image, imgLaneColor)
    imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
    midY = 450
    cv2.putText(imgResult, str(curve), (wT // 2 - 80, 85), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
    cv2.line(imgResult, (wT // 2, midY), (wT // 2 + (curve * 3), midY), (255, 0, 255), 5)
    cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY - 25), (wT // 2 + (curve * 3), midY + 25), (0, 255, 0), 5)
    for x in range(-30, 30):
        w = wT // 20
        cv2.line(imgResult, (w * x + int(curve // 50), midY - 10),
                    (w * x + int(curve // 50), midY + 10), (0, 0, 255), 2)

    #### NORMALIZATION
    curve = curve/100
    if curve>1: curve ==1
    if curve<-1:curve == -1

    curveVal = curve
    cv2.imshow("Result", imgResult)

    sen = 1.3  # SENSITIVITY
    maxVAl= 0.3 # MAX SPEED
    if curveVal>maxVAl:curveVal = maxVAl
    if curveVal<-maxVAl: curveVal =-maxVAl
    #print(curveVal)
    if curveVal>0:
        sen =1.7
        if curveVal<0.05: curveVal=0
    else:
        if curveVal>-0.08: curveVal=0
    move(0.20,-curveVal*sen,0.05)

    key=cv2.waitKey(200)&0xFF
    rawCapture.truncate(0)
    if key == ord("q"):
        #cv.destroyAllWindows()
        camera.close()
