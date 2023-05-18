import RPi.GPIO as gpio
import numpy as np
import time
from picamera.array import PiRGBArray
from picamera import PiCamera
import matplotlib.pyplot as plt
import cv2 as cv
import imutils
import serial
import os
from datetime import datetime
import smtplib
from smtplib import SMTP
from smtplib import SMTPException
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import math

    
def init():
    gpio.setmode(gpio.BOARD)
    gpio.setup(7,gpio.IN,pull_up_down=gpio.PUD_UP)
    gpio.setup(12,gpio.IN,pull_up_down=gpio.PUD_UP)
    gpio.setup(31,gpio.OUT)
    gpio.setup(33,gpio.OUT)
    gpio.setup(35,gpio.OUT)
    gpio.setup(37,gpio.OUT)
    
def gameover():
    gpio.output(31,False)
    gpio.output(33,False)
    gpio.output(35,False)
    gpio.output(37,False)
    #gpio.cleanup()

def measure_distance():
    #init()
    gpio.setmode(gpio.BOARD)
    gpio.setup(trig, gpio.OUT)
    gpio.setup(echo, gpio.IN)

    # ensure output has no value
    gpio.output(trig, False)
    time.sleep(0.01)

    # Generate trigger pulse
    gpio.output(trig, True)
    time.sleep(0.00001)
    gpio.output(trig, False)

    # Generate echo time signal
    while gpio.input(echo) == 0:
        pulse_start = time.time()

    while gpio.input(echo) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start

    # convert time to distance
    obstacle_distance = pulse_duration*17150
    obstacle_distance = round(obstacle_distance, 2)

    # Cleanup gpio pins & return distance estimate
    #gpio.cleanup()
    return obstacle_distance

def forward(distance, pwm_val):
    init()
    num_encoder_ticks = distance * 4701.2
    
    counter_left=0 #np.uint64(0)
    button_left=int(0)
    
    val=pwm_val
    
    counter_right= 0 #np.uint64(0)
    button_right=int(0)
    
    pwm_left=gpio.PWM(31,80)
    pwm_right=gpio.PWM(37,80)
    
    pwm_left.start(val)
    pwm_right.start(val)
    
    while(True):
        if int(gpio.input(7))!=int(button_left):
            button_left=int(gpio.input(7))
            counter_left+=1
        if int(gpio.input(12))!=int(button_right):
            button_right=int(gpio.input(12))
            counter_right+=1
    
        if counter_right>=num_encoder_ticks or counter_left>=num_encoder_ticks:  
            gameover()
            print("done")
            break
        
def reverse(distance, pwm_val):
    init()
    num_encoder_ticks = distance * 4701.2
    
    counter_left=0
    button_left=int(0)
    pwm_left=gpio.PWM(33,80)
    val=pwm_val
    
    counter_right=0
    button_right=int(0)
    pwm_right=gpio.PWM(35,80)
    
    pwm_left.start(val)
    pwm_right.start(val)
    
    while(True):
        if int(gpio.input(7))!=int(button_left):
            button_left=int(gpio.input(7))
            counter_left+=1
        if int(gpio.input(12))!=int(button_right):
            button_right=int(gpio.input(12))
            counter_right+=1

        if counter_right>=num_encoder_ticks or counter_left>=num_encoder_ticks:
            gameover()
            print("done")
            break



def pivot_right(degrees, val):
    global curr_x
    count=0
    init()
    pwm_left = gpio.PWM(31, val)
    pwm_right = gpio.PWM(35, val)
    while True:
        if(ser.in_waiting>0):
            count+=1
            line=ser.readline()
            if count>10:
                line = line.rstrip().lstrip()
                line = str(line)
                line = line.strip("'")
                line = line.strip("b'")
                prev_x = float(line)
                print("Start: ",prev_x)
                break
    
    while 0<=(abs(prev_x-curr_x))<degrees or (360-degrees)<(abs(prev_x-curr_x))<=360:
        print("Entered the pivot right condition")
        line = ser.readline()
        # print("line: ", line)
        line = line.rstrip().lstrip()
        line = str(line)
        line = line.strip("'")
        line = line.strip("b'")
        curr_x = float(line)
        print('X: ',curr_x)   
        pwm_left.start(val)
        pwm_right.start(val)
        #ser.reset_input_buffer()
    ser.reset_input_buffer()
    gameover()

def pivot_left(degrees, val):
    global curr_x
    count=0
    init()
    pwm_left = gpio.PWM(33, val)
    pwm_right = gpio.PWM(37, val)
    while True:
        if(ser.in_waiting>0):
            count+=1
            line=ser.readline()
            if count>10:
                line = line.rstrip().lstrip()
                line = str(line)
                line = line.strip("'")
                line = line.strip("b'")
                prev_x = float(line)
                print("Start: ",prev_x)
                break
    
    while 0<=(abs(prev_x-curr_x))<degrees or (360-degrees)<(abs(prev_x-curr_x))<=360:
        print("Entered pivot_left condition")
        line = ser.readline()
        line = line.rstrip().lstrip()
        line = str(line)
        line = line.strip("'")
        line = line.strip("b'")
        curr_x = float(line)
        print('X: ',curr_x)   
        pwm_left.start(val)
        pwm_right.start(val)
        ser.reset_input_buffer()
    ser.reset_input_buffer()
    gameover()


def send_mail():
    pic_time=datetime.now().strftime('%Y%m%d%H%M%S')
    camera.capture(pic_time+'.jpg')
    smtpUser='vigneshrrumd@gmail.com'
    smtpPass='bqfvxjasaknevkdc'
    toAdd="vignesh31794@gmail.com"
    fromAdd= smtpUser
    subject='test pic '+pic_time
    msg=MIMEMultipart()
    msg['Subject']=subject
    msg['From']=fromAdd
    msg['To']=toAdd
    msg.preamble="Docking Station reached at date/time : "+pic_time
    body=MIMEText("Docking Station reached at date/time : "+pic_time)
    msg.attach(body)
    fp=open(pic_time+'.jpg','rb')
    img=MIMEImage(fp.read())
    fp.close()
    msg.attach(img)
    s=smtplib.SMTP('smtp.gmail.com',587)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login(smtpUser,smtpPass)
    s.sendmail(fromAdd,msg["To"].split(","),msg.as_string())
    s.quit()
    print("email delivered!")


def get_current_orientation():
    line = ser.readline()
    # print("line: ", line)
    line = line.rstrip().lstrip()
    line = str(line)
    line = line.strip("'")
    line = line.strip("b'")
    curr_x = float(line)
    # print("Current Orientation: ", curr_x)
    return curr_x 


def move2QR():
    global robot_x
    global robot_y
    forward(0.2,50)
    robot_y +=20
    time.sleep(0.1)
    pivot_left(90, 90)
    time.sleep(0.1)


    forward(0.1,50)
    robot_x -=10
    
    # Average is taken to reduce the error
    robot_x = (robot_x + measure_distance())/2
    time.sleep(0.1)


def readQR(img):
    data, bbox, _ = detector.detectAndDecode(img)


    if(bbox is not None):
        for i in range(len(bbox)):
            cv.line(img,tuple(bbox[i][0]),tuple(bbox[(i+1) % len(bbox)][0]), color = (0,0,255), thickness = 4)
            cv.putText(img, data, (int(bbox[0][0][0]), int(bbox[0][0][1]) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),2)

    if data:
        print("Data :", data)

    # show result to the screen
    #cv.imshow("QR Code Detector", img)
    #cv.waitKey(100)

    return data




def detect_object(img):
    img = cv.flip(img, -1)
    img_hsv=cv.cvtColor(img, cv.COLOR_BGR2HSV)
    center_x=0
    h=0
    # Grand challenge
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([90, 255, 255])
    
   # lower_red = np.array([140, 65, 0])
   # upper_red = np.array([200, 228, 255])
    
    lower_red1 = np.array([0, 100, 120])
    upper_red1 = np.array([5, 255, 255])

    
    lower_red2 = np.array([150, 100, 120])
    upper_red2 = np.array([180, 255, 255])
    
    
    
    mask1 = cv.inRange(img_hsv, lower_red1, upper_red1)
    mask2 = cv.inRange(img_hsv, lower_red2, upper_red2)
    mask_dynamic = mask1 + mask2
    
    mask_static = cv.inRange(img_hsv, lower_green, upper_green)
   # mask_dynamic = cv.inRange(img_hsv, lower_red, upper_red)
    
    mask = mask_static + mask_dynamic
    #mask = cv.inRange(img_hsv, lower_green, upper_green)
    
    bitwise_mask = cv.bitwise_and(img, img, mask = mask)
    gray_img = cv.cvtColor(bitwise_mask, cv.COLOR_BGR2GRAY)
    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(cnts)
    #_, contours, _ = cv.findContours(gray_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    if len(contours) != 0:
    # identifying the largest countour (c) by the area to ignore other disturbances
        c = max(contours, key = cv.contourArea)
        x,y,w,h = cv.boundingRect(c)
        img = cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        center_x = x+(w/2)
        cv.imshow("masked",img)
        cv.waitKey(100)
        return center_x, h
    
    return center_x, h



# Camera initialization
camera= PiCamera()
camera.resolution=(640,480)
camera.framerate=30

# Capture from Camera
rawCapture=PiRGBArray(camera,size=(640,480))
time.sleep(0.1)
trig = 16
echo = 18

ser = serial.Serial('/dev/ttyUSB0',9600)
print("Program started after camera init")
curr_x = 0
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('docking_robot.avi', fourcc, 10, (640, 480))
# Robot position
robot_x = 50.0
robot_y = 10.0

## Safes collision threshold for the distance 
threshold_proximity = 60

threshold_height = 50

#define QR detector
detector = cv.QRCodeDetector()

# Total Areana height = 300 and width = 250
location_dict = {1:(100,250), 2:(150,250), 3:(200,250)}

# Move to QR code scanner region
move2QR()
robot_x = 0.0
robot_y = 0.0

## Capture image 
image = camera.capture("picture.jpg")
image = cv.imread("picture.jpg")
data = readQR(image)
docking_location = location_dict[int(data)]
time.sleep(0.1)
pivot_right(90,80)
time.sleep(0.1)
pivot_right(90,80)
time.sleep(0.1)
turned = 0
prev_distance = []
obstacle_spotted = 0

i = 0
angle = 0

while True:
   # i+=1
    print("Reading images")
    image_1 = camera.capture("new.jpg")
    image_1 = cv.imread("new.jpg")
    center_x, ht = detect_object(image_1)
    

    # check for all pixel values in the straight line proximity of the robot
    if ht < threshold_height:
        i=0
        print("Moving forward since no obstacle spotted in the proximity threshold")
        if turned == 0:
            if obstacle_spotted == 0:
                if abs(robot_x - docking_location[0]) > 25:
                    forward(0.2,50)
                    robot_x +=20

                else:
                    pivot_left(90, 80)
                    turned = 1
            else:
                distance_remain = docking_location[0] - robot_x
                if distance_remain*math.cos(angle*math.pi/180) > 25:
                    forward(0.2,50)
                    robot_x += (20 * math.cos(angle*math.pi/180))
                    robot_y += (20 * math.sin(angle*math.pi/180))
                else:
                    pivot_left(90-angle, 80)
                    temp_angle = angle
                    angle = 0
                    turned = 1
                          
                
        else:
            curr_pos = get_current_orientation()
            if curr_pos < 90:
                turn_left = curr_pos - 0
                pivot_left(turn_left,90)
            else:
                turn_right = 360-curr_pos
                pivot_right(turn_right,90)
            if measure_distance() > 30:
                if abs(robot_y - docking_location[1]) > 25:
                #if measure_distance() > 30:
                    forward(0.2,90)
                    robot_y +=20
                else:
                    print("Docking station reached")
            else:
                print("Docking station reached")
                break
    
        

    elif ht >= threshold_height:
        i+=1
        #prev_distance.append(distance)
        time.sleep(0.5)     
        if i == 5:
            print ("Static obstacle detected")
            obstacle_spotted = 1
            i=0
            if center_x > 320:
                pivot_left(10,80)
                angle +=10
            else:
                pivot_right(10,80)
                angle -=10
       
        

    key=cv.waitKey(200)&0xFF
    rawCapture.truncate(0)
    #out.write(image)
    if key == ord("q"):
        #cv.destroyAllWindows()
        camera.close()
        break

send_mail()
camera.close()
print("The Robot has reached the docking station")
#send_mail()
print("The final robot position is :")
print("x pos : ", robot_x)
print("y_pos : ", robot_y)
time.sleep(1)
gameover()


    
