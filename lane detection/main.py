from MotorModule import Motor
from LaneModule import getLaneCurve
import WebcamModule
 
##################################################
# motor = Motor(2,3,4,17,22,27)
motor = Motor(3,33,31,5,35,37)
##################################################
 
def main():
 
    img = WebcamModule.getImg()
    curveVal= getLaneCurve(img,1)
 
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
    motor.move(0.20,-curveVal*sen,0.05)
    #cv2.waitKey(1)
     
 
if __name__ == '_main_':
    while True:
        main()