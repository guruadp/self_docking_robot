import RPi.GPIO as GPIO
import time

# Set GPIO mode and warnings
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# Motor A pins
enable_pin_A = 3
input_pin1_A = 33
input_pin2_A = 31

# Motor B pins --- right
enable_pin_B = 5
input_pin1_B = 35
input_pin2_B = 37

print("Setting up pins...")
# Set up GPIO pins
GPIO.setup(enable_pin_A, GPIO.OUT)
GPIO.setup(input_pin1_A, GPIO.OUT)
GPIO.setup(input_pin2_A, GPIO.OUT)

GPIO.setup(enable_pin_B, GPIO.OUT)
GPIO.setup(input_pin1_B, GPIO.OUT)
GPIO.setup(input_pin2_B, GPIO.OUT)

# Create PWM objects for motor A and B
pwm_A = GPIO.PWM(enable_pin_A, 100)  # Frequency = 100Hz
pwm_B = GPIO.PWM(enable_pin_B, 100)  # Frequency = 100Hz

# Start PWM with 0% duty cycle
pwm_A.start(10)
pwm_B.start(10)

# Function to drive the motors
def drive_motors(speed_A, speed_B):
    # Set motor A direction and speed

    # Set motor speeds using PWM duty cycle
    pwm_A.ChangeDutyCycle(abs(speed_A))
    pwm_B.ChangeDutyCycle(abs(speed_B))
    
    if speed_A > 0:
        GPIO.output(input_pin1_A, True)
        GPIO.output(input_pin2_A, False)
    elif speed_A < 0:
        GPIO.output(input_pin1_A, False)
        GPIO.output(input_pin2_A, True)
    else:
        GPIO.output(input_pin1_A, False)
        GPIO.output(input_pin2_A, False)
    
    # Set motor B direction and speed
    if speed_B > 0:
        GPIO.output(input_pin1_B, True)
        GPIO.output(input_pin2_B, False)
    elif speed_B < 0:
        GPIO.output(input_pin1_B, False)
        GPIO.output(input_pin2_B, True)
    else:
        GPIO.output(input_pin1_B, False)
        GPIO.output(input_pin2_B, False)
    
    

# Example usage: drive motors forward for 2 seconds
print("Driving Forward...")
drive_motors(50, 50)  # Set motor speed to 50%
time.sleep(2)        # Drive for 2 seconds
drive_motors(0, 0) 
time.sleep(2)   # Stop motors
drive_motors(50, 25)  # Set motor speed to 50%
time.sleep(2)  
# Cleanup GPIO
GPIO.cleanup()
