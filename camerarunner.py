# Imports
import mediapipe as mp
from picamera2 import Picamera2
import time
import cv2


# Initialize the pi camera
pi_camera = Picamera2()
# Convert the color mode to RGB
config = pi_camera.create_preview_configuration(main={"format": "RGB888"})
pi_camera.configure(config)

# Start the pi camera and give it a second to set up
pi_camera.start()
time.sleep(1)

def draw_pose(image, landmarks):
    ''' 
    TODO Task 1
    
    Code to this fucntion to draw circles on the landmarks and lines
    connecting the landmarks then return the image.
    
    Use the cv2.line and cv2.circle functions.
    
    landmarks is a collection of 33 dictionaries with the following keys
        x: float values in the interval of [0.0,1.0]
        y: float values in the interval of [0.0,1.0]
        z: float values in the interval of [0.0,1.0]
        visibility: float values in the interval of [0.0,1.0]
        
    References:
    https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
    https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    '''
    
    # copy the image
    landmark_image = image.copy()
    
    # get the dimensions of the image
    height, width, _ = image.shape
    
    #My code starts here
    list_points = []
    #Normalized position (%of image) to pixel number
    def extract_image_coords(x_normalized, y_normalized):
        coords = ((int)(x_normalized * width), (int)(y_normalized * height))
        return coords
    #input points as tuples
    def draw_line(point1, point2):
        cv2.line(landmark_image, point1, point2, (20, 180, 255), 4)

    for l in landmarks.landmark:
        list_points.append(extract_image_coords(l.x, l.y))
    
#    for point in list_points:
#        cv2.circle(landmark_image, point, 6, (20, 200, 200), -1)


    lp = list_points
    #eyes
    draw_line(lp[0], lp[1])
    draw_line(lp[1], lp[2])
    draw_line(lp[2], lp[3])
    draw_line(lp[3], lp[7])
    draw_line(lp[0], lp[4])
    draw_line(lp[4], lp[5])
    draw_line(lp[5], lp[6])
    draw_line(lp[6], lp[8])
    #mouth
    draw_line(lp[9], lp[10])
    #right arm
    draw_line(lp[12], lp[14])
    draw_line(lp[14], lp[16])
    draw_line(lp[16], lp[22])
    draw_line(lp[16], lp[20])
    draw_line(lp[16], lp[18])
    draw_line(lp[18], lp[20])

    draw_line(lp[12], lp[11])

    #left arm
    draw_line(lp[11], lp[13])
    draw_line(lp[13], lp[15])
    draw_line(lp[15], lp[21])
    draw_line(lp[15], lp[19])
    draw_line(lp[15], lp[17])
    draw_line(lp[17], lp[19])

    draw_line(lp[12], lp[24])
    draw_line(lp[11], lp[23])
    draw_line(lp[23], lp[24])

    #right leg
    draw_line(lp[24], lp[26])
    draw_line(lp[26], lp[28])
    draw_line(lp[28], lp[30])
    draw_line(lp[28], lp[32])
    draw_line(lp[30], lp[32])

    #left leg
    draw_line(lp[23], lp[25])
    draw_line(lp[25], lp[27])
    draw_line(lp[27], lp[29])
    draw_line(lp[27], lp[31])
    draw_line(lp[29], lp[31])
   
    for point in list_points:
        cv2.circle(landmark_image, point, 6, (20, 220, 220), -1)

    return landmark_image

def main():
    ''' 
    TODO Task 2
        modify this fucntion to take a photo uses the pi camera instead 
        of loading an image

    TODO Task 3
        modify this function further to loop and show a video
    '''

    # Create a pose estimation model 
    mp_pose = mp.solutions.pose
    
    # start detecting the poses
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

        # load test image
        #image = cv2.imread("person.png")	
        while True:
            image = pi_camera.capture_array() 

        # To improve performance, optionally mark the image as not 
        # writeable to pass by reference.
            image.flags.writeable = False
        
        # get the landmarks
            results = pose.process(image)
        
            if results.pose_landmarks != None:
                result_image = draw_pose(image, results.pose_landmarks)
                cv2.imwrite('output.png', result_image)
                cv2.imshow("Video", result_image)
                print(results.pose_landmarks)
            else:
                print('No Pose Detected')

            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    print('done')
