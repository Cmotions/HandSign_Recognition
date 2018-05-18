# import packages
import cv2

# define variables
imgloc = '00 Data/pictures'     # the location where the pictures will be saved
startnumber = 1                 # the first number of the picture   
webcamchannel = 2               # the channel where the webcam can be found 
                                # (usually the webcam at the front of your laptop is channel 0)

# start the stream
cap = cv2.VideoCapture(webcamchannel)

# start a never ending loop
while True:
    # check if the camera gives an image and save that image
    # ret contains TRUE or FALSE
    # img contains the actual image from the camera stream
    ret, img = cap.read()
    
    # flip the image
    #img = cv2.flip(img, 1)
    
    # correct/adapt the image colors
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # check if the camera works
    if ret:
        # define a rectangle on the screen where the hand should be
        x1, y1, x2, y2 = 100, 100, 300, 300
        img_cropped = img[y1:y2, x1:x2]

        # make sure the rectangle won't be in the picture
        cv2.rectangle(img, (x1-2, y1-2), (x2+2, y2+2), (255,0,0), 2)
        # show the webcam image
        cv2.imshow("img", img)
        
        # check for key presses
        key = cv2.waitKey(1)
        
        # check if enter is pressed: TAKE PICTURE
        if  key == 13:
            # save the current image (within the rectangle) as a jpg
            cv2.imwrite(imgloc + '/picture_'+ str(startnumber) + '.jpg', img_cropped)
            # show a message at the screen to notify that a picture has been saved
            print('image ' + str(startnumber) + ' is saved!')
            # make sure the next pictures get's a higher number
            # this prevents pictures from being overwritten
            startnumber += 1
            
        # check if backspace is pressed: STOP STREAM
        elif key == 8:
            # close the camera stream window
            cv2.destroyAllWindows()
            cv2.VideoCapture(webcamchannel).release()
    else:
        break

# close the camera stream window
cv2.destroyAllWindows() 
cv2.VideoCapture(webcamchannel).release()