import cv2                                                              # importing opencv module
import numpy as np                                                      # importing numpy module as np
import datetime

video_name = str(datetime.datetime.now())
video_name = video_name[:19]

def getContours(img):                                                   # defining a function which draws contours around shapes
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)             # it find contours in the image
    for cnt in contours:
        area = cv2.contourArea(cnt)                                     # it finds area of the shapes detected in the video in pixels
        #print(area)
        if area > 500:
            peri = cv2.arcLength(cnt,True)                              # it finds perimeter of the shapes detected
            #print(peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)               # it finds the sides of the shapes/polygons
            #print(len(approx))
            objCor = len(approx)
            x, y, width, height = cv2.boundingRect(approx)              # it draws a rectangle around the shapes

            if objCor == 3: ObjectType = 'regular'
            elif objCor == 4:
                aspRatio = width/float(height)
                if aspRatio > 0.95 and aspRatio < 1.05:
                    ObjectType = 'regular'
                    #print(aspRatio)
                else:
                    ObjectType = 'regular'
            elif objCor == 8:
                ObjectType = 'regular'
                #print("Circle:",objCor)
            else:
                ObjectType = "irregular"
                #print("None:",objCor)
            # cv2.rectangle(imgContour,(x,y),(x+width,y+height),(0,255,0),1)
            if ObjectType == 'regular':
                cv2.drawContours(imgContour, cnt, -1, (0, 0, 255), 3)  # it draws contour around all shapes detected
                cv2.putText(imgContour, ObjectType,
                                (x+(width//2)-10,y+(height//2)-10),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)        # it put text under the shape
            else:
                cv2.drawContours(imgContour, cnt, -1, (0, 255, 0), 3)  # it draws contour around all shapes detected
                cv2.putText(imgContour, ObjectType,
                            (x + (width // 2) - 10, y + (height // 2) - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)  # it put text under the shape


#path = "/home/niks/Downloads/shapes.jpg"
#img = cv2.imread(path,1)
#img = cv2.resize(img,(img.shape[0]//3,img.shape[1]//2))

video = cv2.VideoCapture(0)                                             # you can insert path to a video of 0 for inbuilt web-cam,
                                                                        # and 1 and so on for external web-cams in parenthesess

result = cv2.VideoWriter((video_name+".avi"), cv2.VideoWriter_fourcc(*'MJPG'), 10, (640,480))        # Define the codec and create VideoWriter object

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
                                                                        # inserting an XML file which contains coding to identify face attributes to detect it.
while True:
    boolean, img = video.read()                                         # .read() returns 2 values(Boolean and Numpy array as frames of video)
                                                                        #  True if it starts reading frames and frames itself
    img = cv2.resize(img,(640,480))                                    # resizing the frame
    imgContour = img.copy()                                             # copying the frame
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                         # converting the frame into gray image
    blurr = cv2.GaussianBlur(gray,(7,7),1)                              # creating GaussianBlur image by gray image
    canny = cv2.Canny(blurr,50,50)                                      # creating canny image by GaussianBlur image

    getContours(canny)                                                  # calling function to make contours and names around shapes

    face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for x, y, w, h in face:
        imgContour = cv2.rectangle(imgContour, (x, y), (x + w, y + h), (126, 126, 0), 1)      # drawing a rectangle around the face detected
        cv2.putText(imgContour, "Face",
                            (x,y) , cv2.FONT_HERSHEY_PLAIN, 1, (126,126,2), 2)        # it put "Face" above the rectangle drawn around the face detected
    #cv2.imshow("shapes",img)
    #cv2.imshow("shapes-gray",gray)
    #cv2.imshow("shapes-blurr",blurr)
    #cv2.imshow('canny',canny)
    cv2.imshow('contour',imgContour)                                    # it shows frames on which contours were made
    result.write(imgContour)                                            # output the frame

    if cv2.waitKey(1) & 0XFF == ord('q'):break                         # cv2.waitKey(16) it will refresh frames in each 16 mili-seconds
                                                                        # 0XFF == ord('q'):break it will break the loop

video.release()                                                         # release the video capturing object
result.release()                                                        # release the video output object
cv2.destroyAllWindows()                                                 # it will close all the windows once the loop is broken
