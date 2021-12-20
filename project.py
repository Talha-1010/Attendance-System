import tkinter as tk
from tkinter import *
from tkinter import filedialog
import shutil


# Code*************************************************

# References
# https://www.youtube.com/watch?v=sz25xxF_AVE
# https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78
# https://pypi.org/project/yagmail/      SMTP client for Python
# https://github.com/Spidy20/Attendace_management_system
# https://github.com/Saurabh-Daware/Face-Recognition-Attendance
# https://github.com/shumbul/Smart-Attendance-System
# https://github.com/PrudhviGNV/Face-Recognisation-Based-Attendence
# https://github.com/kmhmubin/Face-Recognition-Attendance-System


import cv2
import numpy as np
import face_recognition
import os
import yagmail
from datetime import datetime
import time


# step1
# importing images
path = "images"
images = []
classNames= []
myList = os.listdir(path)
# print(myList)
for cl in myList:
    curImg =  cv2.imread(f'{path}/{cl}')
    # appending images
    images.append(curImg)
    # appending classNames
    name = os.path.splitext(cl)[0].split()
    # print(name[0])
    classNames.append(name[0])
# print(classNames)



# initializing yagmail
yag = yagmail.SMTP('attendancesystem1010', 'attendance1010')


# find encodings of known images after converting them to rgb
def findEncodings(images):
    encodings = []
    for img in images:
        img =  cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodings.append(encode)
    return encodings


# marking present after detection
def markPresent(name):
    with open('attendance.csv','r+') as f:
        #creating list
        myDataList = f.readlines()
        # print(myDataList)
        nameList = []
        for line in myDataList:
            entry =  line.split(',') #spliting lines based on comma
            nameList.append(entry[0]) #append only the first element i.e names
        if name not in nameList:
            now = datetime.now()
            dateTimeString =  now.strftime('%H:%M:%S') #hour:min:secs
            f.writelines(f'\n{name},{dateTimeString}')

# getting email from name
def getEmail(name):
    with open('email.csv','r+') as f:
        myDataList = f.readlines()
        for line in myDataList:
            entry = line.split(',')
            if name == entry[0]:
                email = entry[1]
                # print("email =",email)
                return email


def my_function(x):
  return list(dict.fromkeys(x))



def Program():
    encodeListKnown = findEncodings(images)

    # if using ip webcam
    # The URL to the javascript web page
    # url = "http://192.168.1.4:8080/video"
    # vid = cv2.VideoCapture(url)

    # if using webcam
    vid = cv2.VideoCapture(0)

    # used to record the time when we processed last frame
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0

    # creating an array of recognized faces
    recognised_faces_names = []

    while True:
        ret, frame = vid.read()
        smallImg = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        smallImg = cv2.cvtColor(smallImg, cv2.COLOR_BGR2RGB)

        # step2
        # finding the faces in our image
        # finding faces location
        facesLocation = face_recognition.face_locations(smallImg)
        # finding their encodings
        encodingsOfCurrFrame = face_recognition.face_encodings(smallImg, facesLocation)

        # time when we finish processing for this frame
        new_frame_time = time.time()

        # Calculating the fps

        # fps = number of frame processed in given time frame
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # step3
        # comparing these faces and finding the distance between them
        # comparing the encodings i.e 128 measurements of both the faces (by using linear SVM at the backend)
        for encodeFace, faceLocation in zip(encodingsOfCurrFrame, facesLocation):
            results = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)

            # getting the index of the image that has lowest faceDistance value
            # the lower the distance the better the match is
            indexMatched = np.argmin(faceDistance)

            y1, x1, y2, x2 = faceLocation
            y1, x1, y2, x2 = y1 * 4, x1 * 4, y2 * 4, x2 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1 - 5), (x2, y1 - 35), (0, 255, 0), cv2.FILLED)

            # font  to display FPS
            font = cv2.FONT_HERSHEY_SIMPLEX

            # if face matches
            if results[indexMatched]:
                name = classNames[indexMatched].upper()
                # print("face distance = ",faceDistance[indexMatched],"%")
                recognised_faces_names.append(name)
                cv2.putText(frame, name, (x2 + 10, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    recognised_faces_names = my_function(recognised_faces_names)
    # print(recognised_faces_names)

    # marking attendance and sending email
    for name in recognised_faces_names:
        markPresent(name)
        contents = [name + " you were marked present today"]
        yag.send(getEmail(name), 'Attendance', contents)

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()



# Code*************************************************



root = tk.Tk()
root.title("Real Time Attendance System")

# creating frame
frame = LabelFrame(root,text="Welcome",padx=20,pady=20)
frame.pack(padx=10,pady=10)


def getFileName():
    global result # to return calculation
    result = str(E1.get())

result = 0


# L1 = Label(root, text="Enter Image Name",)
# # L1.pack( side = LEFT)
# L1.grid(row=0,column=0)

E1 = Entry(frame, bd =10,bg= "gray")
E1.insert(0, 'Enter File Name ')
# E1.pack(side = RIGHT)
E1.grid(row=0,column=0)

button = tk.Button(frame, text="submit", command=getFileName)
button.grid(row=0,column=1)


def open_file():
    filename = filedialog.askopenfilename(initialdir="/", title="Select An Image", filetypes=(("jpeg files", "*.jpg"), ("gif files", "*.gif*"), ("png files", "*.png")))
    original= filename
    target = "C:/Users/Talha/Desktop/Computer Vision/Tutorials/Attendance-System/images/"+result
    shutil.copyfile(original, target)

my_button = Button(frame, text="Train", command=open_file)
my_button.grid(row=0,column=2,padx=5)



my_button = Button(frame, text="Run", command=Program)
my_button.grid(row=1,column=0,pady=10)

root.mainloop()