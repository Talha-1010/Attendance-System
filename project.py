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

# step1

# importing image
path = "images"
images = []
classNames= []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg =  cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)



# initializing yagmail
yag = yagmail.SMTP('attendancesystem1010', 'attendance1010')
# find encodings of known images
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
        print(myDataList)
        nameList = []
        for line in myDataList:
            entry =  line.split(',') #spliting lines based on comma
            nameList.append(entry[0]) #append only the first element i.e names
        if name not in nameList:
            now = datetime.now()
            dateTimeString =  now.strftime('%H:%M:%S') #hour:min:secs
            f.writelines(f'\n{name},{dateTimeString}')

def getEmail(name):
    with open('email.csv','r+') as f:
        myDataList = f.readlines()
        for line in myDataList:
            entry = line.split(',')
            if name == entry[0]:
                email = entry[1]
                print("email =",email)
                return email






encodeListKnown =  findEncodings(images)

# if using ip webcam
# The URL to the javascript web page
# url = "http://192.168.1.4:8080/video"
# vid = cv2.VideoCapture(url)

# if using webcam
vid = cv2.VideoCapture(0)

while True:
    ret,frame = vid.read()
    smallImg =  cv2.resize(frame,(0,0),None,0.25,0.25)
    smallImg = cv2.cvtColor(smallImg,cv2.COLOR_BGR2RGB)

    facesLocation = face_recognition.face_locations(smallImg)
    encodingsOfCurrFrame = face_recognition.face_encodings(smallImg,facesLocation)

    for encodeFace,faceLocation in zip(encodingsOfCurrFrame,facesLocation):
        results = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDistance =  face_recognition.face_distance(encodeListKnown,encodeFace)
        # print(faceDistance)
        indexMatched =  np.argmin(faceDistance)



        if results[indexMatched]:
            name = classNames[indexMatched].upper()
            y1, x1, y2, x2 = faceLocation
            y1, x1, y2, x2 = y1 * 4, x1 * 4, y2 * 4, x2 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1 - 5), (x2, y1 - 35), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x2 + 10, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            # marking attendance
            markPresent(name)
            # sending email
            # "This is the body, and here is just text http://somedomain/image.png",
            #  "You can find an audio file attached.", '/local/path/to/song.mp3'
            contents = [
                name+ " you were marked present today"
            ]
            yag.send(getEmail(name), 'Attendance', contents)
            print('sent')


    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()






