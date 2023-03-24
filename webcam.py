import cv2 as cv

capture = cv.VideoCapture(0)
haar_cascade = cv.CascadeClassifier('opencv_haarcascade_frontalface_default.xml')
eyes_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

while True:
    ret, frame = capture.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5)
    

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+w, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eyes_cascade.detectMultiScale(roi_gray, scaleFactor = 1.1, minNeighbors = 5)
        for (ex, ey, ew, eh) in faces:
            cv.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0, 255, 0), 2)

    cv.imshow('Video', frame)

    #Press Q to quit
    if cv.waitKey(1) == ord('q'):
        break

capture.release()
cv.destroyAllWindows() 
    