import cv2 as cv

detector = cv.CascadeClassifier('haar_cascade_face.xml')
cap = cv.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

    cv.imshow('frame', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
