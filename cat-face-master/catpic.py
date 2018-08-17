# -*- coding=utf-8 -*-
import cv2


notcatPath = "haarcascades/haarcascade_frontalface_alt.xml"
notfaceCascade = cv2.CascadeClassifier(notcatPath)

catPath = "haarcascades/haarcascade_frontaldogface_alt.xml"
faceCascade = cv2.CascadeClassifier(catPath)

def hasMan(img, scale=(1.3, 3, (350,350))):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = notfaceCascade.detectMultiScale(
        gray,
        scaleFactor=scale[0],
        minNeighbors=scale[1],
        minSize=scale[2],
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces

def hasCat(img, scale=(1.3, 3, (350, 350))):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 猫脸检测
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=scale[0],
        minNeighbors=scale[1],
        minSize=scale[2],
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces

def pointCats(img, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img,'cat',(x,y-7), 3, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
    return img

def test_hasCat():
    img = cv2.imread("static/test.jpg")
    scale = (1.05, 22, (175, 175))
    faces = hasCat(img, scale)
    img = pointCats(img, faces)
    cv2.imshow('Cat?', img)
    cv2.imwrite("cat.jpg", img)
    cv2.waitKey(0)

if  __name__ == '__main__':
    test_hasCat()


