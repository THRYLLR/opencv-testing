import cv2 as cv

cap = cv.VideoCapture(0)
cv.namedWindow("Image")

while True:
  ret, image = cap.read()
  cv.imshow("Image", image)
  if cv.waitKey(30) == 1: break