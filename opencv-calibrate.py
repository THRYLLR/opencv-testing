import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

cornerCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

imageObjectPts = np.zeros((1, 49, 3), np.float32)
imageObjectPts[0, :, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

objectPoints = []
imagePoints = []

count = 0

def click(event, x, y, flags, param):
  if event != cv.EVENT_LBUTTONDOWN: return

  imageFound, image = cap.read()
  if imageFound == False: return

  image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

  chessboardFound, corners = cv.findChessboardCorners(image, (7, 7))

  if chessboardFound == True:
    objectPoints.append(imageObjectPts)
    corners = cv.cornerSubPix(image, corners, (11, 11), (-1, -1), cornerCriteria)

    imagePoints.append(corners)

    global count
    count = count + 1

def calibrate():
    ret, image = cap.read()

    ret, matrix, distortionCoeffs, rotationVectors, translationVectors = cv.calibrateCamera(objectPoints, imagePoints, image.shape[1::-1], None, None)

    print("Camera matrix:")
    print(matrix)
    print("\nDistortion coefficients:")
    print(distortionCoeffs)
    print("\nRotation vectors:")
    print(rotationVectors)
    print("\nTranslation vectors:")
    print(translationVectors)

cv.namedWindow("Camera Calibration")
cv.setMouseCallback("Camera Calibration", click)

while True:
  if count >= 12: break
  
  imageFound, image = cap.read()
  gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

  chessboardFound, corners = cv.findChessboardCorners(gray, (7, 7))

  if chessboardFound == True:
    corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), cornerCriteria)
    image = cv.drawChessboardCorners(image, (7, 7), corners, chessboardFound)

  cv.putText(image, str(count), (10, image.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

  cv.imshow("Camera Calibration", image)
  if cv.waitKey(10) == 1: break

cv.destroyAllWindows()
if count >= 12: calibrate()