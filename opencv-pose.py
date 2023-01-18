import cv2 as cv
import numpy as np
from pupil_apriltags import Detector

point = lambda t: (int(t[0][0]), int(t[0][1]))
green = (0, 255, 0)

cameraMat = np.matrix(
  [[604.63310129,   0,         261.83745148],
  [  0,         600.52960814, 248.59373312],
  [  0,           0,           1        ]]
)

distCoeffs = np.matrix([[
  0.07758159, 0.0582485, 0.0021086, -0.0304751, -0.42220893
]])

objectPoints = np.matrix([
  [-3, 3, 0],
  [3, 3, 0],
  [3, -3, 0],
  [-3, -3, 0]
], dtype = "double")

cubePoints = np.concatenate((objectPoints, np.concatenate((objectPoints[:,:2], objectPoints[:,2:] - 6), 1)))
lineThingPoints = np.array([
  (0, 0, 0),
  (0, 1.5, 0),
  (-1.5, 0, 0),
  (0, 0, -1.5)
])

cap = cv.VideoCapture(0)
cv.namedWindow("solvePnP")

detector = Detector(
  families = "tag36h11",
  nthreads = 6,
  quad_decimate = 2.0, # reduces resolution of detection input, improves speed but reduces accuracy
  quad_sigma = 0.0, # blurs detection input, reduces noise
  refine_edges = 1,
  decode_sharpening = 0.25 # sharpens detection input, good for small tags but bad for low light
)

while True:
  ret, image = cap.read()
  
  results = detector.detect(cv.cvtColor(image, cv.COLOR_BGR2GRAY))

  if len(results) > 0:
    ret, rotationVec, translationVec = cv.solvePnP(objectPoints, results[0].corners, cameraMat, distCoeffs)

    computedImagePoints, j = cv.projectPoints(cubePoints, rotationVec, translationVec, cameraMat, distCoeffs)
    for [[x, y]] in computedImagePoints: cv.circle(image, (int(x), int(y)), 5, green, -1)

    computedLineThingPoints, j = cv.projectPoints(lineThingPoints, rotationVec, translationVec, cameraMat, distCoeffs)
    cv.line(image, point(computedLineThingPoints[0]), point(computedLineThingPoints[1]), (255, 0, 0), 3)
    cv.line(image, point(computedLineThingPoints[0]), point(computedLineThingPoints[2]), (0, 255, 0), 3)
    cv.line(image, point(computedLineThingPoints[0]), point(computedLineThingPoints[3]), (0, 0, 255), 3)

    cv.putText(image, "Rot: " + str(np.around(rotationVec * (180 / np.pi), 3)), (10, image.shape[0] - 40), cv.FONT_HERSHEY_PLAIN, 1, green, 2)
    cv.putText(image, "Trans: "+ str(np.around(translationVec, 3)), (10, image.shape[0] - 10), cv.FONT_HERSHEY_PLAIN, 1, green, 2)

  cv.imshow("solvePnP", image)
  if cv.waitKey(1) == 1: break