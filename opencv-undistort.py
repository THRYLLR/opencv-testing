import cv2 as cv
import numpy as np

matrix = np.matrix(
  [[629.31666134,   0,         287.51659823],
  [  0,          629.24993193, 260.05481837],
  [  0,           0,           1,        ]]
)

distortionCoeffs = np.matrix([[
  1, -1, -1, -1, 1
]])

cap = cv.VideoCapture(0)
cv.namedWindow("Raw")
cv.namedWindow("Undistorted")

h, w = cap.read()[1].shape[:2]
newMatrix, (x, y, w, h) = cv.getOptimalNewCameraMatrix(matrix, distortionCoeffs, (w, h), 1, (w, h))

print(matrix)
print(newMatrix)

while True:
  ret, image = cap.read()

  undistorted = cv.undistort(image, matrix, distortionCoeffs, None, newMatrix)
  # undistorted = undistorted[y:y + h, x:x + w]

  cv.imshow("Raw", image)
  cv.imshow("Undistorted", undistorted)

  if cv.waitKey(30) == 1: break