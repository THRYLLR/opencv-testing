import cv2 as cv
from pupil_apriltags import Detector

point = lambda t: (int(t[0]), int(t[1])) # convert point to int
green = (0, 255, 0)

cap = cv.VideoCapture(0)
cv.namedWindow("Image")

detector = Detector(
  families = "tag36h11",
  nthreads = 1,
  quad_decimate = 0.0, # reduces resolution of detection input, improves speed but reduces accuracy
  quad_sigma = 0.0, # blurs detection input, reduces noise
  refine_edges = 1,
  decode_sharpening = 0.25 # sharpens detection input, good for small tags but bad for low light
)

while True:
  ret, image = cap.read()
  
  results = detector.detect(cv.cvtColor(image, cv.COLOR_BGR2GRAY)) 

  for res in results:
    cv.circle(image, point(res.center), 5, green, -1)
    
    cv.line(image, point(res.corners[0]), point(res.corners[1]), green, 5)
    cv.line(image, point(res.corners[1]), point(res.corners[2]), green, 5)
    cv.line(image, point(res.corners[2]), point(res.corners[3]), green, 5)
    cv.line(image, point(res.corners[3]), point(res.corners[0]), green, 5)

  cv.imshow("Image", image)
  if cv.waitKey(30) == 1: break