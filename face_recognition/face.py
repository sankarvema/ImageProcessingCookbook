import cv2

imagePath = "D:/wspc/python/Sample/2233700277_aae20477e8_o.jpg"
cascPath = "D:/wspc/python/imgProc/models/haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
    #flags = cv2.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.namedWindow("image",cv2.WINDOW_AUTOSIZE)
# cv2.resizeWindow("image", 600,600)
cv2.imshow("image", image)


cv2.waitKey(0)