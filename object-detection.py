import cv2

# Load a pre-trained Haar Cascade Classifier for detecting faces (or any object)
object_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image where you want to detect the object
image_path = '/content/WIN_20250417_08_59_35_Pro.jpg'  # Replace this with the path to your image
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

# Detect objects (faces in this case) in the image
objects = object_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# If objects are detected, print the message
if len(objects) > 0:
    print("Object Detected")
else:
    print("No Object Detected")
