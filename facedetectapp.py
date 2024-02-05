import cv2
from matplotlib import pyplot as plt

import urllib.request

# Téléchargement du fichier haarcascade_frontalface_default.xml depuis le référentiel OpenCV
url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
urllib.request.urlretrieve(url, 'haarcascade_frontalface_default.xml')


# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread('NomDuFichier.jpeg') # insert the name of your file here

# Check if the image is loaded successfully
if img is None:
    print("Error: Unable to load the image.")
else:
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4) # on peut jouer sur les paramètres ici

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Print the number of faces detected
    print("Nombre de visages détectés :", len(faces))

    # Display the output using matplotlib
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
