from imutils import build_montages

from helpers import load_mnist_dataset
from helpers import load_az_dataset
from keras.models import load_model
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

print("[INFO] loading datasets...")
(azData, azLabels) = load_az_dataset('A_Z Handwritten Data.csv')
(digitsData, digitsLabels) = load_mnist_dataset()
azLabels += 10
data = np.vstack([azData, digitsData])
labels = np.hstack([azLabels, digitsLabels])
data = [cv2.resize(image, (32, 32)) for image in data]
data = np.array(data, dtype="float32")
data = np.expand_dims(data, axis=-1)
data /= 255.0

print("[INFO] loading data...")
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)
labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

print("[INFO] loading handwriting OCR model...")
model = load_model('handwriting.model')

print("[INFO] testing model...")
images = []
for i in np.random.choice(np.arange(0, len(testY)), size=(49,)):
	probs = model.predict(testX[np.newaxis, i])
	prediction = probs.argmax(axis=1)
	label = labelNames[prediction[0]]
	image = (testX[i] * 255).astype("uint8")
	color = (0, 255, 0)
	if prediction[0] != np.argmax(testY[i]):
		color = (0, 0, 255)
	image = cv2.merge([image] * 3)
	image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
	cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
	images.append(image)

montage = build_montages(images, (96, 96), (7, 7))[0]
cv2.imshow("OCR Results", montage)
cv2.waitKey(0)