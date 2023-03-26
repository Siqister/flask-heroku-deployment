import numpy as np
from flask import Flask, request, render_template
import torch
from torch.jit import load
import cv2
import io

# restore PyTorch model from TorchScript for inference only
model = load('./model/model_scripted.pt')

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	# parse incoming file into OpenCV image
	file = request.files['image']
	image_stream = io.BytesIO(file.read()) # returns BytesIO instance
	img = cv2.imdecode(np.frombuffer(image_stream.read(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
	
	# process the image so as to be compatible with inference model
	# resize to (m, 1, 28, 28)
	# convert dtype to float32 
	# normalize with mean of 0.1307 and std of 0.3081
	img = img.astype(np.float32)
	img = (img/255. - 0.1307)/0.3081
	# crop and resize
	h, w = img.shape
	target_size = min(h, w)
	img = img[int(h/2-target_size/2):int(h/2+target_size/2), int(w/2-target_size/2):int(w/2+target_size/2)] # crop to square
	img = cv2.resize(img, (28, 28))
	img = np.resize(img, (1,1,28,28))

	# inference
	output = model(torch.from_numpy(img))
	prediction = output.data.max(1, keepdim=True)[1][0].item()

	return render_template('index.html', prediction_text=f"Value is {prediction}")

if __name__ == "__main__":
	app.run(debug=True)