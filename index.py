from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pydicom
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import io
import numpy as np
import cv2

output_label = ['any',	'epidural',	'intraparenchymal',	'intraventricular',	'subarachnoid',	'subdural']
app = Flask(__name__, static_folder='build')
cors = CORS(app)
print("before loading keras model")
new_model = tf.keras.models.load_model('model.keras')
print("after loading keras model")

def get_pixels_hu(scan): 
    image = np.stack([scan.pixel_array])
    image = image.astype(np.int16)    
    image[image == -2000] = 0    
    intercept = scan.RescaleIntercept
    slope = scan.RescaleSlope    
    if slope != 1: 
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)    
    image += np.int16(intercept)    
    return np.array(image, dtype=np.int16)

def apply_window(image, center, width):
    image = image.copy()
    min_value = center - width // 2
    max_value = center + width // 2
    image[image < min_value] = min_value
    image[image > max_value] = max_value
    return image

def apply_window_policy(image):
    image1 = apply_window(image, 40, 80) # brain
    image2 = apply_window(image, 80, 200) # subdural
    image3 = apply_window(image, 40, 380) # bone
    image1 = (image1 - 0) / 80
    image2 = (image2 - (-20)) / 200
    image3 = (image3 - (-150)) / 380
    image = np.array([
        image1 - image1.mean(),
        image2 - image2.mean(),
        image3 - image3.mean(),
    ]).transpose(1,2,0)
    return image

def process_image(dcm, x_hape, y_shape):
    image = get_pixels_hu(dcm)
    image = apply_window_policy(image[0])
    image -= image.min((0,1))
    image = (255*image).astype(np.uint8)
    image = cv2.resize(image, (x_hape, y_shape))
    return image

# Serve React App
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/health', methods=['GET'])
def health():
    out = {'status': 'Server is up and running'}
    return jsonify(out)

@app.route('/api/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print('request received')        
        # Get the DICOM file from the request
        file = request.files["file"]
        # Read the DICOM file into a pydicom Dataset object
        dcm = pydicom.dcmread(io.BytesIO(file.read()))
        print(f'dcm is {dcm.pixel_array}')
        image = process_image(dcm, 224, 224)
        print(f'image.shape is {image.shape}')
        image = np.expand_dims(image, axis=0)
        print(f'image.shape is {image.shape}')
        pred = new_model.predict(image)
        pred_list = pred.tolist()
        output_dict = dict(zip(output_label, pred_list[0]))        
    return jsonify(output_dict)

if __name__ == "__main__":
    app.run(port = 8000, use_reloader=True, threaded=True)