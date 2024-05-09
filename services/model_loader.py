from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import pickle

def load_trained_model(selected_model="effnetb4"):
    model = ""
    if selected_model == "effnetb4":
        model = load_model("models/effnetb4.h5")
    return model

def make_prediction(input_image_path, trained_model):
    img = image.load_img(input_image_path, target_size=(299,299))
    img_array = image.img_to_array(img)

    # Rescale the image
    img_array /= 255.0
    
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    preds = trained_model.predict(img_array)
    pred_class = preds.argmax(axis=-1)
    return preds.tolist(), pred_class.tolist()