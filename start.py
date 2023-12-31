from flask import Flask, render_template, request ,jsonify
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import os
import numpy as np

##----------------------------will be pre prepared by AI Team------------------------------------###

IMG_SIZE = 224
BATCH_SIZE = 32
#->List of the Dogs unique breeds
unique = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
       'american_staffordshire_terrier', 'appenzeller',
       'australian_terrier', 'basenji', 'basset', 'beagle',
       'bedlington_terrier', 'bernese_mountain_dog',
       'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound',
       'bluetick', 'border_collie', 'border_terrier', 'borzoi',
       'boston_bull', 'bouvier_des_flandres', 'boxer',
       'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff',
       'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua',
       'chow', 'clumber', 'cocker_spaniel', 'collie',
       'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo',
       'doberman', 'english_foxhound', 'english_setter',
       'english_springer', 'entlebucher', 'eskimo_dog',
       'flat-coated_retriever', 'french_bulldog', 'german_shepherd',
       'german_short-haired_pointer', 'giant_schnauzer',
       'golden_retriever', 'gordon_setter', 'great_dane',
       'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael',
       'ibizan_hound', 'irish_setter', 'irish_terrier',
       'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound',
       'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier',
       'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier',
       'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
       'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
       'miniature_schnauzer', 'newfoundland', 'norfolk_terrier',
       'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog',
       'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian',
       'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler',
       'saint_bernard', 'saluki', 'samoyed', 'schipperke',
       'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier',
       'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier',
       'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
       'standard_poodle', 'standard_schnauzer', 'sussex_spaniel',
       'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier',
       'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel',
       'west_highland_white_terrier', 'whippet',
       'wire-haired_fox_terrier', 'yorkshire_terrier']
#->Turns image into a numerical Form -Tensor- Given the Path
def process_img (image_path , img_size = IMG_SIZE) :
  """
  TAKE PATH TURN AND INTO TENSOR

  """

  #read in an image file
  image = tf.io.read_file(image_path)
  #Turn jpg into numerical tensor with 3 colour channels Red Green Blue
  image = tf.image.decode_jpeg(image , channels = 3)
  #Convert the color channel value from 0 - 255 to 0 - 1 values  (NORMALIZATION)
  image = tf.image.convert_image_dtype(image , tf.float32)
  #resize image to our desired value (224 , 224)
  image = tf.image.resize(image , size = [img_size , img_size])
  return image
#->Slices the Test set into batches (Will be Useful in the case of Big test set to reduce the effort of CPU) 
def create_data_batches(x , y = None , batch_size = BATCH_SIZE , valid_data = False , 
                        test_data = False):
  """
  Create Batches of data 
  """
  if test_data:
    print("Creating test data batches")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x))) #only filepath no labels
    data_batch = data.map(process_img).batch(batch_size)
    return data_batch

#->Loading the model Done By AI Team to use it 
def load_model(model_path):
  """
load a saved model from a path
  """
  print("Loading Saved Model")
  model = tf.keras.models.load_model(model_path, custom_objects={"KerasLayer" : hub.KerasLayer})
  return model
#->Given the image path you will Predict the label
def predict (path):
    test_filenames = [path]
    test_data = create_data_batches(test_filenames, test_data=True)
    test_predictions = model.predict(test_data, verbose = 0)
    return f"label:{unique[np.argmax(test_predictions[0])]}"

##----------------------------will be pre prepared by AI Team------------------------------------###






##----------------------------will be pre prepared by API Team------------------------------------###

app = Flask(__name__)


#Loading The DeepLearning Model 
model = load_model("API_image_TF_MODEL-main/20231116-05051700111111-1000.h5")

#path for the testing photos
test_path = "API_image_TF_MODEL-main\static"


#The Home Route
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


#The submit Root (The route used to predict)
@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "API_image_TF_MODEL-main/static/" + img.filename	
		img.save(img_path) 
		prediction = predict(img_path)
	return prediction


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
