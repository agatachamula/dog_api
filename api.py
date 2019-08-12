from flask import Flask, request
from config import DogConfig, ListURL
import tensorflow as tf
from flask_restful import abort, Resource, Api
from flask_caching import Cache
from typing import BinaryIO



app = Flask(__name__)
app.config.from_object(DogConfig)
url_list=ListURL

PORT = app.config["PORT"]
DEBUG = app.config["DEBUG"]
HOST = app.config["HOST"]
APP_NAME = app.config["APP_NAME"]
MODEL_PATH = app.config["MODEL_PATH"]
LABELS_PATH = app.config["LABELS_PATH"]
CACHE_TYPE = app.config["CACHE_TYPE"]



#configure cache
cache=Cache(app, config={'CACHE_TYPE': CACHE_TYPE})



#problem handling
@cache.memoize(1)
def abort_if_not_image(filename: str):
    if (filename[-4:] != ".png" and filename[-4:] != ".jpg" and filename[-5:] != ".jfif" and filename[-5:] != ".jpeg"):
        return abort(404, message=f"Please provide image in jpg/png/jpeg/jfif format.")



def predict_breed(image_file: BinaryIO) -> list:
    # load model
    graph_def = tf.compat.v1.GraphDef()

    image_data = image_file.read()

    with tf.io.gfile.GFile(MODEL_PATH, 'rb') as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


    with tf.compat.v1.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, \
                               {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    return top_k[0]

def decode_breed(breed_id):

    # load labels
    labels = []

    with open(LABELS_PATH, 'rt') as lf:
        for l in lf:
            labels.append(l.strip())

    breed=labels[breed_id]

    return breed

class Root(Resource):

    url = url_list.RootURL

    def get(self) -> str:

        return("Welcome to DogAPI. Available commands are: /predict/breed")



class PredictBreed(Resource):

    url = url_list.PredictBreedURL

    def get(self) -> str:

        try:
            image_file = request.files["image"]

        except:
            return abort(404, message=f"Please provide file with image.")

        image_filename=image_file.filename
        abort_if_not_image(image_filename)

        breed_id=predict_breed(image_file)
        breed_name=decode_breed(breed_id)

        return breed_name

#    def post(self) -> str:
#
#        text=request.text["print"]
#
 #       print(text)

  #      return(text)


if __name__ == '__main__':
    api = Api(app)

    api.add_resource(PredictBreed, PredictBreed.url)
    api.add_resource(Root, Root.url)

    app.run(port=PORT, debug=DEBUG, host=HOST)