import flask
import werkzeug
import time
import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

app = flask.Flask(__name__)
config = Cfg.load_config_from_name('vgg_transformer')
config['cnn']['pretrained']=False
config['device'] = 'cpu'
detector = Predictor(config)

@app.route('/', methods = ['GET', 'POST'])
def handle_request():
    files_ids = list(flask.request.files)
    print("\nNumber of Received Images : ", len(files_ids))
    image_num = 1
    for file_id in files_ids:
        print("\nSaving Image ", str(image_num), "/", len(files_ids))
        imagefile = flask.request.files[file_id]
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        print("Image Filename : " + imagefile.filename)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        imagefile.save(timestr+'_'+filename)
        image_num = image_num + 1
        print("Recognition result:" + detector.predict(Image.open(imagefile)))
    print("\n")
    return "Image(s) Uploaded Successfully. Come Back Soon."

app.run(host="0.0.0.0", port=5002, debug=True)

