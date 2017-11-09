#!flask/bin/python
from flask import Flask, jsonify, request, make_response, send_file
import os
os.environ['PATH'] = r'D:\home\python354x64;' + os.environ['PATH']
import uuid
from config import cfg
from cntk import load_model
app = Flask(__name__)


model_path = os.path.join(cfg["CNTK"].MODEL_DIRECTORY, cfg["CNTK"].MODEL_NAME)
print("Loading existing model from %s" % model_path)
loadedModel = load_model(model_path)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/')
def index():
    return  "<html>" \
            "<body>" \
            "Hello, World!<br>" \
            "This is a sample web service written in Python using <a href=""http://flask.pocoo.org/"">Flask</a> module.<br>" \
            "Use one of the following urls to evaluate images:<br>" \
            "<a href=""/hotelidentifier/api/v1.0/evaluate/returntags"">/hotelidentifier/api/v1.0/evaluate/returntags</a> - takes image as parameter and returns cloud of tags<br>" \
            "<a href=""/hotelidentifier/api/v1.0/evaluate/returntags"">/hotelidentifier/api/v1.0/evaluate/returnimage</a> - takes image as parameter and returns tagged image<br>" \
            "</body>" \
            "</html>"


@app.route('/hotelidentifier/api/v1.0/evaluate/returntags', methods=['POST'])
def return_tags():
    file_upload = request.files['file']
    if file_upload:
        temp_file_path=os.path.join('./Temp',str(uuid.uuid4())+'.jpg')
        file_upload.save(temp_file_path)
        app.logger.debug('File is saved as %s', temp_file_path)
    from evaluate import evaluateimage
    return jsonify(tags=[e.serialize() for e in evaluateimage(temp_file_path,"returntags",eval_model=loadedModel)])

@app.route('/hotelidentifier/api/v1.0/evaluate/returnimage', methods=['POST'])
def return_image():
    file_upload = request.files['file']
    if file_upload:
        temp_file_path=os.path.join('./Temp',str(uuid.uuid4())+'.jpg')
        file_upload.save(temp_file_path)
        app.logger.debug('File is saved as %s', temp_file_path)
    from evaluate import evaluateimage
    return send_file(evaluateimage(temp_file_path,"returnimage",eval_model=loadedModel), mimetype='image/jpg')
    #return send_file(os.path.join('./Temp', temp_filename), mimetype='image/jpg')



if __name__ == '__main__':
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)


""" add UI later
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/api/uploader", methods=['POST'])
@cross_origin()
def api_upload_file():
    img = Image.open(BytesIO(request.files['imagefile'].read())).convert('RGB')
    img = ImageOps.fit(img, (224, 224), Image.ANTIALIAS)
    return json.dumps(run_some_deep_learning_cntk(img))


def run_some_deep_learning_cntk(rgb_pil_image):
    # Convert to BGR
    rgb_image = np.array(rgb_pil_image, dtype=np.float32)
    bgr_image = rgb_image[..., [2, 1, 0]]
    img = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

    # Use last layer to make prediction
    z_out = combine([MODEL.outputs[3].owner])
    result = np.squeeze(z_out.eval({z_out.arguments[0]: [img]}))

    # Sort probabilities 
    a = np.argsort(result)[-1]
    predicted_category = " ".join(LABELS[a].split(" ")[1:])
    
    return predicted_category.split(",")[0]
"""