from flask import Flask, make_response
from flask_cors import CORS, cross_origin
from flask import request
from flask import jsonify
import os
import my_yolov6
import cv2

# Khởi tạo Flask Server Backend
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = "static"

yolov6_model = my_yolov6.my_yolov6("weights/best_ckpt.pt", "cpu", "data/mydataset.yaml", 640, False)

@app.route('/', methods=['POST'] )
def predict_yolov6():
    image = request.files['file']
    if image:
        # Lưu file
        path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        print("Save = ", path_to_save)
        image.save(path_to_save)

        frame = cv2.imread(path_to_save)

        # Nhận diên qua model Yolov6
        frame, no_object, flowerList = yolov6_model.infer(frame)
        if no_object >0:
            cv2.imwrite(path_to_save, frame)
            # print(no_object)

        del frame
        # Trả về đường dẫn tới file ảnh đã bounding box
        return jsonify(flowerList)# http://server.com/static/path_to_save
    return 'Upload file to detect'

# Start Backend
def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response
if __name__ == '__main__':
    app.run(host='127.0.0.1', port='6868')