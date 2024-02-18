
import numpy as np # linear algebra

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Input, Flatten, BatchNormalization
from keras.models import Model
import cv2
from flask import Flask,render_template,redirect,url_for,request,flash,jsonify
import requests,os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'e8yrh0fd9j8yd8hfi0a;kmipsdjhiw'

IMG_DIR = './static/images'


def process_images(img):
    img = cv2.imread(str(img),cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (150,150))

    return img[None]

def predict(model,img):
    """returns confidence and clss"""
    pred = model.predict(img)
    return np.max(pred),np.argmax(pred)

def base_model():
    model = Sequential()
    model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (150,150,1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Flatten())
    model.add(Dense(units = 128 , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1 , activation = 'sigmoid'))
    # model.compile(optimizer = "rmsprop" , loss = 'binary_crossentropy' , metrics = ['accuracy'])
    
    return model

def load_model(model,path='best-balanced.weights.h5'):
    model.load_weights(path)
    return model



def download_file_from_google_drive(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded successfully as {filename}")
    else:
        print(f"Failed to download file from {url}")

# Example usage



@app.route('/predict',methods=['POST'])
def index():
    # filename = 'best-balanced.weights.h5'  # Specify the filename you want to save the downloaded file as
    if request.method == 'POST':
        print([i for i in request.files])
        # print(request.json)
        if 'picture' not in request.files:
            error = 'No file part'
            return jsonify({'conf': conf, 'clss': error})
        file = request.files['picture']
        if file.filename == '':
            error = 'No selected file'
            return jsonify({'conf': conf, 'clss': error})
        
        model = load_model(base_model())
        user_img = request.files['picture']
        user_img.save('./static/images/my_img.jpg')

        pro_img = process_images('./static/images/my_img.jpg')
        pro_img = process_images(user_img)
        
        conf,clss = predict(model,pro_img)
        # print(conf,clss)
        # print('yes that it')

        return jsonify({'conf': conf, 'clss': clss})
    return jsonify({'conf': conf, 'clss': clss})

# @app.route('/validate')
# def validate():
    
#     stat = 'validated you can proceed now!'
#     return redirect(url_for('status',stat=stat))

# @app.route('/validate/<stat>:')
# def status(stat):
#     return render_template('success.html',stat=stat)
if __name__ == '__main__':
    app.run()