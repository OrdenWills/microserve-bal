
import numpy as np # linear algebra

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D,BatchNormalization
from keras.models import Model
import cv2
from flask import Flask,render_template,redirect,url_for,request,flash,jsonify
import requests,os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'e8yrh0fd9j8yd8hfi0a;kmipsdjhiw'

IMG_DIR = './static/images'


def process_images(img):
    img = cv2.imread(str(img))
    img = cv2.resize(img, (224,224))
    if img.shape[2] ==1:
        img = np.dstack([img, img, img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.
    
    return img[None]

def predict(model,img):
    """returns confidence and clss"""
    pred = model.predict(img)
    return np.max(pred),np.argmax(pred)

def base_model():
    input_img = Input(shape=(224,224,3), name='ImageInput')
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_1')(input_img)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_2')(x)
    x = MaxPooling2D((2,2), name='pool1')(x)
    
    x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_1')(x)
    x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_2')(x)
    x = MaxPooling2D((2,2), name='pool2')(x)
    
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_3')(x)
    x = MaxPooling2D((2,2), name='pool3')(x)
    
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_1')(x)
    x = BatchNormalization(name='bn3')(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_2')(x)
    x = BatchNormalization(name='bn4')(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_3')(x)
    x = MaxPooling2D((2,2), name='pool4')(x)
    
    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(0.7, name='dropout1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(2, activation='softmax', name='fc3')(x)
    
    model = Model(inputs=input_img, outputs=x)
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