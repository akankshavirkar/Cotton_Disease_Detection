
import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
from ultralytics import YOLO
import numpy as np
import torch
import pandas as pd
import cv2


disease_info = pd.read_csv('disease_information.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplyinfo.csv',encoding='cp1252')


model = YOLO(r"Cotton_Disease2.pt")
names = model.names
def prediction(image_path):
    im1 = cv2.imread(image_path)

    result = model.predict(im1, show=False)
    boxes = result[0].boxes.xyxy.cpu().tolist()
    clss = result[0].boxes.cls.cpu().tolist()

    detected_objects = []

    if boxes is not None:
        for box, cls in zip(boxes, clss):
            class_name = names[int(cls)]
            detected_objects.append((class_name, box))


            for obj in detected_objects:
                pred = obj[0]
                if pred == "Alternaria leaf spot":
                    pred = int(0)
                if pred == "Aphids":
                    pred = int(1)
                if pred == "Armyworm":
                    pred = int(2)
                if pred == "Bacterial Blight":
                    pred = int(3)
                if pred =="Fusarium Wilt":
                    pred = int(4)
                if pred =="Grey milddew":
                    pred = int(5)
                if pred =="Healthy":
                    pred = int(6)
                if pred =="Leaf Curl":
                    pred = int(7)
    return pred


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')
    

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)

        title = disease_info['detected_objects'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['Fertilizer_name'][pred]
        supplement_image_url = supplement_info['Fertilizer_image'][pred]
        supplement_buy_link = supplement_info['Buy Link'][pred]
        return render_template('submit.html' , title = title , desc = description , prevent = prevent ,
                            image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link,Disease_img = file_path)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['Fertilizer_image']),
                           supplement_name = list(supplement_info['Fertilizer_name']), disease = list(disease_info['detected_objects']), buy = list(supplement_info['Buy Link']))

if __name__ == '__main__':
    app.run(debug=True)