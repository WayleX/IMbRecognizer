from models.crnn import CRNN2
from ultralytics import YOLO
from PIL import Image as image
from skimage import io
import os
from PIL import ImageEnhance
from find import decode_cv
import numpy as np
from angle import angle_cv

def process_barcodes_directory(path_to_directory, bright=1.2, contrast=1.2, sharpness=1.5):
    results  = dict()
    photos = os.listdir(path_to_directory)
    yolo_model = YOLO(model = "trained_models/best_yolo.pt")
    yolo_model = YOLO(model = "trained_models/best_yolo.pt")
    for photo in photos:
        path_to_photo = os.path.join(path_to_directory, photo)
        img = image.open(path_to_photo)
        predict = yolo_model.predict(path_to_photo,conf=0.3,imgsz=640)
        box = predict[0].boxes
        if box.data.size()[0] == 0:
            print("Not found")
            continue
        bounds = (int(box.data[0,0]-5),int(box.data[0,1]),int(box.data[0,2]+5),int(box.data[0,3]))
        img = img.crop(bounds)
        filter =ImageEnhance.Brightness(img)
        img = filter.enhance(bright)
        filter =ImageEnhance.Contrast(img)
        img = filter.enhance(contrast)
        filter =ImageEnhance.Sharpness(img)
        img = filter.enhance(sharpness)
        img = img.resize((256*10,32*10))
        img.save('cache.jpg')
        total_angle = 0
        while abs(angle_cv('cache.jpg')) > 0.01:
            pred = angle_cv('cache.jpg')
            total_angle += pred
            img = img.resize((256,32))
            img = img.rotate(360+np.arctan(pred)*180/3.14)
            img = img.resize((256*10,32*10))
            img.save('cache.jpg')
        predicted = decode_cv('cache.jpg')
        results[photo] = predicted
        print(f'{photo} : {predicted}')
    return results
    
def process_barcode(path_to_photo):
    img = image.open(path_to_photo)
    predict = yolo_model.predict(path_to_photo,conf=0.3,imgsz=640)
    box = predict[0].boxes
    if box.data.size()[0] == 0:
        print("Not found")
        continue
    bounds = (int(box.data[0,0]-5),int(box.data[0,1]),int(box.data[0,2]+5),int(box.data[0,3]))
    img = img.crop(bounds)
    filter =ImageEnhance.Brightness(img)
    img = filter.enhance(bright)
    filter =ImageEnhance.Contrast(img)
    img = filter.enhance(contrast)
    filter =ImageEnhance.Sharpness(img)
    img = filter.enhance(sharpness)
    img = img.resize((256*10,32*10))
    img.save('cache.jpg')
    total_angle = 0
    while abs(angle_cv('cache.jpg')) > 0.01:
        pred = angle_cv('cache.jpg')
        total_angle += pred
        img = img.resize((256,32))
        img = img.rotate(360+np.arctan(pred)*180/3.14)
        img = img.resize((256*10,32*10))
        img.save('cache.jpg')
    predicted = decode_cv('cache.jpg')
    results[photo] = predicted
    print(f'{photo} : {predicted}')
    return results