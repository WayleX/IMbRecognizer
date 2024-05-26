"""
Main module containing functions to detect and recognize IMb given image path
"""
from ultralytics import YOLO
from PIL import Image
import os
from find import decode_cv
import numpy as np
from angle import angle_cv
from image_process import process_image

ANGLE_DETECTION = False


def process_barcodes_directory(path_to_directory, bright=1.2, contrast=1.2, sharpness=1.5):
    """
    Processes every image in directory to detect and recognize barcode
    """
    results = dict()
    photos = os.listdir(path_to_directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(script_dir, "cache/cache.jpg")
    file_path = os.path.join(script_dir, "yolo_model/best_yolo.pt")
    yolo_model = YOLO(model=file_path)
    for photo in photos:
        path_to_photo = os.path.join(path_to_directory, photo)

        img = Image.open(path_to_photo)
        predict = yolo_model.predict(
            path_to_photo, conf=0.3, imgsz=640, verbose=False)
        box = predict[0].boxes
        if box.data.size()[0] == 0:
            print(photo, ' : ', "Not found")
            continue
        bounds = (int(box.data[0, 0]-5), int(box.data[0, 1]),
                  int(box.data[0, 2]+5), int(box.data[0, 3]))

        img = img.crop(bounds)
        img = process_image(img, bright, contrast, sharpness)
        img = img.resize((256*10, 32*10))

        total_angle = 0
        if ANGLE_DETECTION:
            while abs(angle_cv(cache_path)) > 0.01:
                pred = angle_cv(cache_path)
                total_angle += pred
                img = img.resize((256, 32))
                img = img.rotate(360+np.arctan(pred)*180/3.14)
                img = img.resize((256*10, 32*10))
                img.save(cache_path)
        predicted = decode_cv(np.array(img)[:, :, ::-1].copy())
        results[photo] = predicted
        print(f'{photo} : {predicted}')
    return results


def process_barcode(path_to_photo, bright=1.2, contrast=1.2, sharpness=1.5):
    """
    Processes image to detect and recognize barcode
    """
    results = dict()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(script_dir, "cache/cache.jpg")
    file_path = os.path.join(script_dir, "yolo_model/best_yolo.pt")
    yolo_model = YOLO(model=file_path)
    img = Image.open(path_to_photo)
    predict = yolo_model.predict(
        path_to_photo, conf=0.3, imgsz=640, verbose=False)
    box = predict[0].boxes
    if box.data.size()[0] == 0:
        print("Not found")
    bounds = (int(box.data[0, 0]-5), int(box.data[0, 1]),
              int(box.data[0, 2]+5), int(box.data[0, 3]))

    img = img.crop(bounds)
    img = process_image(img, bright, contrast, sharpness)
    img = img.resize((256*10, 32*10))

    total_angle = 0
    if ANGLE_DETECTION:
        while abs(angle_cv(cache_path)) > 0.02:
            pred = angle_cv(cache_path)
            total_angle += pred
            img = img.resize((256, 32))
            img = img.rotate(360+np.arctan(pred)*180/3.14)
            img = img.resize((256*10, 32*10))
            img.save(cache_path)
    predicted = decode_cv(np.array(img)[:, :, ::-1].copy())
    results[path_to_photo] = predicted
    print(f"{path_to_photo.split('/')[-1]} : {predicted}")
    return results
