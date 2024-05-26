# IMbRecognizer

IMbRecognizer - Intelligent Mail Barcode Recognizer - an app designed to read such barcodes used in the US mail system

![image](https://github.com/WayleX/IMbRecognizer/assets/91287481/58769dc3-2684-48ab-90fc-1c4ec981a1c5)

Every barcode is combination of 65 bars - each corresponding to a letter (either A(ascending), D(descending), T(tracking) or F(full))

## How to install 
```
git clone https://github.com/WayleX/IMbRecognizer/
cd IMbRecognizer
pip install requirements.txt
```

## How to use
You can find example of usage in main.py
Just provide path to photo for process_barcode function
or path to directory for process_barcodes_directory function
```
process_barcode(path_to_image)
process_barcodes_directory(path_to_dir)
```

## Architecture

### First stage
  Using YOLO - to detect exactly where is a barcode

### Second stage
  Either OpenCV hand-engineered algorithm for detection or CRNN model

## Examples
![img1_1](https://github.com/WayleX/IMbRecognizer/assets/91287481/7defac8f-aaf0-4375-af88-405132b47e3a)

Output:

img1_1.png : TATFDDFADDTFTATFFFADDADTTTAFADTFDTTTDTAFFTFDAFAFADFATAATDTAATAATF
