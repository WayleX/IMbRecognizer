"""
Simple usage of program
"""
from app import process_barcode, process_barcodes_directory

TEST_PATH = './src/test/img1_1.png'
TEST_DIR = './src/test/'

def main(path_to_image, path_to_dir):
    process_barcode(path_to_image)
    process_barcodes_directory(path_to_dir)


if __name__ == '__main__':
    main(TEST_PATH, TEST_DIR)
