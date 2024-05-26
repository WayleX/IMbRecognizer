"""
Module contains simple function for preprocessing image
"""
from PIL import ImageEnhance

def process_image(img, bright, contrast, sharpness):
    """
    Preprocessing image for better detection
    """
    filter = ImageEnhance.Brightness(img)
    img = filter.enhance(bright)
    filter = ImageEnhance.Contrast(img)
    img = filter.enhance(contrast)
    filter = ImageEnhance.Sharpness(img)
    img = filter.enhance(sharpness)
    return img