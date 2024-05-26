"""
Hand-engineered module with functions needed
to correctly get text representation of IMb image
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans


def find_match(target, id_list):
    """
    Simple helper function to find index of closest match
    """
    min = 10**9
    min_index = -1
    for index, elem in enumerate(id_list):
        if abs(target - elem) < min:
            min_index = index
            min = abs(target - elem)
    return min_index


def filter_error(lines, y_distr, h_distr):
    """
    Function to filter out noisy lines that are not related
    to barcode from detected lines
    """
    i = 0
    while i < len(lines):
        if lines[i][3] < h_distr[0]*0.65 or lines[i][3] > h_distr[2]*1.2:
            lines.pop(i)
            continue
        if lines[i][1] < y_distr[0]*0.8 or lines[i][1] > y_distr[1]*1.2:
            lines.pop(i)
            continue
        i += 1


def decode_cv(image):
    """
    Function to give text description of an image of barcode
    Implemented using OpenCV
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(
        image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    image_copy = image.copy()
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1,
                     color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    vertical_lines = []
    set_y = []
    set_h = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        set_y.append(y)
        set_h.append(h)
        vertical_lines.append((x, y, w, h))
    vertical_lines.sort(key=lambda x: x[0])

    if len(set_y) < 20:
        return 'Not recognized'

    set_y = sorted(list((set_y)))
    set_y.pop(0)
    set_h = sorted(list((set_h)))
    hh = set_h.pop(-1)
    j = 0
    while vertical_lines[j][3] != hh:
        j += 1
    vertical_lines.pop(j)
    k_means = KMeans(n_clusters=2, n_init=20)
    low_y = sum(set_y[:len(set_y)//3])/(len(set_y[:len(set_y)//3]))
    high_y = sum(set_y[len(set_y)//3:])/(len(set_y[len(set_y)//3:]))
    y_distr = [low_y, high_y]

    low_h = sum(set_h[:len(set_h)//4])/(len(set_h[:len(set_h)//4]))
    mid_h = sum(set_h[len(set_h)//3:2*len(set_h)//3]) / \
        (len(set_h[len(set_h)//3:2*len(set_h)//3]))
    high_h = sum(set_h[3*len(set_h)//4:])/(len(set_h[3*len(set_h)//4:]))
    h_distr = [low_h, mid_h, high_h]

    filter_error(vertical_lines, y_distr, h_distr)
    string = ''
    i = 0
    medium_bars = []
    for line in vertical_lines:
        if find_match(line[3], h_distr) == 1:
            medium_bars.append(line[1])
    medium_bars = sorted(list(medium_bars))
    y_predictions = k_means.fit_predict(np.array(medium_bars).reshape(-1, 1))

    for line in vertical_lines:
        id_h = find_match(line[3], h_distr)
        id_y = find_match(line[1], y_distr)
        i += 1
        if id_h == 0:
            string += 'T'
        elif id_h == 2:
            string += 'F'
        else:
            id_y = y_predictions[medium_bars.index(
                line[1])] != y_predictions[0]
            if id_y == 0:
                string += 'A'
            else:
                string += 'D'
    return string
