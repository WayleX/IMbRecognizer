"""
Experimental module to get angle of barcode in the image
As per low resolution photo, they can't be rotated without dramatic lose of quality
NOT USED
"""
import numpy as np
import cv2

def angle_cv(path):
    """
    Experimental function of detecting angle of barcode
    NOT USED
    """
    def find_match(id, id_list):
        results = []
        for elem in id_list:
            results.append(abs(id - elem))
        return results.index(min(results))
    image = cv2.imread(path)
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
    y_arr = []
    h_arr = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        y_arr.append(y)
        h_arr.append(h)
        vertical_lines.append((x, y, w, h))
    vertical_lines.sort(key=lambda x: x[0])
    set_y = sorted(list((y_arr)))
    set_y.pop(0)
    set_h = sorted(list((h_arr)))
    hh = set_h.pop(-1)
    j = 0
    while vertical_lines[j][3] != hh:
        j += 1
    vertical_lines.pop(j)
    low_h = sum(set_h[:len(set_h)//4])/(len(set_h[:len(set_h)//4]))
    mid_h = sum(set_h[len(set_h)//3:2*len(set_h)//3]) / \
        (len(set_h[len(set_h)//3:2*len(set_h)//3]))
    high_h = sum(set_h[3*len(set_h)//4:])/(len(set_h[3*len(set_h)//4:]))
    h_distr = [low_h, mid_h, high_h]
    high_bars = []
    for line in vertical_lines:
        if find_match(line[3], h_distr) == 2:
            high_bars.append(line)
    high_bars = sorted(list(high_bars))
    x_nums = np.array([(elem[0]+elem[2])/2 for elem in high_bars])
    y_nums = np.array([(elem[1]+elem[3])/2 for elem in high_bars])
    if len(x_nums)*np.sum(x_nums*x_nums) - np.sum(x_nums)**2 == 0.0:
        return 0
    a = (len(x_nums)*np.sum(x_nums*y_nums) - np.sum(x_nums)*np.sum(y_nums)
         )/(len(x_nums)*np.sum(x_nums*x_nums) - np.sum(x_nums)**2)

    return a
