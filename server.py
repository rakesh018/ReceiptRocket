import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
from PIL import Image
import pytesseract
import re

def opencv_resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def process_receipt(image_path, action):
    img = Image.open(image_path)
    img.thumbnail((800, 800), Image.LANCZOS)

    image = cv2.imread(image_path)
    resize_ratio = 500 / image.shape[0]
    original = image.copy()
    image = opencv_resize(image, resize_ratio)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(blurred, rectKernel)
    edged = cv2.Canny(dilated, 100, 200, apertureSize=3)

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    def approximate_contour(contour):
        peri = cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, 0.032 * peri, True)

    def get_receipt_contour(contours):    
        for c in contours:
            approx = approximate_contour(c)
            if len(approx) == 4:
                return approx

    receipt_contour = get_receipt_contour(largest_contours)

    def contour_to_rect(contour):
        pts = contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect / resize_ratio

    def wrap_perspective(img, rect):
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    scanned = wrap_perspective(original.copy(), contour_to_rect(receipt_contour))

    if action == 'scan':
        def bw_scanner(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            T = threshold_local(gray, 21, offset=5, method="gaussian")
            return (gray > T).astype("uint8") * 255

        result = bw_scanner(scanned)
        output = Image.fromarray(result)
        output_path = 'result-of-part1.png'
        output.save(output_path)
        return output_path

    elif action == 'extract':
        def bw_scanner(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            T = threshold_local(gray, 21, offset=5, method="gaussian")
            return (gray > T).astype("uint8") * 255

        result = bw_scanner(scanned)
        output = Image.fromarray(result)
        output.save('result-of-part1.png')

        image = cv2.imread('result-of-part1.png', cv2.IMREAD_GRAYSCALE)
        extracted_text = pytesseract.image_to_string(image)

        def find_amounts(text):
            amounts = re.findall(r'\d+\.\d{2}\b', text)
            floats = [float(amount) for amount in amounts]
            unique = list(dict.fromkeys(floats))
            return unique

        amounts = find_amounts(extracted_text)
        grand_total = max(amounts) if amounts else 0

        return grand_total, extracted_text

if __name__ == "__main__":
    image_path = sys.argv[1]
    action = sys.argv[2]
    if action == 'scan':
        output_path = process_receipt(image_path, action)
        print(f"Scanned Image Path: {output_path}")
    elif action == 'extract':
        grand_total, extracted_text = process_receipt(image_path, action)
        print(f"Grand Total: {grand_total}")
        print(f"Extracted Text: {extracted_text}")
