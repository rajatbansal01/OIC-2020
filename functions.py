import os
from csv import DictWriter
import sys
import local_utils
import pandas as pd
import time
import requests
#try:
from PIL import Image

import cv2
import numpy as np
#import matplotlib.pyplot as plt
#%matplotlib inline
from local_utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


#laod model from json file
def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)

#processing image.        
def preprocess_image(img,resize=False):
    #img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

#get plate from frame 
# LpImg may contain more than one plate image in list
def get_plate(img, Dmax=608, Dmin=256):
    vehicle = preprocess_image(img)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return(vehicle, LpImg, cor)

# return image to  be passed for OCR, it return a list of different images
def image_for_ocr(LpImg):
    if (len(LpImg)): #check if there is at least one license image
        # Scales, calculates absolute values, and converts the result to 8-bit.
        plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
        #plt.imshow(plate_image)
        #plt.title("Plate image")
        # convert to grayscale and blur the image
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        #blur = cv2.GaussianBlur(gray,(7,7),0)
        
        # Applied inversed thresh_binary 
        binary = cv2.threshold(gray, 180, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        binary= cv2.bitwise_not(binary)
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
    return([plate_image,gray,binary,kernel3,thre_mor])


#finally reurns a string of registration number and a image of plate
def find_num(img):
    #img = cv2.imread(path)
    LpImg =0
    try:
        vehicle, LpImg,cor = get_plate(img)
    except AssertionError:
        pass
    s=[]
    if LpImg:
        ocr_image=image_for_ocr(LpImg)[0]
        img_tosave = image_for_ocr(LpImg)[2]
        number= pytesseract.image_to_string(ocr_image,lang="eng")
        for i in number:
            if i.isalnum() and (not i.islower()):
                s.append(i)
        return("".join(s),img_tosave)
    else:
        return(0,0)
    
# draw box on plate in image if found
def draw_box(image_path, cor, thickness=3):
    pts=[]  
    x_coordinates=cor[0][0]
    y_coordinates=cor[0][1]
    # store the top-left, top-right, bottom-left, bottom-right 
    # of the plate license respectively
    for i in range(4):
        pts.append([int(x_coordinates[i]),int(y_coordinates[i])])
    
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1,1,2))
    vehicle_image = preprocess_image(image_path)
    
    cv2.polylines(vehicle_image,[pts],True,(0,255,0),thickness)
    return vehicle_image

#detect face from a image (ndarray) and return face if found otherwise the same frame.
def detect(frame):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #changing RGB to gray for better classification
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    if faces is not ():
        for (x,y,w,h) in faces:
            cv2.rectangle(frame ,(x-w//2,y-h//2), (x+2*w ,y+2*h),(255,0,0),2)
            img = frame[y-h//2:y+2*h , x-w//2:x+2*w]
        return(img)
    else:
        return(frame)

wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)

#number plate csv updater
def csv_updater(v_number):
    name = time.asctime()[4:16]
    name = name.replace(":", "_")
    with open('data_new.csv','a', newline = '') as f:
        csv_file = DictWriter(f, fieldnames = ["date","v_number",'plate_path','face_path'])
        csv_file.writerows([
        {'date': [time.asctime( time.localtime(time.time()) )], 'v_number': v_number,
         'plate_path':"plates/"+v_number+ ".jpg",'face_path':"faces/"+name+".jpg"}
            ])
    f.close()
        

# plate path updater        
def plate_updater():    
        with open('data_new.csv','a', newline = '') as f:
            csv_file = DictWriter(f, fieldnames = ["date","v_number"])
            csv_file.writerows([
            {'date': [time.asctime( time.localtime(time.time()) )], 'v_number': v_number}
                ])
        f.close()
        
        
#find a most common string from list
def most_frequent(List): 
    dict = {} 
    count, itm = 0, '' 
    for item in reversed(List): 
        dict[item] = dict.get(item, 0) + 1
        if dict[item] >= count : 
            count, itm = dict[item], item 
    return(itm)
# erode image 
def erode(image):
    kernel = np.ones((5,5), np.uint8)
    # Now we erode
    erosion = cv2.erode(image, kernel, iterations = 1)
    return(erosion)




#save the image of face name is the current time date
def save_face(img):
    name = time.asctime()[4:16]
    name = name.replace(":", "_")
    cv2.imwrite("faces/"+name+".jpg" , img )
    
# save the image of plate with the name as number 
def save_plate(image,number):
    cv2.imwrite("plates/"+number+ ".jpg" , image)
    
    