import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import time
from datetime import datetime
import torch
import torch.nn as nn
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytesseract
from io import StringIO
from training.model import TableNet

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

TRANSFORM = A.Compose([
                #ToTensor --> Normalize(mean, std)
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value = 255,
                ),
                ToTensorV2()
    ])

model = TableNet(encoder = 'densenet', use_pretrained_model = True, basemodel_requires_grad = True)
model.eval()

#load checkpoint
model.load_state_dict(torch.load("densenet_config_4_model_checkpoint.pth.tar")['state_dict'])


def predict(img_path):
    orig_image = Image.open(img_path).resize((1024, 1024))
    test_img = np.array(orig_image.convert('LA').convert("RGB"))

    now = datetime.now()
    image = TRANSFORM(image=test_img)["image"]
    with torch.no_grad():
        image = image.unsqueeze(0)
        # with torch.cuda.amp.autocast():
        table_out, _ = model(image)
        table_out = torch.sigmoid(table_out)

    # remove gradients
    table_out = (table_out.detach().numpy().squeeze(0).transpose(1, 2, 0) > 0.5).astype(np.uint8)

    # get contours of the mask to get number of tables
    contours, table_heirarchy = cv2.findContours(table_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    table_contours = []
    # ref: https://www.pyimagesearch.com/2015/02/09/removing-contours-image-using-python-opencv/
    # remove bad contours
    for c in contours:

        if cv2.contourArea(c) > 3000:
            table_contours.append(c)

    if len(table_contours) == 0:
        print("No Table detected")

    table_boundRect = [None] * len(table_contours)
    for i, c in enumerate(table_contours):
        polygon = cv2.approxPolyDP(c, 3, True)
        table_boundRect[i] = cv2.boundingRect(polygon)

    # table bounding Box
    table_boundRect.sort()

    orig_image = np.array(orig_image)
    # draw bounding boxes
    color = (0, 0, 255)
    thickness = 4

    for x, y, w, h in table_boundRect:
        cv2.rectangle(orig_image, (x, y), (x + w, y + h), color, thickness)

    plt.figure(figsize=(20, 10))
    plt.imshow(orig_image)
    plt.show()

    end_time = datetime.now()
    difference = end_time - now
    # print("Total Time : {} seconds".format(difference))
    time = "{}".format(difference)

    print(f"Time Taken on cpu : {time} secs")

    print("Predicted Tables")

    image = test_img[..., 0].reshape(1024, 1024).astype(np.uint8)

    for i, (x, y, w, h) in enumerate(table_boundRect):
        image_crop = image[y:y + h, x:x + w]
        data = pytesseract.image_to_string(image_crop)
        try:
            df = pd.read_csv(StringIO(data), sep=r'\|', lineterminator=r'\n', engine='python')
            print(f" ## Table {i + 1}")
            df = pd.read_csv(StringIO(data), sep=r'\|', lineterminator=r'\n', engine='python')
            print(df)
        except pd.errors.ParserError:
            try:
                df = pd.read_csv(StringIO(data), delim_whitespace=True, lineterminator=r'\n', engine='python')
                print(f" ## Table {i + 1}")
                print(df)
            except pd.errors.ParserError:
                print(f" ## Table {i + 1}")
                print(df)

predict(img_path = '../marmot_processed/image/10.1.1.180.553_10.jpg')