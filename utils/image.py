import cv2
import torch

def load_image(filename):
    return cv2.imread(filename)

def resize_image(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

def fastest_det_preprocess(image, input_height, input_width, device="cpu"):
    img = image.reshape(1, input_height, input_width, 3)
    img = torch.from_numpy(img.transpose(0, 3, 1, 2))
    img = img.to(device).float() / 255.0
    return img

def load_test_image(filename, model_type, width, height, device="cpu"):
    if model_type == "fastestdet":  
        image =  load_image(filename)
        original_image = image.copy()
        image = resize_image(image, width, height)
        preprocessed = fastest_det_preprocess(image, width, height,device)
        return preprocessed, original_image
    else :
        raise ValueError("only fastestdet is implemented till now")