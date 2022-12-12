
import numpy as np
import cv2
from numpy.linalg import norm
import math
from tqdm import tqdm

def cosine_similarity(img1, img2):
    vector1=img1.flatten()
    vector1=vector1.astype(np.float64)
    
    vector2=img2.flatten()
    vector2=vector2.astype(np.float64)
    
    dot_product =np.dot(vector1,vector2)
    norm1 =norm(vector1)
    norm2 =norm(vector2)
    
    result = dot_product / (norm1*norm2)
    return result

def correlation_cofficient(img1, img2):
    
    vector1=img1.flatten()
    vector1=vector1.astype(np.float64)
    
    vector2=img2.flatten()
    vector2=vector2.astype(np.float64)
    
    LEN =  len(vector1)
    mean1 = [np.mean(vector1)] * LEN
    mean2 = [np.mean(vector2)] * LEN

    standard_deviation_1 = np.std(vector1)
    standard_deviation_2 = np.std(vector2)
    
    covariance = np.dot((vector1-mean1),(vector2-mean2)) / LEN
    correlation = covariance / (standard_deviation_1* standard_deviation_2);
    return correlation

def create_template(img):
    roi=cv2.selectROI(img)
    roi_cropped=img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    
    cv2.imwrite('template.jpg',roi_cropped)

def template_matching(img:np.ndarray,
                      template:np.ndarray,
                      mode ="cosine_similarity")-> np.ndarray:
    
    Height, Width =template.shape[:2]
    H,W =img.shape[:2]
    score_matrix = np.zeros(shape=(H -Height,W -Width),dtype=np.float64)
    if mode== "cosine_similarity":
        distance = cosine_similarity
    elif mode== "correlation_cofficient": 
        distance = correlation_cofficient
    else:
        print(f'Mode not matched !!!')
        return score_matrix
    
    for h in tqdm(range(H -Height)):
        for w in range(W -Width):
            temp =img[h:h+ Height,w:w+Width] 
            dis = distance(temp,template)
            score_matrix[h][w] =dis
    return  score_matrix

def brg_to_gray(img):
    img_gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img_gray

def draw_template(img,location:list):
    img_clone = img.copy()
    for loc in location:
        h,w,width,height =loc
        img_clone= cv2.rectangle(img_clone,(w,h),(w+width,h+height),(0,255,0),1)
    
    return img_clone