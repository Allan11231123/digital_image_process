import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal
from enum import Enum

def show_image(image):
    print("The shape of the given image is {}.".format(image.shape))
    print("The data type of the given image is {}.".format(image.dtype))
    # cv2.imwrite('{}'.format(image_name),image)
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_two_image(image1,image2):
    print("The shape of the given image is {}.".format(image1.shape))
    print("The shape of the given image is {}.".format(image2.shape))
    print("The data type of the given image is {}.".format(image1.dtype))
    print("The data type of the given image is {}.".format(image2.dtype))
    cv2.imshow('image1',image1)
    cv2.imshow('image2',image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def image_save(image, image_name):
    cv2.imwrite('{}'.format(image_name),image)
    
# def generate_image_histogram_value(image): 
#     result_count = []
#     value_range = []
#     for i in range(256):
#         result_count.append(np.sum(image == i))
#         value_range.append(i)
#     return result_count, value_range

def generate_image_histogram(image: np.ndarray, image_name): # TODO: write the result graph into a specific folder
    plt.hist(image.flatten(),range=(0,270),bins=256)
    plt.title('Histogram of {}'.format(image_name))
    plt.xlabel('Pixel Value')
    plt.ylabel('Pixel Number')
    # plt.savefig('{}_hist.png'.format(image_name))
    plt.show()

def change_to_grayscale(arr): #input is constraint to a ndarray or a list
    temp = [[] for i in range(len(arr))]
    for i in range(0,len(arr)):
        for j in range(0,len(arr[i])):
            grayscale = 0.2989*arr[i][j][2] + 0.587*arr[i][j][1] + 0.114*arr[i][j][0]
            temp[i].append(grayscale)
    result = np.array(temp).astype(np.uint8)
    return result

def hit(target,pat):
    if pat==0 and target!=0:
        return False
    elif pat!=0 and target==0:
        return False
    return True
def hit_four_ero(target):
    pattern = np.array([[0,1,0],[1,1,1],[0,1,0]])
    if not np.any(target) or not target[1][1]:
        return False
    if target[2][1] and target[1][0] and target[0][1] and target[1][2]:
        return False
    result = target*pattern
    if np.any(result):
        return True
    else:
        return False
def hit_four_dil(target):
    pattern = np.array([[1,1,1],[1,0,1],[1,1,1]])
    if target[1][1]:
        return False
    result = target*pattern
    if np.any(result):
        return True
    else:
        return False
    
def erosion(image):
    temp_image = np.pad(image,((2,2),(2,2)),'constant',constant_values=(0,0))
    temp = image
    for i in range(len(image)):
        for j in range(len(image[i])):
            if hit_four_ero(temp_image[i:i+3,j:j+3]):
                temp[i][j] = 0
    result = np.array(temp).astype(np.uint8)
    return result
def dilation(image):
    temp_image = np.pad(image,((2,2),(2,2)),'constant',constant_values=(0,0))
    temp = image
    for i in range(len(image)):
        for j in range(len(image[i])):
            if hit_four_dil(temp_image[i:i+3,j:j+3]):
                temp[i][j] = 255
    result = np.array(temp).astype(np.uint8)
    return result

def erode(image,iteration=2):
    temp_image = image
    for i in range(iteration):
        temp_image = erosion(temp_image)
    return temp_image
def dilate(image,iteration=2):
    temp_image = image
    for i in range(iteration):
        temp_image = dilation(temp_image)
    return temp_image

#1.
def output_boundary(image):
    temp_image_ero = erode(image,iteration=2)
    temp_image_dil = dilate(image,iteration=1)
    result = temp_image_dil-temp_image_ero
    return result

#2.
def is_connected(target,i,j):
    return np.any(target[i-2:i+2,j-2:j+2])
def hole_filling(image):
    check = np.pad(np.array([[0 for j in range(len(image[0]))] for i in range(len(image))]),((1,1),(1,1)),'constant',constant_values=(0,0))
    temp = image
    background = image[0][0]
    check[1][1] = 1
    check[image.shape[0]+1][image.shape[1]+1] = 1

    for i in range(len(image)):
        for j in range(len(image[i])):
            if temp[i][j] == background and is_connected(check,i+1,j+1):
                check[i+1][j+1] = 1
                
    for i in reversed(range(len(image))):
        for j in reversed(range(len(image[i]))):
            if is_connected(check,i+1,j+1) and temp[i][j] == background:
                check[i+1][j+1] = 1
    for i in range(len(image)):
        for j in range(len(image[i])):
            if temp[i][j] == background and not is_connected(check,i+1,j+1):
                temp[i][j] = 255
    return temp

#3.
def median_filter(image,mask_s):
    temp = image

    for i in range(len(image)):
        for j in range(len(image[i])):
            mask = temp[i:i+mask_s,j:j+mask_s]
            image[i][j] = int(np.median(mask))
    return image
def gaussian_kernel(size, sigma):
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)
def gaussian_filter(image,ker_s,sigma):
    kernel = gaussian_kernel(ker_s,sigma)
    result = signal.convolve2d(image,kernel,mode='same',boundary='symm')
    return result
def eliminate_noise(image):
    temp_image = erode(image,iteration=10)
    temp_image = dilate(temp_image,iteration=10)
    return temp_image

image1 = change_to_grayscale(cv2.imread("./hw3_sample_images-2/hw3_sample_images/sample1.png"))
result1 = output_boundary(image1)
show_image(result1)
image_save(result1,'result1.png')
result2 = hole_filling(image1)
show_image(result2)
image_save(result2,'result2.png')
result3 = eliminate_noise(image1)
show_image(result3)
image_save(result3,'result3.png')