import cv2
import numpy as np

import matplotlib.pyplot as plt
import math
from scipy import signal
from enum import Enum
from collections import Counter


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

def dithering(img,dither_size): #dither size is power of 2
    n = int(math.log2(dither_size))
    dither = np.array([[1,2],[3,0]])
    N = 2
    for _ in range(n-1):
        temp_I = np.zeros((N*2,N*2))
        temp_I[0:N,0:N] = 4*dither+1
        temp_I[N:2*N,0:N] = 4*dither+2
        temp_I[0:N,N:2*N] = 4*dither+3
        temp_I[N:2*N,N:2*N] = 4*dither
        dither = temp_I
        N*=2
    x,y = img.shape
    threshold = 255*(dither+0.5)/(N*N)
    res = np.zeros((x,y))
    res[0:(x//N)*N,0:(y//N)*N] = np.tile(threshold,(x//N,y//N))
    res[(x//N)*N:x][:] = res[0:x-(x//N)*N][:]
    res[:,(y//N)*N:y] = res[:,0:y-(y//N)*N]
    result = np.zeros((x,y))
    result[img>=res] = 255
    return result

def error_diffusion(image,filter):
    if filter == 'Floyd_Steinberg':
        mask = np.array([[0,0,0],[0,0,7],[3,5,1]])/16
        mask_r = np.array([[0,0,0],[7,0,0],[1,5,3]])/16
    elif filter == 'Jarvis':
        mask = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,7,5],
                         [3,5,7,5,3],[1,3,5,3,1]])/48
        mask_r = np.array([[0,0,0,0,0],[0,0,0,0,0],[5,7,0,0,0],
                           [3,5,7,5,3],[1,3,5,3,1]])/48
    else:
        print('Please input the correct filter type, Floyd_Steinberg or Jarvis.')
        return image
    pad_size = mask.shape[0]//2
    x,y = image.shape
    target = np.pad(image,(pad_size,pad_size),'constant')/255
    result = np.zeros((x+2*pad_size,y+2*pad_size))
    for i in range(pad_size,x+pad_size):
        if (i-pad_size)%2==0:
            j = pad_size
            while j<y+pad_size:
                if target[i,j]>0.5:
                    result[i,j] = 1
                else:
                    result[i,j] = 0
                error = target[i,j] - result[i,j]
                target[i-pad_size:i+pad_size+1,j-pad_size:j+pad_size+1] += mask*error
                j+=1

        else:
            j = y+pad_size-1
            while j>=pad_size:
                if target[i,j]>0.5:
                    result[i,j] = 1
                else:
                    result[i,j] = 0
                error = target[i,j]-result[i,j]
                target[i-pad_size:i+pad_size+1,j-pad_size:j+pad_size+1] += mask_r*error
                j-=1
    return result[pad_size:x+pad_size,pad_size:y+pad_size]*255

def construct_training_data():
    image = change_to_grayscale(cv2.imread("./hw4_sample_images-2/hw4_sample_images/TrainingSet.png"))
    target = np.zeros(image.shape)
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j] < 200:
                target[i][j] = 1
    
    return target.astype(np.int8)

def construct_reference():
    image = construct_training_data()
    num_label,labels = cv2.connectedComponents(image)
    print("there are {} object in the given image".format(num_label))
    for i in range(num_label):
        image[labels==i] = i/num_label*255
    return image

def generate_binary_image(image):
    result = np.zeros(image.shape)
    result[image>0] = 1
    return result.astype(np.int8)
def calculate_bit_quad(image,patterns):
    result = 0
    x,y = image.shape
    for i in range(x-1):
        for j in range(y-1):
            for pattern in patterns:
                if (image[i:i+len(pattern),j:j+len(pattern)]==pattern).all():
                    result+=1
    return result
def bit_quad_ratio(image):
    x,y = image.shape
    q1 = [
    np.array([[1,0],[0,0]]),
    np.array([[0,1],[0,0]]),
    np.array([[0,0],[1,0]]),
    np.array([[0,0],[0,1]])]
    q2 = [
    np.array([[1,1],[0,0]]),
    np.array([[0,1],[0,1]]),
    np.array([[1,0],[1,0]]),
    np.array([[0,0],[1,1]])]
    q3 = [
    np.array([[1,1],[1,0]]),
    np.array([[1,0],[1,1]]),
    np.array([[1,1],[0,1]]),
    np.array([[0,1],[1,1]])]
    q4 = [np.array([[1,1],[1,1]])]
    qd = [
    np.array([[1,0],[0,1]]),
    np.array([[0,1],[1,0]])]
    gray_area = 0
    gray_perimeter = 0
    duda_area = 0
    duda_perimeter_1 = 0
    duda_perimeter_2 = 0
    for i in range(x-1):
        for j in range(y-1):
            grab_local = np.array(image[i:i+2, j:j+2])
            if np.any([np.array_equal(grab_local, q) for q in q1]):
                gray_area += 1
                gray_perimeter += 1
                duda_area += 2
                duda_perimeter_2 += 1
            elif np.any([np.array_equal(grab_local, q) for q in q2]):
                gray_area += 2
                gray_perimeter += 1
                duda_area += 4
                duda_perimeter_1 += 1
            elif np.any([np.array_equal(grab_local, q) for q in q3]):
                gray_area += 3
                gray_perimeter += 1
                duda_area += 7
                duda_perimeter_2 += 1
            elif np.any([np.array_equal(grab_local, q) for q in q4]):
                gray_area += 4
                gray_perimeter += 0
                duda_area += 8
            elif np.any([np.array_equal(grab_local, q) for q in qd]):
                gray_area += 2
                gray_perimeter += 2
                duda_area += 6
                duda_perimeter_2 += 2
    gray_ratio = (gray_area/4)/gray_perimeter
    gray_cir = 4*np.pi*gray_area/(gray_perimeter**2)
    duda_perimeter = duda_perimeter_1+duda_perimeter_2/(2**0.5)
    duda_ratio = (duda_area/8)/duda_perimeter
    duda_cir = 4*np.pi*duda_area/(duda_perimeter**2)
    return gray_ratio,gray_cir,duda_ratio,duda_cir

def generate_reference_table():
    result_gray = []
    result_duda = []
    table = ['A','B','C','D','E','F','G','H','I',
             'J','K','L','M','N','O','P','Q','R',
             'S','T','U','V','W','X','Y','Z','0',
             '1','2','3','4','5','6','7','8','9']
    image = construct_training_data()
    temp = generate_binary_image(image)
    num_of_label,labeled = cv2.connectedComponents(temp)
    assert num_of_label==len(table)+1, 'the numbers of connected components is incorrect, detected {}'.format(num_of_label)

    visited = [False for _ in range(len(table))]
    reference_table =[]
    for i in range(len(labeled)):
        for j in range(len(labeled[0])):
            if labeled[i][j] > 0 and not visited[labeled[i][j]-1]:
                visited[labeled[i][j]-1] = True
                reference_table.append(labeled[i][j])
    print('the reference table is {}\n'.format(reference_table))

    for i in range(num_of_label):
        res = np.zeros(labeled.shape)
        res[labeled==reference_table[i]]=1
        target = table[i]
        gray_ratio, gray_cir, duda_ratio, duda_cir = bit_quad_ratio(res)
        result_gray.append((target,gray_ratio,gray_cir))
        result_duda.append((target,duda_ratio,duda_cir))
    print(result_gray)
    return result_gray,result_duda

def find_sign(image,references,background=1):
    result = ''
    target = np.zeros(image.shape).astype(np.int8)
    temp = generate_binary_image(image)
    if background!=0:
        target[temp==1]=1
    else:
        target[temp!=1]=1
    num_of_label,labeled = cv2.connectedComponents(target)
    print('the objects of given image is {}'.format(num_of_label))
    for i in range(num_of_label):
        res = np.zeros(labeled.shape)
        res[labeled==i]=1
        temp_result=[float('inf') for _ in range(4)]
        temp = [0 for _ in range(4)]
        
        gray,gray_cir,duda,duda_cir = bit_quad_ratio(res)
        for i in range(len(references[0])):
            if abs(references[0][i][1]-gray) < temp_result[0]:
                temp_result[0] = references[0][i][1]-gray
                temp[0] = references[0][i][0]
            if abs(references[0][i][2]-gray_cir) < temp_result[1]:
                temp_result[1] = references[0][i][2]-gray_cir
                temp[1] = references[0][i][0]
            if abs(references[1][i][1]-duda) < temp_result[2]:
                temp_result[1] = references[1][i][1]-duda
                temp[2] = references[1][i][0]
            if abs(references[1][i][2]-duda_cir) < temp_result[3]:
                temp_result[3] = references[1][i][2]-duda_cir
                temp[3] = references[1][i][0]
        
        result += Counter(temp).most_common()[0][0]
    print(result)

def main():
    image1 = change_to_grayscale(cv2.imread("./hw4_sample_images-2/hw4_sample_images/sample1.png"))
    image2 = change_to_grayscale(cv2.imread("./hw4_sample_images-2/hw4_sample_images/sample2.png"))
    image3 = change_to_grayscale(cv2.imread("./hw4_sample_images-2/hw4_sample_images/sample3.png"))
    image4 = change_to_grayscale(cv2.imread("./hw4_sample_images-2/hw4_sample_images/sample4.png"))

    #problem1
    result1 = dithering(image1,2)
    image_save(result1,'result1.png')
    result2 = dithering(image1,256)
    image_save(result2,'result2.png')
    result3 = error_diffusion(image1,'Floyd_Steinberg')
    image_save(result3,'result3.png')
    result4 = error_diffusion(image1,'Jarvis')
    image_save(result4,'result4.png')

    #problem2
    reference_gray, reference_duda = generate_reference_table()
    references = [reference_gray,reference_duda]

    find_sign(image2,references) # output: FK
    find_sign(image3,references,background=0) # output: K
    find_sign(image4,references) # output: FK

if __name__=='__main__':
    main()