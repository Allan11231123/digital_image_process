import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import math
from scipy import signal

def image_flip(arr):
    result = []
    for i in range(0,len(arr)):
        result.insert(0,arr[i])
    return np.array(result)

def show_image(image,image_name,output_dir):
    output = os.path.join(output_dir,image_name)
    cv2.imwrite(output,image)
    # cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def change_to_grayscale(arr): #input is constraint to a ndarray or a list
    temp = [[] for i in range(len(arr))]
    for i in range(0,len(arr)):
        for j in range(0,len(arr[i])):
            grayscale = 0.2989*arr[i][j][2] + 0.587*arr[i][j][1] + 0.114*arr[i][j][0]
            temp[i].append(grayscale)
    result = np.array(temp).astype(np.uint8)
    return result

def intensity_change(arr,dir,factor):
    temp_list = [[] for i in range(len(arr))]
    if dir == -1:
        for i in range(len(arr)):
            for j in arr[i]:
                temp_list[i].append(j/factor)
    elif dir == 1:
        for i in range(len(arr)):
            for j in arr[i]:
                if (j*factor) >255:
                    temp_list[i].append(255)
                else:    
                    temp_list[i].append(j*factor)
    else :
        return arr
    result = np.array(temp_list,dtype = object).astype(np.uint8)
    # print(result)
    return result

def generate_image_histogram_value(image): 
    result_count = []
    value_range = []
    for i in range(256):
        result_count.append(np.sum(image == i))
        value_range.append(i)
    return result_count, value_range

# def generate_image_histogram(image: np.ndarray, image_name): # TODO: write the result graph into a specific folder
#     plt.hist(image.flatten(),range=(0,270),bins=256)
#     plt.title('Histogram of {}'.format(image_name))
#     plt.xlabel('Pixel Value')
#     plt.ylabel('Pixel Number')
#     # plt.savefig('./generated_images/{}_hist.png'.format(image_name))
#     plt.show()

def generate_histogram_equalization_lookup(image):
    pixel_value_count, _pixel_value_range  = generate_image_histogram_value(image)
    shape = image.shape
    total_pixel = shape[0]*shape[1]
    pdf_list =  [(x/total_pixel)*255 for x in pixel_value_count]
    cdf_list = [0 for i in range(len(pdf_list))]
    for i in range(len(pdf_list)):
        cdf_list[i] = sum(pdf_list[:i+1])
    for i in range(len(cdf_list)):
        cdf_list[i] = round(cdf_list[i])
    return cdf_list

def histogram_equalization_global(image): # Assume the image is a grayscale image
    lookup = generate_histogram_equalization_lookup(image)
    for i in range(len(image)):
        for j in range(len(image[i])):
            image[i][j] = lookup[(image[i][j])]
    return image

def transfer_for_local(image):
    
    min = image.min()
    max = image.max()
    if min == max:
        return image
    tan = 255 / (max-min) #斜率
    dis = 0 - tan*min #截距
    for i in range(len(image)):
        for j in range(len(image[i])):
            image[i][j] = int(tan * image[i][j] + dis)
    
    return image

def histogram_equalization_local(image,factor):
    shape = image.shape
    if shape[0] % factor != 0:
        print("Cannot factor input image into smaller part, try using other parameter.")
        return image
    else:
        length = int(shape[0]/factor)
    if shape[1] % factor != 0:
        print("Cannot factor input image into smaller part, try using other parameter.")
        return image
    else:
        width = int(shape[1]/factor)
    le = 0

    for i in range(factor):
        wi = 0
        for j in range(factor):
            sampled = image[le:le+length,wi:wi+width]
            equalized = transfer_for_local(sampled)
            image[le:le+length,wi:wi+width] = equalized
            wi += width
        le += length
    return image


def transfer(image, breakpoint):
    darkside_tan = (breakpoint[1])/(breakpoint[0])
    brightside_tan = (255-breakpoint[1]) / (255-breakpoint[0])
    brightside_dis = breakpoint[1] - brightside_tan*breakpoint[0]
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] < breakpoint[0]:
                image[i][j] = int(image[i][j] * darkside_tan)
            else: 
                image[i][j] = int(image[i][j] * brightside_tan + brightside_dis)
    return image

def calculate_psnr(image1, image2):
    mse = np.mean((image1/1.0 - image2/1.0)**2)
    if mse < 1e-10:
        return 100
    psnr = 20 * math.log10(255/ math.sqrt(mse))
    return psnr

def gaussian_kernel(size, sigma):
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)
def gaussian_filter(image,ker_s,sigma):
    kernel = gaussian_kernel(ker_s,sigma)
    result = signal.convolve2d(image,kernel,mode='same',boundary='symm')
    return result

def median_filter(image,mask_s):
    temp = image

    for i in range(len(image)):
        for j in range(len(image[i])):
            mask = temp[i:i+mask_s,j:j+mask_s]
            image[i][j] = int(np.median(mask))
    return image

def main():
    argparser = argparse.ArgumentParser(description="Script for homework1")
    argparser.add_argument(
        "-i",
        "--input",
        type=str,
        help='input folder where sample images stored'
    )
    argparser.add_argument(
        '-o',
        '--output',
        type=str,
        help='folder to store output images',
        default=os.getcwd()
    )
    arg = argparser.parse_args()

    input_dir = arg.input
    output_dir = os.path.join(arg.output,'generated_images')
    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)
    image1 = cv2.imread("{}/sample1.png".format(input_dir))
    result1 = image_flip(image1)
    show_image(result1,'result1.png',output_dir)
    result2 = change_to_grayscale(result1)
    show_image(result2,'result2.png',output_dir) 

    image2 = cv2.imread("{}/sample2.png".format(input_dir))
    result3 = intensity_change(change_to_grayscale(image2),-1,3)
    show_image(result3,'result3.png',output_dir)
    result4 = intensity_change(result3,1,3)
    show_image(result4,'result4.png',output_dir)
    image2_gray = change_to_grayscale(image2)

    image2_equalized = histogram_equalization_global(image2_gray)
    result3_equalized = histogram_equalization_global(result3)
    result4_equalized = histogram_equalization_global(result4)
    show_image(image2_equalized,'result5.png',output_dir)
    show_image(result3_equalized,'result6.png',output_dir)
    show_image(result4_equalized,'result7.png',output_dir)
    result8 = histogram_equalization_local(image2_gray,40)
    show_image(result8,'result8.png',output_dir)

    image3 = cv2.imread("{}/sample3.png".format(input_dir))
    image3_gray = change_to_grayscale(image3)
    result9 = transfer(image3_gray,(50,200))
    show_image(result9,'result9.png',output_dir)

    image4 = cv2.imread("{}/sample4.png".format(input_dir))
    image4_gray = change_to_grayscale(image4)

    image5 = cv2.imread("{}/sample5.png".format(input_dir))
    image5_gray = change_to_grayscale(image5)
    result10 = gaussian_filter(image5_gray,3,2)
    show_image(result10,'result10.png',output_dir)

    image6 = cv2.imread("{}/sample6.png".format(input_dir))
    image6_gray = change_to_grayscale(image6)
    result11 = median_filter(image6_gray,4)
    show_image(result11,'result11.png',output_dir)

    print('the PSNR of result10 and sample5 is {}'.format(calculate_psnr(result10,image5_gray)))
    print('the PSNR of result11 and sample6 is {}'.format(calculate_psnr(result11,image6_gray)))

main()