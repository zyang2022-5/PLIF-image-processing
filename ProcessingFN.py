import sys
import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

FILTER_KER_SIZE = 31
BIN_SIZE = 10
INT_CUTOFF = 1.5
WL_STEP_SIZE =0.0002
TEMP_CUTOFF = 600.0
FILTERING = True
FIT_BOUNDED = True
test_pixels = [(55,75), (58, 75), (66,74), (74,74), (87,72), (106, 74), (122, 75), (135, 76), (150, 75), (157, 75), (168, 75), (175, 76)]


def image_import(folder_path):
    #full_path = "C:/Users/test/Desktop/AR_642_10974/data/" + folder_name

    imgs = glob.glob(folder_path + "/*.tif")
    if imgs is None:
        sys.exit("Could not read the image.")
    return imgs

def ave_image(folder_name):
    images = image_import(folder_name)
    image_data = []
    for img in images:
        this_image = cv.imread(img, 1)
        image_data.append(this_image)

    ave_image = image_data[0]
    for i in range(len(image_data)):
        if i != 0:
            alpha = 1.0/(i+1)
            beta = 1.0 - alpha
            ave_image = cv.addWeighted(image_data[i], alpha, ave_image, beta, 0.0)
    return ave_image

def median_filter(im):
    return cv.medianBlur(im, FILTER_KER_SIZE)

def image_subtraction(im1, im2):
    return cv.subtract(im1, im2)

def image_bin(im, bin_size_x, bin_size_y, bin_alg="binning"):
    reso = im.shape
    scale_x = reso[1] // bin_size_x
    scale_y = reso[0] // bin_size_y
    if reso[1] % bin_size_x != 0 or reso[0] % bin_size_y != 0:
        sys.exit("bin size not divisable for resolution")

    if bin_alg == 'bilinear':
        return cv.resize(im, dsize=(scale_x, scale_y), interpolation= cv.INTER_LINEAR)
    else:
        test = im.reshape(scale_y, bin_size_y, scale_x, bin_size_x)
        return test.mean(axis=3).mean(axis=1)

def gaussian_func(x, a, x0, sigma, c):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))+c

def calc_temp(pixel_vals, step_size, bounded_fit):
    x_data = np.arange(1.0, 1.0+len(pixel_vals)*step_size, step_size)
    try:
        if bounded_fit:
            params, _ = curve_fit(gaussian_func, x_data, pixel_vals, [np.max(pixel_vals), 1.0+np.argmax(pixel_vals)*step_size, step_size*2, 0], bounds=([np.min(pixel_vals), 1.0, 0.0001, -np.inf], [np.max(pixel_vals)*1.3, 1.0028, 0.0010, np.inf]), method= "dogbox")
        else:
            params, _ = curve_fit(gaussian_func, x_data, pixel_vals, [np.max(pixel_vals), 1.0+np.argmax(pixel_vals)*step_size, step_size, 0])
        temp = (params[2]**2)*8.893e8    #calculates the temperature
        if temp < TEMP_CUTOFF: # cutoff 
            return temp   
        else:
            return 0
    except RuntimeError:
        return 0

def color_pixel_selection(pixels_g, pixels_r):
    mean_g = np.mean(pixels_g)
    mean_r = np.mean(pixels_r)
    if mean_g > mean_r:
        return pixels_g
    else:
        return pixels_r

def gaussian_fit(image_array_g, image_array_r):
    _, y, x = image_array_g.shape
    temp_map = np.zeros((y, x))

    for i in range(y):
        for j in range(x):
            vals_g = image_array_g[:, i, j]
            vals_r = image_array_r[:, i, j]
            if (vals_r > (np.max(vals_r)-0.3)).sum() > 2:                     #
                vals_r = np.zeros(len(vals_r))
            if np.mean(vals_g) > INT_CUTOFF or np.mean(vals_r) > INT_CUTOFF:
                temp_map[i, j] = calc_temp(color_pixel_selection(vals_g, vals_r), WL_STEP_SIZE, FIT_BOUNDED)

    return temp_map

def plot_pixel_fit(pos_array, image_array_g, image_array_r, step_size):

    for pos in pos_array:
        vals_g = image_array_g[:, pos[1], pos[0]]
        vals_r = image_array_r[:, pos[1], pos[0]]
        if (vals_r > (np.max(vals_r)-0.3)).sum() > 2:                     #
            vals_r = np.zeros(len(vals_r))
        if np.mean(vals_g) > INT_CUTOFF or np.mean(vals_r) > INT_CUTOFF:
            pixel_vals = color_pixel_selection(vals_g, vals_r)
            x_data = np.arange(1.0, 1.0+len(pixel_vals)*step_size, step_size)
            
            if FIT_BOUNDED:
                params, _ = curve_fit(gaussian_func, x_data, pixel_vals, [np.max(pixel_vals), 1.0+np.argmax(pixel_vals)*step_size, step_size*2, 0], bounds=([np.min(pixel_vals), 1.0, 0.0001, -np.inf], [np.max(pixel_vals)*1.3, 1.0028, 0.0008, np.inf]), method= "dogbox")
            else:
                params, _ = curve_fit(gaussian_func, x_data, pixel_vals, [np.max(pixel_vals), 1.0+np.argmax(pixel_vals)*step_size, step_size, 0])
            temp = (params[2]**2)*8.893e8    #calculates the temperature
            plt.figure()
            plt.scatter(x_data, pixel_vals)
            x_line = np.linspace(1.0, 1.0+len(pixel_vals)*step_size, 1000)
            plt.plot(x_line, gaussian_func(x_line, params[0], params[1], params[2], params[3]))
    plt.show()

image_folders = [f.path for f in os.scandir('C:/users/test/Desktop/AR_642_10974/data') if f.is_dir()]
background = ave_image('C:/Users/test/Desktop/AR_642_10974/no laser beg')
if FILTERING:
    background = median_filter(background)
processed_im_g = []
processed_im_r = []

for folder in image_folders:
    image = ave_image(folder)
    if FILTERING:
        image = median_filter(image)
    subtracted_image = image_subtraction(image, background)
    _,g,r = cv.split(subtracted_image)
    binned_im_g = image_bin(g, BIN_SIZE, BIN_SIZE)
    processed_im_g.append(binned_im_g)
    #processed_im_g.append(np.zeros((160, 256)))

    binned_im_r = image_bin(r, BIN_SIZE, BIN_SIZE)
    #processed_im_r.append(binned_im_r)
    processed_im_r.append(np.zeros((160, 256)))

temp_map = gaussian_fit(np.asarray(processed_im_g), np.asarray(processed_im_r))

fig = plt.figure()
plt.imshow(temp_map, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()

plot_pixel_fit(test_pixels, np.asarray(processed_im_g), np.asarray(processed_im_g), WL_STEP_SIZE)


#print(np.arange(0.0, 14*0.0002, 0.0002))

'''
im1 = ave_image("696.7350 full power")
im2 = ave_image("no laser beg")
subtracted = cv.subtract(im1, im2)

alpha = 2.0
beta = 4.0
final_image = cv.convertScaleAbs(subtracted, alpha, beta)
print(final_image.shape)
b,g,r = cv.split(final_image)
print(g.shape)
binned_im = np.round(image_bin(r, 5, 5))
print(binned_im.shape)

#cv.namedWindow("Image", cv.WINDOW_NORMAL)
#cv.namedWindow("stuff", cv.WINDOW_NORMAL)
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(r)
plt.title("R")
fig.add_subplot(1,2,2)
plt.imshow(binned_im)
plt.title("Binned")
plt.show()
#cv.imwrite('C:/Users/test/Desktop/test.tif', binned_im)
#cv.waitKey(0)
#cv.destroyAllWindows()
'''
'''
a = np.arange(32).reshape(4,8)
print(a)
a_view = a.reshape(4 // 2, 2, 8 // 4, 4)
print(a_view)
print('/////////////')
print(a_view.mean(axis=3).mean(axis=1))
'''