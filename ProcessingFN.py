import sys
import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

FILTER_KER_SIZE = 21
BIN_SIZE = 5
INT_CUTOFF = 0.35
WL_STEP_SIZE =0.0002
TEMP_CUTOFF = 450.0
FILTERING = True
FIT_BOUNDED = True
test_pixels = [(163,199), (236,209), (287,209), (407, 209)]

#no overexposure
#FOLDER_PATH = 'D:/PLIF data/data4_nofilter/AR_642_10974/data'
#BACKGROUND_PATH = 'D:/PLIF data/data4_nofilter/AR_642_10974/no laser beg'

#red overexposed
FOLDER_PATH = 'D:/PLIF data/data2_nofilter/AR_642_10974/data'
BACKGROUND_PATH = 'D:/PLIF data/data2_nofilter/AR_642_10974/no laser beg'


def image_import(folder_path):
    imgs = glob.glob(folder_path + "/*.tif")
    if imgs is None:
        sys.exit("Could not read the image.")
    return imgs

'''
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
'''

def ave_image(folder_name, red = False):
    images = image_import(folder_name)
    image_data = []
    for img in images:
        this_image = cv.imread(img, 1)
        image_data.append(this_image)

    total_sum = None
    count = None
   # mask = None

    for image in image_data:
        if total_sum is None:
            total_sum = np.zeros_like(image, dtype=np.float64)
            count = np.zeros_like(image, dtype=np.float64)
        mask = image < 255
        total_sum += image * mask
        count += mask
        #count += np.ones_like(image, dtype=np.float64)

    count[count == 0] = 1
    average_image = total_sum / count
    average_image = np.clip(average_image, 0, 255).astype(np.uint8)
    return average_image

'''
    for i, image in enumerate(image_data):
        if (i >= 1):
            curr_mask = image < 252
            mask = np.logical_or(mask, curr_mask)
        if (i == 0):
            mask = image < 252
'''
    

def median_filter(im):
    return cv.medianBlur(im, FILTER_KER_SIZE)

def image_subtraction(im1, im2):
    return cv.subtract(im1, im2)
'''
def image_subtraction(im1, im2):
    result = np.zeros_like(im1, dtype=np.float64)
    result[:, :, 1] = np.clip(im1[:, :, 1] - im2[:, :, 1], 0.0, 255.0)
    result[:, :, 2] = np.clip(im1[:, :, 2] - im2[:, :, 2], 0.0, 255.0)
    return result.astype(np.uint8)
'''

def zero_to_nan(values):
    return np.where(values < 0.5, np.nan, values)

def nan_to_zero(values):
    values[np.isnan(values)] = 0
    return values

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
        #nan_im = zero_to_nan(im)
        #test = nan_im.reshape(scale_y, bin_size_y, scale_x, bin_size_x)
        #temp = np.nanmean(np.nanmean(test, axis=3), axis=1)
        #return nan_to_zero(temp)
    

def gaussian_func(x, a, x0, sigma, c):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))+c

def calc_temp(pixel_vals, step_size, bounded_fit):
    end_val = 1.0+(len(pixel_vals)-1)*step_size
    x_data = np.linspace(1.0, end_val, len(pixel_vals))

    try:
        if bounded_fit:
            params, _ = curve_fit(gaussian_func, x_data, pixel_vals, [np.max(pixel_vals), 1.0+np.argmax(pixel_vals)*step_size, step_size*2, 0], bounds=([np.min(pixel_vals)-0.001, 1.0, 0.0001, -np.inf], [np.max(pixel_vals)*1.3, end_val, 0.0008, np.inf]), method= "dogbox")
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
    if mean_g > mean_r and mean_g > 1.1:
        return pixels_g
    elif mean_r > mean_g and (mean_r+mean_g)/2 > 4.38:   #4.2  r+g/2=3.8
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
            if (vals_r > (np.max(vals_r)-0.2)).sum() > 6:                     
                vals_r = np.zeros(len(vals_r))
            if (vals_g > (np.max(vals_g)-0.2)).sum() > 6:                     
                vals_g = np.zeros(len(vals_g))
            if np.mean(vals_g) > INT_CUTOFF or np.mean(vals_r) > INT_CUTOFF:
                temp_map[i, j] = calc_temp(color_pixel_selection(vals_g, vals_r), WL_STEP_SIZE, FIT_BOUNDED)

    return temp_map

def plot_pixel_fit(pos_array, image_array_g, image_array_r, step_size):

    for pos in pos_array:
        vals_g = image_array_g[:, pos[1], pos[0]]
        vals_r = image_array_r[:, pos[1], pos[0]]
        if (vals_r > (np.max(vals_r)-0.3)).sum() > 100:                     #
            vals_r = np.zeros(len(vals_r))
        if np.mean(vals_g) > INT_CUTOFF or np.mean(vals_r) > INT_CUTOFF:
            pixel_vals = color_pixel_selection(vals_g, vals_r)

            end_val = 1.0+(len(pixel_vals)-1)*step_size
            x_data = np.linspace(1.0, end_val, len(pixel_vals))
            
            if FIT_BOUNDED:
                params, _ = curve_fit(gaussian_func, x_data, pixel_vals, [np.max(pixel_vals), 1.0+np.argmax(pixel_vals)*step_size, step_size*2, 0], bounds=([np.min(pixel_vals), 1.0, 0.0001, -np.inf], [np.max(pixel_vals)*1.3, end_val, 0.0008, np.inf]), method= "trf")
            else:
                params, _ = curve_fit(gaussian_func, x_data, pixel_vals, [np.max(pixel_vals), 1.0+np.argmax(pixel_vals)*step_size, step_size, 0])
            temp = (params[2]**2)*8.893e8    #calculates the temperature
            plt.figure()
            plt.scatter(x_data, pixel_vals)
            x_line = np.linspace(1.0, 1.0+len(pixel_vals)*step_size, 1000)
            plt.plot(x_line, gaussian_func(x_line, params[0], params[1], params[2], params[3]))
    plt.show()
'''
def foreground(foreground_im, signal_im):
    binned_fore = image_bin(foreground_im, BIN_SIZE, BIN_SIZE)
    mask = np.zeros_like(binned_fore, dtype=np.float64)
    cv.rectangle(mask, (69,0), (445,320), 255, -1)
    masked = cv.bitwise_and(binned_fore, binned_fore, mask=mask)
    return cv.addWeighted(signal_im, 0.5, masked, 0.1,0)
'''

'''
img = ave_image(BACKGROUND_PATH)
cv.namedWindow("image", cv.WINDOW_NORMAL)
cv.imshow("image", img)
cv.imwrite("C:/Users/test/Desktop/test_image.tif", img)
cv.waitKey(0)
cv.destroyAllWindows()
'''


image_folders = [f.path for f in os.scandir(FOLDER_PATH) if f.is_dir()]
background = ave_image(BACKGROUND_PATH)
if FILTERING:
    background = median_filter(background)
processed_im_g = []
processed_im_r = []
#test_imgs = []

for folder in image_folders:
    image = ave_image(folder)
  #  subtracted_image = image_subtraction(image, background)
    #test_imgs.append(image)
    if FILTERING:
        image = median_filter(image)
    subtracted_image = image_subtraction(image, background)
    _,g,r = cv.split(subtracted_image)
    binned_im_g = image_bin(g, BIN_SIZE, BIN_SIZE)
    processed_im_g.append(binned_im_g)
    #processed_im_g.append(np.zeros((320, 512)))

    binned_im_r = image_bin(r, BIN_SIZE, BIN_SIZE)
    processed_im_r.append(binned_im_r)
    #processed_im_r.append(np.zeros((320, 512)))
'''
_, b_g, b_r = cv.split(background)
binned_bg_g = image_bin(b_g, BIN_SIZE, BIN_SIZE)
binned_bg_r = image_bin(b_r, BIN_SIZE, BIN_SIZE)

for folder in image_folders:
    image = ave_image(folder)
    _,g,r = cv.split(image)
    binned_im_g = image_bin(g, BIN_SIZE, BIN_SIZE)
    binned_im_r = image_bin(r, BIN_SIZE, BIN_SIZE)
    subtracted_g = image_subtraction(binned_im_g, binned_bg_g)
    subtracted_r = image_subtraction(binned_im_r, binned_bg_r)
    if FILTERING:
        image_g = median_filter(subtracted_g)
        image_r = median_filter(subtracted_r)
    
    processed_im_g.append(subtracted_g)
    #processed_im_g.append(np.zeros((320, 512)))
    processed_im_r.append(subtracted_r)
    #processed_im_r.append(np.zeros((320, 512)))
'''

#for i, img in enumerate(test_imgs):
    #cv.imwrite(os.path.join('D:/test_data', 'img_{}.tif'.format(i)), img)

temp_map = gaussian_fit(np.asarray(processed_im_g), np.asarray(processed_im_r))

'''
fore_ground = cv.imread('D:/PLIF data/data2_nofilter/AR_642_10974/spatial ref/AR_642_10974_1905.tif')
fore_ground = cv.resize(fore_ground, (1280, 800))
resized_image = cv.cvtColor(fore_ground, cv.COLOR_BGR2RGB)
'''
fig = plt.figure()
plt.imshow(temp_map, cmap='hot', interpolation=None)
cbar = plt.colorbar()
#fig.figimage(resized_image, 250, 160, zorder=3, alpha=0.6)
plt.xticks(np.arange(0, 512, step=55.111), labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.yticks(np.arange(0, 320, step=55.111), labels=['0', '1', '2', '3', '4', '5'])
plt.xlabel('Distance (cm)')
plt.ylabel('Distance (cm)')
cbar.ax.set_ylabel('Temperature (K)', rotation=270, labelpad=15)
plt.show()
#372
#plot_pixel_fit(test_pixels, np.asarray(processed_im_g), np.asarray(processed_im_g), WL_STEP_SIZE)



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