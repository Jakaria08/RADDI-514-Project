from __future__ import print_function, division
from PIL import Image
from glob import glob
import pandas as pd
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening, convex_hull_image
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pydicom
import scipy.misc
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
import cv2

def get_segmented_lungs(im, plot=False):
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    plt_number = 0
    # Original image label: 0
    if plot:
        f, plots = plt.subplots(12, 1, figsize=(5, 40))
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(im, cmap=plt.cm.bone)
        plt_number += 1

    # Step 1: Convert into a binary image.
    # image label: 1
    binary = im < -604
    if plot:
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(binary, cmap=plt.cm.bone)
        plt_number += 1
    # Step 2: Remove the blobs connected to the border of the image.
    # image label: 2
    cleared = clear_border(binary)
    if plot:
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(cleared, cmap=plt.cm.bone)
        plt_number += 1
    # Step 3: Label the image.
    # image label: 3
    label_image = label(cleared)
    if plot:
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(label_image, cmap=plt.cm.bone)
        plt_number += 1

    # Step 4: Keep the labels with 2 largest areas and segment two lungs.
    # image label: 4
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    labels = []
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
            else:
                coordinates = region.coords[0]
                labels.append(label_image[coordinates[0], coordinates[1]])
    else:
        labels = [1, 2]
    if plot:
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(label_image, cmap=plt.cm.bone)
        plt_number += 1
    # Step 5: Fill in the small holes inside the mask of lungs which we seperate right and left lung. r and l are symbolic and they can be actually left and right!
    # image labels: 5, 6
    r = label_image == labels[0]
    l = label_image == labels[1]
    r_edges = roberts(r)
    l_edges = roberts(l)
    r = ndi.binary_fill_holes(r_edges)
    l = ndi.binary_fill_holes(l_edges)
    if plot:
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(r, cmap=plt.cm.bone)
        plt_number += 1

        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(l, cmap=plt.cm.bone)
        plt_number += 1

    # Step 6: convex hull of each lung
    # image labels: 7, 8
    r = convex_hull_image(r)
    l = convex_hull_image(l)
    if plot:
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(r, cmap=plt.cm.bone)
        plt_number += 1

        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(l, cmap=plt.cm.bone)
        plt_number += 1
    # Step 7: joint two separated right and left lungs.
    # image label: 9
    sum_of_lr = r + l
    binary = sum_of_lr > 0
    if plot:
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(binary, cmap=plt.cm.bone)
        plt_number += 1
    # Step 8: Closure operation with a disk of radius 10. This operation is
    # to keep nodules attached to the lung wall.
    # image label: 10
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot:
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(binary, cmap=plt.cm.bone)
        plt_number += 1
    # Step 9: Superimpose the binary mask on the input image.
    # image label: 11
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot:
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(im, cmap=plt.cm.bone)
        plt_number += 1

    return im


def normalize(image):
    MIN_BOUND = -1200
    MAX_BOUND = 600.
    image2 = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image2[image2 > 1] = 1.
    image2[image2 < 0] = 0.
    image2 *= 255.
    return image2


def zero_center(image):
    PIXEL_MEAN = 0.25 * 256
    image2 = image - PIXEL_MEAN
    return image2


def make_circle_mask(img):
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # get the single external contours
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    if (np.sum(contours)<=0):
        mask = np.zeros(img.shape, dtype="uint8")
        return mask
    big_contour = max(contours, key=cv2.contourArea)
    
    rotrect = cv2.minAreaRect(big_contour)
    (center), (width,height), angle = rotrect
    
    mask = np.zeros(img.shape, dtype="uint8")
    mask = cv2.circle(mask, (int(center[0]), int(center[1])), int((width+height)/4), 255, -1)
    
    return mask

def getRangImageDepth(image):
    """
    :param image:
    :return:range of image depth
    """
    # start, end = np.where(image)[0][[0, -1]]
    fistflag = True
    startposition = 0
    endposition = 0
    for z in range(image.shape[0]):
        notzeroflag = np.max(image[z])
        if notzeroflag and fistflag:
            startposition = z
            fistflag = False
        if notzeroflag:
            endposition = z
    return startposition, endposition


def resize_image_itk(itkimage, newSpacing, resamplemethod=sitk.sitkNearestNeighbor):
    """
    image resize withe sitk resampleImageFilter
    :param itkimage:
    :param newSpacing:such as [1,1,1]
    :param resamplemethod:
    :return:
    """
    newSpacing = np.array(newSpacing, float)
    originSpcaing = itkimage.GetSpacing()
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    factor = newSpacing / originSpcaing
    newSize = originSize / factor
    newSize = newSize.astype(np.int)
    resampler.SetReferenceImage(itkimage)
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetSize(newSize.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    #if resamplemethod == sitk.sitkNearestNeighbor:
        #itkimgResampled = sitk.Threshold(itkimgResampled, 0, 1.0, 255)
    imgResampled = sitk.GetArrayFromImage(itkimgResampled)
    return imgResampled, itkimgResampled


def load_itk(filename):
    """
    load mhd files and normalization 0-255
    :param filename:
    :return:
    """
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(255)
    rescalFilt.SetOutputMinimum(0)
    # Reads the image using SimpleITK
    itkimage = rescalFilt.Execute(sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32))
    return itkimage


def load_itkfilewithtrucation(filename, upper=200, lower=-200):
    """
    load mhd files,set truncted value range and normalization 0-255
    :param filename:
    :param upper:
    :param lower:
    :return:
    """
    # 1,tructed outside of lungs value
    srcitkimage = sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32)
    srcitkimagearray = sitk.GetArrayFromImage(srcitkimage)
    srcitkimagearray[srcitkimagearray > upper] = upper
    srcitkimagearray[srcitkimagearray < lower] = lower
    # 2,get tructed outside of lungs value image
    sitktructedimage = sitk.GetImageFromArray(srcitkimagearray)
    origin = np.array(srcitkimage.GetOrigin())
    spacing = np.array(srcitkimage.GetSpacing())
    sitktructedimage.SetSpacing(spacing)
    sitktructedimage.SetOrigin(origin)
    # 3 normalization value to 0-255
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(255)
    rescalFilt.SetOutputMinimum(0)
    itkimage = rescalFilt.Execute(sitk.Cast(sitktructedimage, sitk.sitkFloat32))
    return itkimage

# Helper function to get rows in data frame associated with each file
def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return (f)

def processOriginaltraindata():
    expandslice = 13
    trainImage = "/DATA/jakaria_data/resources/processed_n/images/"
    trainMask = "/DATA/jakaria_data/resources/processed_n/masks/"
    trainMask_c = "/DATA/jakaria_data/resources/2d_data/masks_c/"
    """
    load itk image,change z Spacing value to 1,and save image ,liver mask ,tumor mask
    :return:None
    """
    seriesindex = 0
    for subsetindex in range(10): #range(10)
        luna_path = "/DATA/jakaria_data/resources/"
        luna_subset_path = luna_path + "subset" + str(subsetindex) + "/"
        output_path = "/DATA/jakaria_data/resources/masks/"
        luna_subset_mask_path = output_path + "subset" + str(subsetindex) + "/"
        file_list = glob(luna_subset_path + "*.mhd")
        
        file_list_path=[]
        for i in range(len(file_list)):
            file_list_path.append(file_list[i][0:-4])
        
        luna_csv_path = "/DATA/jakaria_data/resources/"
        df_node = pd.read_csv(luna_csv_path + "annotations.csv")
        df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list_path, file_name))
        df_node = df_node.dropna()
        
        for fcount in range(len(file_list)): #len(file_list)
            
            mini_df = df_node[df_node["file"] == file_list[fcount][0:-4]]
            print(mini_df.shape[0])
            print("debug: "+str(subsetindex)+" "+str(fcount))
            if mini_df.shape[0] == 0:
                              continue
            # 1 load itk image and truncate value with upper and lower
            #src = load_itkfilewithtrucation(file_list[fcount], 600, -1000)
            src = sitk.ReadImage(file_list[fcount])
            sub_img_file = file_list[fcount][len(luna_subset_path):-4]
            seg = sitk.ReadImage(luna_subset_mask_path + sub_img_file + "_segmentation.mhd", sitk.sitkUInt8)
            segzspace = seg.GetSpacing()[-1]
            # 2 change z spacing >1.0 to 1.0
            if segzspace > 1.0:
                _, seg = resize_image_itk(seg, (seg.GetSpacing()[0], seg.GetSpacing()[1], 1.0),
                                          resamplemethod=sitk.sitkNearestNeighbor)
                _, src = resize_image_itk(src, (src.GetSpacing()[0], src.GetSpacing()[1], 1.0),
                                          resamplemethod=sitk.sitkLinear)
            # 3 get resample array(image and segmask)
            segimg = sitk.GetArrayFromImage(seg)
            srcimg = sitk.GetArrayFromImage(src)

            trainimagefile = trainImage + str(seriesindex)
            trainMaskfile = trainMask + str(seriesindex)
            trainMaskfile_c = trainMask_c + str(seriesindex)
            if not os.path.exists(trainimagefile):
                os.makedirs(trainimagefile)
            #if not os.path.exists(trainMaskfile):
                #os.makedirs(trainMaskfile)
            #if not os.path.exists(trainMaskfile_c):
                #os.makedirs(trainMaskfile_c)    
            # 4 get mask
            seg_liverimage = segimg.copy()
            seg_liverimage[segimg > 0] = 255
            # 5 get the roi range of mask,and expand number slices before and after,and get expand range roi image
            startpostion, endpostion = getRangImageDepth(seg_liverimage)
            if startpostion == endpostion:
                continue
            imagez = np.shape(seg_liverimage)[0]
            startpostion = startpostion - expandslice
            endpostion = endpostion + expandslice
            if startpostion < 0:
                startpostion = 0
            if endpostion > imagez:
                endpostion = imagez
            srcimg = srcimg[startpostion:endpostion, :, :]
            seg_liverimage = seg_liverimage[startpostion:endpostion, :, :]
            # 6 write src, liver mask and tumor mask image
            for z in range(seg_liverimage.shape[0]): # range(seg_liverimage.shape[0])
                srcimg[z] = get_segmented_lungs(srcimg[z])
                srcimg[z] = normalize(srcimg[z])
                srcimg[z] = zero_center(srcimg[z])
                srcimg[z] = np.clip(srcimg[z], 0, 255).astype('uint8')
                #seg_liverimage[z] = zero_center(seg_liverimage[z])
                #seg_liverimage_c = make_circle_mask(seg_liverimage[z])
                cv2.imwrite(trainimagefile + "/" + str(z) + ".bmp", srcimg[z])
                #cv2.imwrite(trainMaskfile + "/" + str(z) + ".bmp", seg_liverimage[z])
                #seg_liverimage[z] = make_circle_mask(seg_liverimage[z])
                #cv2.imwrite(trainMaskfile_c + "/" + str(z) + ".png", seg_liverimage[z])
            seriesindex += 1
processOriginaltraindata()