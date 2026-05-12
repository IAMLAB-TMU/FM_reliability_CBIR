import os, sys
import numpy as np
import openslide
import PIL
from PIL import Image
import cv2

from skimage import img_as_ubyte

os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras
import segmentation_models as sm


Seg_model = sm.Unet('mobilenet')
Seg_model.load_weights("D:/AB/Tissue_Segmentation/UNET_Weights/unet_mobilenet_50ep_Final_Dataset_Augment_Npp_Loss_snsp.h5")



def unet_tissue_segmentation(Slide):
    
    patch_read_size = 256
    in_patch_pad = (1024 - patch_read_size)//2
    
    Img = Slide.get_thumbnail((1024, 1024)) ## when you read thumbnail
    Img = np.asarray(Img.convert("RGB"))
    
    
    h = Img.shape[0]
    w = Img.shape[1]
    bottom_padding_size = (patch_read_size-(h%patch_read_size))%patch_read_size
    right_padding_size = (patch_read_size-(w%patch_read_size))%patch_read_size
    padded_img = cv2.copyMakeBorder(
        Img,
        top=0,
        bottom=bottom_padding_size,
        left=0,
        right=right_padding_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )
    
    # This part goes through the patches of the thumbnail and segments each patch. 
    mask_patch_list = []
    for j in range(0, padded_img.shape[0], patch_read_size):
        row_mask_patch_list = []

        for i in range(0, padded_img.shape[1], patch_read_size):
            # This part adds padding to each patch to make its size to 1024, which is the network input size.
            # Experiments show that the network yields better results if some padding is added and the parts 
            # at the patch borders are not needed to be segmented
            cur_patch = padded_img[max(0, j-in_patch_pad):min((j+patch_read_size+in_patch_pad), padded_img.shape[0]), 
                                    max(0, i-in_patch_pad):min((i+patch_read_size+in_patch_pad), padded_img.shape[1])]
                            
            top_border = (1024-cur_patch.shape[0])//2
            bottom_bordder = 1024 - (top_border + cur_patch.shape[0])
            left_border = (1024-cur_patch.shape[1])//2
            right_border = 1024 - (left_border + cur_patch.shape[1])
            
            cur_patch = cv2.copyMakeBorder(
                cur_patch,
                top=top_border,
                bottom=bottom_bordder,
                left=left_border,
                right=right_border,
                borderType=cv2.BORDER_CONSTANT,
                value=[255, 255, 255]
            )
                
            # Segmenting the patches using the network
            network_inp = np.array([cur_patch/255])
            cur_patch_mask = Seg_model.predict(network_inp)
                
                
            res_patch_st_j = top_border+min(j, in_patch_pad)
            res_patch_st_i = left_border+min(i, in_patch_pad)
            cur_patch_mask = cur_patch_mask[:, res_patch_st_j:res_patch_st_j + patch_read_size, 
                                                   res_patch_st_i:res_patch_st_i + patch_read_size, :]

                
            cur_patch_mask = cur_patch_mask.reshape((patch_read_size, patch_read_size, 1))

            row_mask_patch_list.append(cur_patch_mask) # adding the patch to a list to concat them later

        mask_patch_list.append(row_mask_patch_list)

    # This part concat the patches together, binarizes it and saves the segmented mask
    padded_thmb_mask = cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in mask_patch_list])
    padded_thmb_mask = (padded_thmb_mask*255).astype('uint8')
    ret2, padded_thmb_mask_binary = cv2.threshold(padded_thmb_mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    padded_thmb_mask_binary = padded_thmb_mask_binary[0:h, 0:w]

    return_val, seg_mask = cv2.threshold(padded_thmb_mask_binary, 0 , 255, cv2.THRESH_BINARY)
    
    return Img, seg_mask


### Otsu Segmentation is on 1024 thumbnail
def Otsu_Seg(Slide):
    
    Img = Slide.get_thumbnail((1024, 1024)) ## when you read thumbnail
    Img = np.asarray(Img.convert("RGB"))
    
    #H, E = HE_deconv(Img)
    
    gray_Img = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
    gray_Img = cv2.GaussianBlur(gray_Img, (5,5), 0)
    ret, seg = cv2.threshold(gray_Img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    seg = (255 - seg)
    
    return Img, seg