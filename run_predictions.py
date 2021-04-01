import os
import numpy as np
import json
from PIL import Image

def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    
    '''
    BEGIN YOUR CODE
    '''

    # Use an example traffic light from the first image
    im = Image.open('./data/RedLights2011_Small/RL-001.jpg')
    ex_light = im.crop((316, 154, 323, 171))
    ex_light = np.asarray(ex_light) - 127.5

    # Find the dimensions of the traffic light
    (lt_rows, lt_cols, lt_channels) = np.shape(ex_light)
    box_height = lt_rows
    box_width = lt_cols

    # Find the dimensions of the image I and set threshold
    (n_rows, n_cols, n_channels) = np.shape(I)
    threshold = 0.9

    lt_vecs = []

    # For each channel, convert traffic light into normalized vector
    for i in range(3):
        lt_ch = ex_light[:, :, i]
        lt_vec = lt_ch.flatten()
        lt_norm = np.linalg.norm(lt_vec)
        if lt_norm != 0:
            lt_vec = lt_vec / lt_norm

        lt_vecs.append(lt_vec)

    # Go through all patches of this size 
    for i in range(round(n_rows / 2)):
        # Only check the bottom half of the iamge
        for j in range(n_cols - box_width):
            tl_row = i
            tl_col = j
            br_row = tl_row + box_height
            br_col = tl_col + box_width
            ch_inner_prod = []      

            # Go through each channel
            for ch in range(3):
                # Get one channel of the image and the same channel of the light
                img_ch = I[:, :, ch]
                lt_vec = lt_vecs[ch]

                # Convert this patch to a normalized vector
                patch = img_ch[tl_row:br_row, tl_col:br_col]
                patch_vec = patch.flatten() - 127.5
                patch_norm = np.linalg.norm(patch_vec)
                if patch_norm != 0:
                    patch_vec = patch_vec / patch_norm

                # Take the inner product of the traffic light with a patch.
                ch_inner_prod.append(np.dot(lt_vec, patch_vec))

            # If it's above the threshold add the box to the bounding boxes.
            for k in range(3):
                prod = ch_inner_prod[k]
                if prod < threshold:
                    break
                elif k == 2:
                    bounding_boxes.append([tl_row, tl_col, br_row, br_col]) 

    '''
    END YOUR CODE
    '''
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes

# set the path to the downloaded data: 
data_path = './data/RedLights2011_Small'

# set a path for saving predictions: 
preds_path = './data/hw01_preds' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

preds = {}
for i in range(len(file_names)):
    
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(I)
    
    preds[file_names[i]] = detect_red_light(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
