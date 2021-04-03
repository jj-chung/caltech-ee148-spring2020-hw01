import os
import numpy as np
import json
from PIL import Image, ImageDraw

import time
import sys
np.set_printoptions(threshold=sys.maxsize)


def rgb_to_hsv(r, g, b):
    r0 = r / 255.0
    g0 = g / 255.0
    b0 = b / 255.0

    c_max = max(r0, g0, b0)
    c_min = min(r0, g0, b0)
    delta = c_max - c_min
    
    if delta == 0:
        h = 0
    elif c_max == r0:
        h = (60 * ((g0 - b0) / delta + 6)) % 360
    elif c_max == g0:
        h = (60 * ((b0 - r0) / delta + 2)) % 360
    elif c_max == b0:
        h = (60 * ((r0 - g0) / delta + 4)) % 360

    if c_max == 0:
        s = 0
    else:
        s = (delta / c_max) * 100

    v = c_max * 100

    return h, s, v

def find_red_black(I):
    '''
    Takes a numpy array <I> and returns two lists <red_coords> and <black_coords>,
    and a 2D array of shape np.shape(I) with 
        black coordinates having entry 0,
        red coordinates having entry 1, and 
        all other colors -1. 
    <red_coords> contains all coordinates in I which are approx. red, and 
    <black_coords> contains all coordinates in I which are approx. black.
    '''
    # Find the dimensions of the image I and set threshold
    (n_rows, n_cols, n_channels) = np.shape(I)
    new_img = np.zeros((n_rows, n_cols)) 
    red_coords = []
    black_coords = []

    for row in range(n_rows):
        for col in range(n_cols):
            if row > n_rows / 2:
                new_img[row, col] = -1
            else:
                r, g, b = I[row, col, :]
                h, s, v = rgb_to_hsv(r, g, b)

                '''
                # Check if this pixel is red or black
                if r > g * 2.5 and r > b * 2.5:
                    red_coords.append([row, col])
                    new_img[row, col] = 1
                elif r < 100 and g < 100 and b < 100:
                    black_coords.append([row, col])
                    new_img[row, col] = 0
                else:
                    new_img[row, col] = -1
                '''
                # Check if this pixel is red or black
                if (h < 20 or h > 320) and v > 50 and s > 50:
                    red_coords.append([row, col])
                    new_img[row, col] = 1
                elif v < 35:
                    black_coords.append([row, col])
                    new_img[row, col] = 0
                else:
                    new_img[row, col] = 0.3

    return red_coords, black_coords, new_img

def to_normalized_vec(matrix):
    vec = matrix.flatten()
    vec_norm = np.linalg.norm(vec)
    if vec_norm != 0:
        return vec/ vec_norm
    else:
        return vec

def delete_duplicates(bounding_boxes, l_rows, l_cols):
    # Clean up bounding boxes by removing duplicates
    visited = set()
    new_boxes = []

    for box in bounding_boxes:
        tl_row, tl_col, br_row, br_col = box

        if (tl_row, tl_col) not in visited:
            new_boxes.append(box)

            for i in range(tl_row - l_rows, br_row + l_rows):
                for j in range(tl_col - l_cols, br_col + l_cols):
                    visited.add((i, j))

    return new_boxes

def detect_red_light_color(I, bounding_boxes):
    '''
    Called for functionality in detect_red_light, using a different algorithm (color-
    based) algorithm.
    '''

    r_coords, b_coords, img = find_red_black(I)
    n_rows, n_cols, n_channels = np.shape(I)
    
    '''
    # For visualization purposes only: draw where it sees red/black
    data_path = './data/RedLights2011_Small/'

    with Image.open(os.path.join(data_path, name)) as im:
        draw = ImageDraw.Draw(im)
        for r_coord in r_coords:
            draw.point([r_coord[1], r_coord[0]], fill='white')

        for b_coord in b_coords:
            draw.point([b_coord[1], b_coord[0]], fill='green')

        f_name = './data/boxed_images_color_example/' + name.split('.')[0] + '_red.jpg'
        im.save(f_name, 'JPEG')
    '''

    # For each red coordinate, we travel down until we reach black.
    # Taking this dist as the diameter, we draw a bounding box to check if it has 
    # high inner product with a traffic light of that size.
    for r_coord in r_coords:
        curr_row, curr_col = r_coord 

        curr_color = 1
        while curr_row - r_coord[0] < 80:
            # If we've reached black for the first time, change the current color.
            # Otherwise continue to make sure the area below the light is black.
            if img[curr_row][curr_col] == 0:
                # Save this row and diameter
                row = curr_row
                diam = row - r_coord[0]
                curr_color = 0
                break

            curr_row += 1

        if curr_color != 0:
            break

        # Find the radius of the circle, margin around the light, and center of
        # the light.
        radius = diam / 2.0
        margin = round(diam * 0.1)
        center_row = round(margin + radius)
        center_col = center_row

        # Determine the expected size of the light.
        l_cols = round(2 * margin + diam)
        l_rows = round(margin + diam + 1.5 * diam)
        light = np.zeros((l_rows, l_cols))

        # Determine points which we expect to be red (circular)
        for row in range(l_rows):
            for col in range(l_cols):
                dist = (row - center_row) ** 2 + (col - center_col) ** 2
                if dist < (radius) ** 2 and dist > (radius / 2.0) ** 2:
                    light[row, col] = 1

        # Draw bounding box
        s_row = max(0, r_coord[0] - margin) 
        e_row = min(n_rows, s_row + l_rows)
        s_col = max(0, r_coord[1] - round(radius) - margin) 
        e_col = min(n_cols, s_col + l_cols)
        
        # Normalize things 
        light_vec = to_normalized_vec(light)

        patch = img[s_row:e_row, s_col:e_col]
        patch_vec = to_normalized_vec(patch)

        # If the patch approximately matches what we expect for a traffic light,
        # then add the bounding box. 
        try:
            prod = np.dot(patch_vec, light_vec)

            temp_img = img
            temp_img[r_coord[0], r_coord[1]] = 100
            temp_light = temp_img[s_row:e_row, s_col:e_col + 2]
            

            '''
            if 'RL-010' in name:
                print(light)
                print(r_coord)
                print(temp_light)
                print(prod)
                time.sleep(5)
            '''
            
            if prod > 0.5:
                bounding_boxes.append([s_row, s_col, e_row, e_col])

        except ValueError:
            pass

    # Remove duplicate bounding boxes 
    if len(bounding_boxes) > 1:
        return delete_duplicates(bounding_boxes, l_rows, l_cols)
    else:
        return bounding_boxes


def detect_red_light_match(I, bounding_boxes):
    '''
    Called for functionality in detect_red_light, using match filtering algorithm. 
    '''
    
    # Use an example traffic light from the first image
    im = Image.open('./data/RedLights2011_Medium/RL-001.jpg')
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

    # Remove duplicate bounding boxes 
    if len(bounding_boxes) > 1:
        return delete_duplicates(bounding_boxes, lt_rows, lt_cols)
    else:
        return bounding_boxes

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
    
    bounding_boxes = detect_red_light_match(I, bounding_boxes, name)
    
    '''
    END YOUR CODE
    '''

    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes

# set the path to the downloaded data: 
data_path = './data/RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = './data/hw01_preds_match' 
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
