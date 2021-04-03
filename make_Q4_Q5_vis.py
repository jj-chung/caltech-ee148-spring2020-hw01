import os
import numpy as np
import json
from PIL import Image, ImageDraw

def draw_boxes(img_to_box, img_path):

	# get sorted list of files: 
	file_names = sorted(os.listdir(img_path)) 

	# remove any non-JPEG files: 
	file_names = [f for f in file_names if '.jpg' in f] 

	for img in file_names:
		boxes = img_to_box[img]

		with Image.open(os.path.join(img_path, img)) as im:
			draw = ImageDraw.Draw(im)
			for box in boxes:
				new_box = [box[1], box[0], box[3], box[2]]
				draw.rectangle(new_box, outline='white', width=2)
			f_name = img_path + '_boxed/' + img.split('.')[0] + 'boxed.jpg'
			im.save(f_name, 'JPEG')

def create_visualization():
	'''
	Create visualizations of the bounding boxes on red light images
	and save them in the specified directory.
	save_loc specifies where to save visualizations.
	'''
	with open('./data/hw01_preds/preds.json') as f:
		data = f.read()

	img_to_box = json.loads(data)

	# For the best and worst examples, draw boxes and save the images.
	data_path_best = './data/RedLights2011_Medium_Best'
	data_path_worst = './data/RedLights2011_Medium_Worst'
	draw_boxes(img_to_box, data_path_best)
	draw_boxes(img_to_box, data_path_worst)


create_visualization()