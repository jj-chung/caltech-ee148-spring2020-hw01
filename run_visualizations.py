import os
import numpy as np
import json
from PIL import Image, ImageDraw

def create_visualization():
	'''
	Create visualizations of the bounding boxes on red light images
	and save them in the specified directory.
	'''
	with open('./data/hw01_preds/preds.json') as f:
		data = f.read()

	img_to_box = json.loads(data)
	data_path = './data/RedLights2011_Small'

	for img, boxes in img_to_box.items():
		with Image.open(os.path.join(data_path, img)) as im:
			draw = ImageDraw.Draw(im)
			for box in boxes:
				new_box = [box[1], box[0], box[3], box[2]]
				draw.rectangle(new_box, outline='white', width=1)
				f_name = './data/boxed_images/' + img.split('.')[0] + 'boxed.jpg'
				im.save(f_name, 'JPEG')

create_visualization()