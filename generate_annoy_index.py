# Standard library imports
import os
import re
import glob
import json
import random
import argparse

# Third party library imports
import torch
from torch.utils import data
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw

from annoy import AnnoyIndex

# local imports
import config


class ImageDataset(data.Dataset):
	def __init__(self, images, transforms=None):
		self.images = images
		self.transforms = transforms

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		image = Image.open(self.images[index])
		image = np.asarray(image)
		image = image[:,:,:3]
		image = Image.fromarray(image)
		if self.transforms is not None:
			return self.images[index], self.transforms(image)
		else:
			return self.images[index], image


# generate annoy_index tree given image embeddings and embedding size
def create_annoy_index(image_embeddings, embedding_size):
	index_to_label = {}
	annoy_index = AnnoyIndex(embedding_size, metric="euclidean")
	for index, embedding in tqdm(enumerate(image_embeddings)):
		index_to_label[index] = embedding["image"].split("/")[-2]
		annoy_index.add_item(index, embedding["embedding"])
	annoy_index.build(10000)
	return annoy_index, index_to_label


def get_embeddings(emb_dataloader, base_model):
	embeddings = []				# list to store the embeddings in dict format as name, embedding
	base_model.eval()
	with torch.no_grad():				# no update of parameters
		for image_names, images in tqdm(emb_dataloader):
			images = images.to(config.DEVICE)
			image_embeddings = base_model(images)
			embeddings.extend([{"image": image_names[index], "embedding": embedding} for index, embedding in enumerate(image_embeddings.cpu().data)])
	return embeddings


def load_model(weight_path):
	# loading the trained model and generating embedding based on that
	base_model = models.resnet18(pretrained=False).to(config.DEVICE)
	for param in base_model.parameters():
		param.requires_grad = False
	num_ftrs = base_model.fc.in_features
	base_model.fc = nn.Sequential(nn.Linear(num_ftrs, 256), nn.Linear(256, 128))
	base_model = base_model.to(config.DEVICE)

	# loading the trained model with trained weights
	checkpoint = torch.load(weight_path)
	base_model.load_state_dict(checkpoint['state_dict'])
	base_model = base_model.eval()

	return base_model


def main(args):
	img_list = [os.path.join(root, name)
				for root, dirs, files in os.walk(args.source_data)
				for name in files]

	dataset = ImageDataset(img_list, config.data_transforms["val"])
	data_loader = data.DataLoader(dataset, **config.PARAMS)

	base_model = load_model(args.weight_path)

	print("Generating embeddings for source images...")
	image_embeddings = get_embeddings(data_loader, base_model)

	annoy_index, annoy_index_to_label = create_annoy_index(image_embeddings, 128)


	annoy_index.save("annoy_index.ann")
	with open('annoy_index_to_label.json', 'w') as f:
		json.dump(annoy_index_to_label, f)




if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	# arguments to evaluate the model on test data
	parser.add_argument(
		"--source_data",
		type=str,
		default="./dataset/train/",
		help="Path to the train dataset, using which the training was done. Inside the dir each class images are kept in different folders.")

	parser.add_argument(
		"--weight_path",
		type=str,
		default="./weights/model_best.pth",
		help="Path to the weight file")

	main(parser.parse_args())

