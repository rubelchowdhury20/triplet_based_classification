# Standard library imports
import os
import json
import math
import random

# Third party imports
import glob
import json
import pickle
from PIL import Image, ImageDraw
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from torchsummary import summary

# Local imports
import config
from modules import model
from modules import train
from modules import util

DEVICE = config.DEVICE

CURRENT_FREEZE_EPOCH = 0
CURRENT_UNFREEZE_EPOCH = 0
BEST_LOSS = 4													


def training(args):
	# declaring global variables
	global BEST_LOSS
	global CURRENT_FREEZE_EPOCH
	global CURRENT_UNFREEZE_EPOCH

	# steps for preparing and splitting the data for training
	root_path = config.ARGS.data_directory
	train_dataset = torchvision.datasets.ImageFolder(root = os.path.join(root_path, "train"), transform=config.data_transforms["train"])
	train_loader = data.DataLoader(train_dataset, **config.PARAMS)
	val_dataset = torchvision.datasets.ImageFolder(root = os.path.join(root_path, "val"), transform=config.data_transforms["val"])
	val_loader = data.DataLoader(val_dataset, **config.PARAMS)


	# loading the pretrained model and changing the dense layer. Initially the convolution layers will be freezed
	base_model = models.resnet18(pretrained=True).to(DEVICE)
	for param in base_model.parameters():
		param.requires_grad = False
	num_ftrs = base_model.fc.in_features
	# base_model.fc = nn.Sequential(nn.Linear(num_ftrs, 256))
	# base_model.fc = nn.Sequential(nn.Linear(num_ftrs, 256), nn.Linear(256, 128), nn.Linear(128, 64))
	base_model.fc = nn.Sequential(nn.Linear(num_ftrs, 256), nn.Linear(256, 128))
	base_model = base_model.to(DEVICE)
	# tnet = model.Tripletnet(base_model).to(DEVICE)

	if(config.ARGS.pretrained_weights):
		try:
			_, _, BEST_LOSS, base_model = util.load_checkpoint(config.ARGS.pretrained_weights, base_model)
		except:
			print("Not able to load from the pretrained_weights")
	elif(config.ARGS.resume):
		try:
			CURRENT_FREEZE_EPOCH, CURRENT_UNFREEZE_EPOCH, BEST_LOSS, base_model = util.load_checkpoint(config.ARGS.checkpoint_name, base_model)
		except:
			print("not able to load checkpoint because of non-availability")

	# Initializing the loss function and optimizer
	criterion = torch.nn.MarginRankingLoss(margin=config.TRIPLET_MARGIN)
	optimizer = optim.SGD(base_model.parameters(), lr=config.LR, momentum=config.MOMENTUM)

	# # Decay LR by a factor of 0.1 every 7 epochs
	# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

	# Printing total number of parameters
	n_parameters = sum([p.data.nelement() for p in base_model.parameters()])
	print('  + Number of params: {}'.format(n_parameters))

	# training for the first few iteration of freeze layers where apart from the dense layers all the other layres are frozen
	if(CURRENT_FREEZE_EPOCH < config.FREEZE_EPOCHS):
		for epoch in range(CURRENT_FREEZE_EPOCH + 1, config.FREEZE_EPOCHS + 1):
			train_class = train.Train(train_loader, val_loader, base_model, criterion, optimizer, epoch)

			train_class.train(batch_hard=True)
			loss = train_class.validate(batch_hard=True)

			# remember best loss and save checkpoint
			is_best = loss < BEST_LOSS
			BEST_LOSS = min(loss, BEST_LOSS)
			util.save_checkpoint({
				'current_freeze_epoch': epoch,
				'current_unfreeze_epoch': 0,
				'state_dict': base_model.state_dict(),
				'best_loss': BEST_LOSS,
			}, is_best, epoch)
			CURRENT_FREEZE_EPOCH = epoch


	# Unfreezing the last few convolution layers
	for param in base_model.parameters():
		param.requires_grad = True
	ct = 0
	for name, child in base_model.named_children():
		ct += 1
		if ct < 7:
			for name2, parameters in child.named_parameters():
				parameters.requires_grad = False

	# training the remaining iterations with the last few layers unfrozen
	if(CURRENT_UNFREEZE_EPOCH < config.UNFREEZE_EPOCHS):
		for epoch in range(CURRENT_UNFREEZE_EPOCH + 1, config.UNFREEZE_EPOCHS + 1):
			train_class = train.Train(train_loader, val_loader, base_model, criterion, optimizer, epoch)


			train_class.train(batch_hard=True)
			loss = train_class.validate(batch_hard=True)

			# remember best loss and save checkpoint
			is_best = loss < BEST_LOSS
			BEST_ACC = min(loss, BEST_LOSS)
			util.save_checkpoint({
				'current_freeze_epoch': CURRENT_FREEZE_EPOCH,
				'current_unfreeze_epoch': epoch,
				'state_dict': base_model.state_dict(),
				'best_loss': BEST_LOSS,
			}, is_best, CURRENT_FREEZE_EPOCH+epoch)
			CURRENT_UNFREEZE_EPOCH = epoch

