# Standard library imports
import os
import glob
import argparse

# Third party library imports
import torch
import torch.nn as nn
from torchvision import models

# Local imports
import config
import training

def main(args):
	# updating all the global variables based on the input arguments
	if(args.freeze_epochs):
		config.FREEZE_EPOCHS = args.freeze_epochs
	if(args.unfreeze_epochs):
		config.UNFREEZE_EPOCHS = args.unfreeze_epochs

	# updating batch size
	if(args.batch_size):
		config.PARAMS["batch_size"] = args.batch_size

	# updating command line arguments to the ARGS variable
	config.ARGS = args

	# calling required functions based on the input arguments
	training.training(args)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	# arguments for training
	parser.add_argument(
		"--data_directory",
		type=str,
		default="dataset",
		help="Path to the dataset Directory")
	parser.add_argument(
		"--batch_size",
		type=int,
		default=128,
		help="the batch_size for training as well as for inference")
	parser.add_argument(
		"--freeze_epochs",
		type=int,
		default=10,
		help="the total number of epochs for which the initial few layers will be frozen")
	parser.add_argument(
		"--unfreeze_epochs",
		type=int,
		default=400,
		help="the total number of epochs for which the full network will be unfrozen")
	parser.add_argument(
		"--resume",
		type=bool,
		default=True,
		help="Flag to resume the training from where it was stopped")
	parser.add_argument(
		"--checkpoint_name",
		type=str,
		default="model_best.pth",
		help="Name of the checkpoint from where to resume")
	parser.add_argument(
		"--pretrained_weights",
		type=str,
		default=None,
		help="Use pretrained weight to start the training")
	parser.add_argument(
		"--weights_directory",
		type=str,
		default="weights",
		help="Directory to save the weights")

	main(parser.parse_args())