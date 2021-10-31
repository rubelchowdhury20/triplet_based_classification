# Third party imports
import torch
from torchvision import transforms

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data preprocessing details
data_transforms = {
	'train': transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.RandomResizedCrop(200),
		transforms.RandomHorizontalFlip(0.5),
		transforms.RandomVerticalFlip(0.5),
		# transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
		# transforms.RandomPerspective(distortion_scale=0.3, p=0.1),
		transforms.RandomRotation(degrees=(0, 180)),
		# transforms.RandomPosterize(bits=2),
		# transforms.RandomEqualize(),
		# transforms.RandomAdjustSharpness(sharpness_factor=2),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	'val': transforms.Compose([
		transforms.Resize((224, 224)),
		# transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
}

# train related values
LR = 0.003													
MOMENTUM = 0.2

# FREEZE_EPOCHS = 5
# UNFREEZE_EPOCHS = 50

TRIPLET_MARGIN = 0.9								# the margin to be used for triplet loss
PARAMS = {'batch_size': 8,
			'shuffle': True,
			'num_workers': 16}

EMBEDDING_SIZE = 128

# command line arguments
ARGS = {}