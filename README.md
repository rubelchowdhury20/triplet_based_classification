# README #


### Command to start the training ###

python3 main.py --data_directory ./dataset/refined_dataset --freeze_epochs 10 --unfreeze_epochs 400 --weights_directory ./weights/refined_weights_256

### To resume the training from a certain checkpoint ###

python3 main.py --data_directory ./dataset/refined_dataset --freeze_epochs 10 --unfreeze_epochs 400 --weights_directory ./weights/refined_weights_256 --resume True --checkpoint_name ./weights/refined_weights_256/model_best.pth

### Command to check the evaluation of model ###
python3 evaluate.py --source_data ./dataset/refined_dataset/train/ --test_data ./dataset/test_crops/ --weight_path ./weights/refined_weights/model_best.pth

### Command to generate annoy index and related json file ###
python3 generate_annoy_index.py --source_data ./dataset/refined_dataset/train/ --weight_path ./weights/refined_weights/model_best.pth