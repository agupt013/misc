import math, json, os, sys
sys.path.append("..")  # Adds higher directory to python modules path.
import numpy as np
from PIL import ImageFile
from joblib import Parallel, delayed
import h5py
import glob
import argparse
import time
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torchvision import models, transforms, datasets

def read_image(img_path):
    trans = transforms.ToTensor()
    return Image.open(img_path) #, target_size=(224, 224)))
    
    return img2vec.get_vec(img)

def checkDirExists(dir_p):
    if not os.path.isdir(dir_p):
        dir_p = os.path.dirname(dir_p)

    if not os.path.exists(dir_p):        
        os.makedirs(dir_p)

def hdf5_write(hf,data_1,data_2,data_3):
    dset_1 = 'features'
    dset_2 = 'img_path'
    dset_3 = 'labels'
    
    
    try:
        hf.create_dataset(dset_1, data = data_1, maxshape=(None,np.array(data_1).shape[1]),dtype= 'f')
        hf.create_dataset(dset_2, data = np.string_(data_2), maxshape=(None,))
        hf.create_dataset(dset_3, data = data_3, maxshape=(None,))
	
    except:
        l_data = np.array(data_1).shape[0]	
        hf[dset_1].resize((hf[dset_1].shape[0] + l_data), axis = 0)	    
        hf[dset_1][-l_data:] = data_1	    
        hf[dset_2].resize((hf[dset_2].shape[0] + l_data), axis = 0)
        hf[dset_2][-l_data:] = np.string_(data_2)
        hf[dset_3].resize((hf[dset_3].shape[0] + l_data), axis = 0)
        hf[dset_3][-l_data:] = data_3
   
import torch
from torchvision import datasets

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        #tuple_with_path = (original_tuple + (path,))
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Feature Extractor')
    parser.add_argument("-d", "--data_dir",\
                    help="path to data directory. It must contain a train and val subdirectories.")
    parser.add_argument("-hf", "--hdf5_path",\
                    help="path to store hdf5 file.")
    parser.add_argument("-m", "--model_path",\
                    help="path to model to be used.Default ResNet50.")
    parser.add_argument("-g", "--gpus",\
                    help="gpu to use.")
    parser.add_argument("-b", "--batch_size",\
                    help="batch size to feed to network")    
    
    
    args = parser.parse_args()

    if len(sys.argv) < 2*(len(vars(args))-3) +1 :
        sys.exit('[Error] Source Code 0: Usage: python {0} -d <path to data directory> -h <output path to store hdf5 file>'.format(sys.argv[0]))

    if args.model_path == None:
        model = models.resnet50(pretrained = True)
    else:
        model = torch.load(args.model_path)

    if args.gpus != None:
        device = torch.device("cuda:{}".format(args.gpus) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    if args.batch_size == None:
        batch_size = 64
    else:
        batch_size = args.batch_size
	
    model = torch.nn.Sequential(*list(model.children())[:-1])

    model.to(device)
    model.eval()
    #input(model)
    data_dir = args.data_dir
    hdf5_path = args.hdf5_path
    #TRAIN_DIR = os.path.join(data_dir, 'train')
    #VALID_DIR = os.path.join(data_dir, 'val')

    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    #image_datasets = ImageFolderWithPaths(data_dir,data_transforms)
    image_datasets = ImageFolderWithPaths(TRAIN_DIR,data_transforms)   

    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size,
                                             shuffle=False, num_workers=4)
      
    dataset_sizes = len(image_datasets)
    total_batches = len(dataloaders)
    
    checkDirExists(hdf5_path)
    

    hf = h5py.File(hdf5_path,'a')
    #print(dataloaders.shape)
    b_idx = 0
    for inputs, labels, paths in dataloaders:
        inputs = inputs.to(device)
        labels = labels.detach().numpy()
        features = model(inputs)
        #input(features.detach().cpu().numpy().shape)
        hdf5_write(hf,features.detach().cpu().numpy().reshape(-1,2048),paths,labels)
        del features
        sys.stdout.write('[INFO] Processed {0}/{1} batches.\r'.format(b_idx+1,total_batches))
        b_idx += 1
    sys.stdout.write('\n')
    hf.close()
    


