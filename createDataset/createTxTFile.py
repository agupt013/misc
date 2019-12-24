import os
import random
import shutil
import sys
from itertools import chain
from PIL import Image

createImageNet = False
createLabelFile = True



test_data_path =""
training_data_path =""


#Put the path to the dataset you have
dataset = "/home/projects/places365/places365_standard/val"
txt_file = "/home/projects/places365/places365_standard/val_new.txt"
labelMap_txt_file = "/home/projects/places365/places365_standard/labels.txt"

dirCount = 0
createLabelFile = True
label = 0
sys.stdout.write("Creating training and validation dataset...\n")

saveFile = open(txt_file, "w")

for(dirpath,dirnames,filenames) in os.walk(dataset):
    dirCount += 1
    createLabelFile = False
    len_filenames = len(filenames)

    if len(filenames) != 0:  
        dirname = dirpath.split('/')[-1]
        for name in filenames:
            temp_filename = os.path.join(dirpath,name)
            #raw_input(temp_filename)

            saveFile.write(temp_filename + " "+ str(label)+"\n")
            
                                   
        if createLabelFile == True:
            with open(labelMap_txt_file, "a") as labelMap:
                labelMap.write(dirname +" "+ str(label) + "\n")
        label += 1
        
        sys.stdout.write("\033[K")
        sys.stdout.write("\tProcessed: {0:25} \t\tCompleted:{1}/{1}\n".format(dirname,len_filenames))
sys.stdout.write("\nDone")
saveFile.close()

        
                                    
            
        
