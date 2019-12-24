# Repository for helper codes
This repo contains some helper codes that I used frequently and could be useful to other. 

## Code descriptions

There are codes for the following :

1. server : Contains bash scripts that can be helpful for checking active nodes, for sending email from server.
2. createDataset : Contains one script to generate a  text file of the files and their respective  labels. It can be used to write a custom datalaoder when dataset is huge.
3. featExtractor.py :  A script to extract features from a CNN and store it in a HDF5 file. This script is faster as joblib is used for parallel processing.
