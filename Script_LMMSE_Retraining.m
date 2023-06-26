%% This script retrain the LMMSE algorithm from a dataset of images
% A small dataset of 3 images is provided as a demonstration. 
% Please use the same image structure as in Data/Dataset if you want to
% retrain with your own dataset.
% At the end of this Script, you will find the new trained D matrix in
% folder "Data".
% References : 1-"Spote A, Lapray PJ, Thomas JB, Farup I. Joint demosaicing of
%              colour and polarisation from filter arrays. 
%              In 29th Color and Imaging Conference Final Program and Proceedings 2021. 
%              Society for Imaging Science and Technology."
%              2- "Dumoulin R, Lapray P.-J., Thomas J.-B., (2022), Impact of training data on
%              LMMSE demosaicing for Colour-Polarization Filter Array,  16th International Conference on Signal-Image Technology & Internet-Based Systems (SITIS),
%              2022, Dijon, France.
clc
close all
clear all

%% Add path to Matlab for access
addpath(genpath('Function/'))

%% Select pfa or cpfa
mosaic = 'pfa';

%% Load full resolution dataset
path=pwd();
folder_path = strcat(path, '/Data/Dataset/');
nbr_of_img = 12;
FullDataset = Load_Dataset(true, folder_path, nbr_of_img,mosaic); % This small dataset is made of 3 images, feel free to add you own or use existing datasets

%% Mosaicking of the dataset
MosDataset = Mosaicking(FullDataset, true,folder_path,mosaic)

%% Computation of D matrix (demosaicing matrix)
[DemosDataset, y1, y, D] = D_matrix(FullDataset, MosDataset, folder_path,mosaic);

%% Save the new re-trained D matrix
save("Data\D_Matrix_retrained.mat","D");
display('Retrained D matrix saved');
