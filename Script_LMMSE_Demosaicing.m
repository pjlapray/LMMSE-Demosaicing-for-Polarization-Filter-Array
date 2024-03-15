%% This script load, and demosaic with LMMSE the data from the selected image
% References : 1-"Spote A, Lapray PJ, Thomas JB, Farup I. Joint demosaicing of
%              colour and polarisation from filter arrays. 
%              In 29th Color and Imaging Conference Final Program and Proceedings 2021. 
%              Society for Imaging Science and Technology."
%              2- "Dumoulin R, Lapray P.-J., Thomas J.-B., (2022), Impact of training data on
%              LMMSE demosaicing for Colour-Polarization Filter Array,  16th International Conference on Signal-Image Technology & Internet-Based Systems (SITIS),
%              2022, Dijon, France."

clc
close all
clear all

%% Global parameter
D_Matrix_name = 'D_Matrix.mat'; % If retrained, call D_Matrix_retrained
Save = true; % true to save the demosaiced image

%% Add path to Matlab for access
addpath(genpath('Function/'))

%% Load mosaiced image
MosImg=im2double(imread(['Data/im.tif'])); % (The scene has a polarizer sheet in background)
figure;imshow(MosImg);title('Mosaiced image');

%% Sizes definition
height = 2;                                 % height of the superpixel
width = 2;                                 % width of the superpixel
P = 4;                                % number of color-pola channels
[rows, cols, ~] = size(MosImg);
r_superpix = rows/height;                % number of superpixel in a line
c_superpix = cols/width;                 % number of superpixel in a column

%% Load D_Matrix
D = load(['Data/' D_Matrix_name]).D;

%% Demosaicing
[DemosImg] = Function_LMMSE_Demosaicing(MosImg,D);

%% Show result images
figure;
subplot(2,2,1),imshow(DemosImg(:,:,:,1)),title('Demosaiced image for 0° polarization')
subplot(2,2,2),imshow(DemosImg(:,:,:,2)),title('Demosaiced image for 45° polarization')
subplot(2,2,3),imshow(DemosImg(:,:,:,3)),title('Demosaiced image for 90° polarization')
subplot(2,2,4),imshow(DemosImg(:,:,:,4)),title('Demosaiced image for 135° polarization')

%% Save result
if Save == true
    save(['Data/im_demosaiced.mat'],'DemosImg','-v7.3');
    if exist('Data/im_demosaiced.tif')
        delete Data/im_demosaiced.tif;
    end
    for i=1:P
        imwrite(DemosImg(:,:,:,i),'Data/im_demosaiced.tif',"WriteMode","append"); % Multipage tif file (you can use irfanview for ex. to read it)
    end
end
