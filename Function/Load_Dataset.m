% Function that converts the input dataset in an output which is used in
% main to mosaiced the images.
%%
function Dataset = Load_Dataset(Save, folder, nbr_of_img, mosaic)

% ---------------------------------------------------------------------
% All images are load as double.
% This code is adapted to read and extract data from
% DataQiuLaprayResizedVariable.mat and Monno dataset.
% ---------------------------------------------------------------------
% The return dataset contains images like :
% name  RGB_img_0  RGB_img_45  RGB_img_90  RGB_img_135
% ---------------------------------------------------------------------

path = [folder];
filename = dir(path);
Dataset = cell((nbr_of_img), 5);
if strcmp(mosaic,'cpfa')
    for k = 1:nbr_of_img
        img_0 = im2double(imread([path, '0_' num2str(k) '.png']));
        img_45 = im2double(imread([path, '45_' num2str(k) '.png']));
        img_90 = im2double(imread([path,'90_' num2str(k) '.png']));
        img_135 = im2double(imread([path, '135_' num2str(k) '.png']));

        Dataset(k, 1:5) = {num2str(k), img_0, img_45, img_90, img_135};
    end
elseif strcmp(mosaic,'pfa')
    for k = 1:nbr_of_img
        img_0 = im2double(rgb2gray(imread([path, '0_' num2str(k) '.png'])));
        img_45 = im2double(rgb2gray(imread([path, '45_' num2str(k) '.png'])));
        img_90 = im2double(rgb2gray(imread([path,'90_' num2str(k) '.png'])));
        img_135 = im2double(rgb2gray(imread([path, '135_' num2str(k) '.png'])));

        Dataset(k, 1:5) = {num2str(k), img_0, img_45, img_90, img_135};
    end
end

if Save == true
    save(['Data/FullDataset.mat'], 'Dataset', '-v7.3');
end

disp('Dataset loaded');
disp('---------------------------------------------------');
end
