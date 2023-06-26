%% Main code for mosaicing images in datasets 
function MosDataset = Mosaicking(Dataset, Save,folder_path,mosaic)
% -------------------------------------------------------------------------
%     Loading data to have the same type in input of the Filter funtion
% -------------------------------------------------------------------------
disp('Mosaicking');
disp('---------------------------------------------------');

%%  Mosaicking

MosDataset = cell( size(Dataset,1), 2 );
for k = 1:size(Dataset,1)
    [r, c, l] = size(Dataset{k,2});
    image = cat(3, Dataset{k, 2}, Dataset{k, 3},  Dataset{k, 4}, Dataset{k, 5});
    image = reshape(image, r, c, 4*l);
    MosDataset{k,1} = Dataset{k, 1};
    MosDataset{k,2} = Filter(image,mosaic);
end

if Save == true
    save(['Data/MosDataset.mat'], 'MosDataset', '-v7.3');
end

disp('Mosaicking done');
end
