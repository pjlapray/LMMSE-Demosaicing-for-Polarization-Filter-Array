# Demosaicing-PFA-LMMSE
This code allow you to demosaic a polarimetric image from a filter array sensor with the Linear Minimum Mean Square Error demosaicing algorithm. The arrangement is the Chun polarization pattern from the SONY IMX250 MZR sensor. The code will be readapted for any square arrangement in a future release.

The LMMSE demosaicing algorithm is a learning-based technique. The initial training (matrix 'D_matrix.mat' in Data folder) is done with the data used in the publication (training with 12 images from the Wen et al. database). If using the code, please cite these publications:

  <b>1- Dumoulin R., Lapray P.-J., Thomas J.-B., Farup I., Impact of training data on LMMSE demosaicing for Colour-Polarization Filter Array, submitted to SITIS conference, 2022. (update when published).
  
  2- Spote A., Lapray P.-J., Thomas J.-B., Farup I., Joint demosaicing of colour and polarisation from filter arrays, In 29th Color and Imaging Conference Final Program and Proceedings 2021, Society for Imaging Science and Technology, 2021.</b>
  
<img src="https://github.com/pjlapray/LMMSE-Demosaicing-for-Polarization-Filter-Array/assets/10449075/37502265-ee02-4e7b-940d-6aadaa094b2e" width=35%>

Two scripts are provided as demonstration:
   - The script to use for demosaicing with the furbished pre-trained matrix D is "Script_LMMSE_Demosaicing".
   - If you may want to retrain the demosaicing matrix with your own specific data, use the Script named "Script_LMMSE_training.m". Please use the same image structure as in "Data/Dataset". It is recommanded to train with sufficient data (at least 12 images of resolution 1456 × 1088 pixels, see our conference paper from 2022 for more information).

The "Data" folder contains a mosaiced image, a matrix used for demosaicing, along woth a dataset in case of retraining matrix D with the script "Script_LMMSE_training.m".

The "Function" folder contains Matlab functions needed for retraining.
