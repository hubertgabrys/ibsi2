import SimpleITK as sitk
import numpy as np
import pandas as pd

from feature_extraction import extract_phase2_features
from filters import Mean, LoG, Laws2, Gabor, Wavelets, Wavelets2D
from utils import Image, Mask, ROI

# read the submission file
df_submission = pd.read_csv('phase2/IBSI-2-Phase2-Submission-Template.csv', sep=';')


# input data
img_path = "data_sets/ibsi_2_ct_radiomics_phantom/nifti/image/phantom.nii.gz"
mask_path = "data_sets/ibsi_2_ct_radiomics_phantom/nifti/mask/mask.nii.gz"

# extract roi config A
original_image = Image(modality='CT')
original_image.read_nifti(img_path)
original_mask = Mask()
original_mask.read_nifti(mask_path)
roi_a = ROI(original_image, original_mask, -1000, 400)


# extract roi config B
new_spacing = [1, 1, 1]
interpolated_image = Image(modality='CT')
interpolated_image.read_nifti(img_path)
interpolated_image.interpolate(new_spacing=new_spacing, method=sitk.sitkBSpline)
interpolated_image.round_intensities()
interpolated_mask = Mask()
interpolated_mask.read_nifti(mask_path)
interpolated_mask.interpolate(new_spacing=new_spacing, method=sitk.sitkLinear)
interpolated_mask.round_intensities()
roi_b = ROI(interpolated_image, interpolated_mask, -1000, 400)


def get_results(img_filter, roi):
    filtered_data = img_filter.filter(roi.image)
    filtered_roi = filtered_data.copy()
    ind = np.where(np.isnan(roi.resegmented_mask))
    filtered_roi[ind] = np.nan
    # feature extraction
    features = extract_phase2_features(roi_a.morphological_mask, roi.resegmented_mask, filtered_roi)
    for key, value in features.items():
        try:
            print(f"{key}: {value:.3f}")
        except TypeError:
            print(f"{key}: {value}")
    return features.values()


def get_results_law(filtered_data, roi):
    filtered_roi = filtered_data.copy()
    ind = np.where(np.isnan(roi.resegmented_mask))
    filtered_roi[ind] = np.nan
    # feature extraction
    features = extract_phase2_features(roi_a.morphological_mask, roi.resegmented_mask, filtered_roi)
    for key, value in features.items():
        try:
            print(f"{key}: {value:.3f}")
        except TypeError:
            print(f"{key}: {value}")
    return features.values()


# 1.A
print('\n1.A')
# feature extraction
features = extract_phase2_features(roi_a.morphological_mask, roi_a.resegmented_mask, roi_a.resegmented_mask)
for key, value in features.items():
    try:
        print(f"{key}: {value:.3f}")
    except TypeError:
        print(f"{key}: {value}")
df_submission["1.A"] = features.values()


# 1.B
print('\n1.B')
# feature extraction
features = extract_phase2_features(roi_a.morphological_mask, roi_b.resegmented_mask, roi_b.resegmented_mask)
for key, value in features.items():
    try:
        print(f"{key}: {value:.3f}")
    except TypeError:
        print(f"{key}: {value}")
df_submission["1.B"] = features.values()

# 2.A
print('\n2.A')
# filtering
mean_filter = Mean(padding_type="constant", support=5, dimensionality="2D")
df_submission["2.A"] = get_results(mean_filter, roi_a)

# 2.B
print('\n2.B')
# filtering
mean_filter = Mean(padding_type="constant", support=5, dimensionality="3D")
df_submission["2.B"] = get_results(mean_filter, roi_b)

# 3.A
print('\n3.A')
# filtering
log = LoG(padding_type="reflect", sigma_mm=1.5, cutoff=4.0, res_mm=0.977, dimensionality="2D")
df_submission["3.A"] = get_results(log, roi_a)

# 3.B
print('\n3.B')
# filtering
log = LoG(padding_type="reflect", sigma_mm=1.5, cutoff=4.0, res_mm=1.0, dimensionality="3D")
df_submission["3.B"] = get_results(log, roi_b)

# 4.A
print('\n4.A')
# filtering
laws2 = Laws2(response_map="L5E5", padding_type="reflect", dimensionality="2D", rotation_invariance=True, pooling="max", energy_map=True, distance=7)
df_submission["4.A"] = get_results(laws2, roi_a)

# 4.B
print('\n4.B')
# filtering
laws2 = Laws2(response_map="L5E5E5", padding_type="reflect", dimensionality="3D", rotation_invariance=True, pooling="max", energy_map=True, distance=7)
df_submission["4.B"] = get_results(laws2, roi_b)

# 5.A
print('\n5.A')
# filtering
gabor = Gabor(padding_type="reflect", res_mm=0.977, sigma_mm=5.0, lambda_mm=2.0, gamma=3/2, theta=np.pi/8, rotation_invariance=True, orthogonal_planes=False)
df_submission["5.A"] = get_results(gabor, roi_a)

# 5.B
print('\n5.B')
# filtering
gabor = Gabor(padding_type="reflect", res_mm=1.0, sigma_mm=5.0, lambda_mm=2.0, gamma=3/2, theta=np.pi/8, rotation_invariance=True, orthogonal_planes=True)
df_submission["5.B"] = get_results(gabor, roi_b)

# 6.A
print('\n6.A')
# filtering
wavelets = Wavelets2D(wavelet_type="db3", padding_type="reflect", response_map="LH", decomposition_level=1,
                    rotation_invariance=True)
df_submission["6.A"] = get_results(wavelets, roi_a)

# 6.B
print('\n6.B')
# filtering
wavelets = Wavelets(wavelet_type="db3", padding_type="reflect", response_map="LLH", decomposition_level=1,
                    rotation_invariance=True)
df_submission["6.B"] = get_results(wavelets, roi_b)

# 7.A
print('\n7.A')
# filtering
wavelets = Wavelets2D(wavelet_type="db3", padding_type="reflect", response_map="HH", decomposition_level=2,
                    rotation_invariance=True)
df_submission["7.A"] = get_results(wavelets, roi_a)

# 7.B
print('\n7.B')
# filtering
wavelets = Wavelets(wavelet_type="db3", padding_type="reflect", response_map="HHH", decomposition_level=2,
                    rotation_invariance=True)
df_submission["7.B"] = get_results(wavelets, roi_b)


df_submission.to_csv("phase2/results7.csv", index=False)
