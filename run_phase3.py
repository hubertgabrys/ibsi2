import SimpleITK as sitk
import pandas as pd

from feature_extraction import calc_stat_mean, calc_stat_var, calc_stat_skew, calc_stat_kurt, calc_stat_median, \
    calc_stat_min, calc_stat_p10, calc_stat_p90, calc_stat_max, calc_stat_iqr, calc_stat_range, calc_stat_mad, \
    calc_stat_rmad, calc_stat_medad, calc_stat_cov, calc_stat_qcod, calc_stat_energy, calc_stat_rms
from filters import Mean, LoG, Laws2, Gabor, Wavelets
from utils import Image, Mask, ROI
import numpy as np
from tqdm import tqdm


def preprocess_img(img_path, mask_path, modality):
    # image processing
    if modality == 'PET':
        new_spacing = [3, 3, 3]
    else:
        new_spacing = [1, 1, 1]
    interpolated_image = Image(modality=modality)
    interpolated_image.read_nifti(img_path)
    interpolated_image.interpolate(new_spacing=new_spacing, method=sitk.sitkBSpline)
    if modality == 'CT':
        interpolated_image.round_intensities()
    interpolated_mask = Mask()
    interpolated_mask.read_nifti(mask_path)
    interpolated_mask.interpolate(new_spacing=new_spacing, method=sitk.sitkLinear)
    interpolated_mask.round_intensities()
    if modality == 'CT':
        roi = ROI(interpolated_image, interpolated_mask, -200, 200)
    else:
        roi = ROI(interpolated_image, interpolated_mask, 0, np.inf)
    return roi


def extract_features(img_filter, roi):
    if img_filter is not None:
        # print(f'1_{img_filter}')
        filtered_data = img_filter.filter(roi.image)
    else:
        # print(f'2_{img_filter}')
        filtered_data = roi.image.copy()
    filtered_roi = filtered_data.copy()
    ind = np.where(np.isnan(roi.resegmented_mask))
    filtered_roi[ind] = np.nan

    features_dict = {}
    features_dict['stat_mean'] = calc_stat_mean(filtered_roi)
    features_dict['stat_var'] = calc_stat_var(filtered_roi)
    features_dict['stat_skew'] = calc_stat_skew(filtered_roi)
    features_dict['stat_kurt'] = calc_stat_kurt(filtered_roi)
    features_dict['stat_median'] = calc_stat_median(filtered_roi)
    features_dict['stat_min'] = calc_stat_min(filtered_roi)
    features_dict['stat_p10'] = calc_stat_p10(filtered_roi)
    features_dict['stat_p90'] = calc_stat_p90(filtered_roi)
    features_dict['stat_max'] = calc_stat_max(filtered_roi)
    features_dict['stat_iqr'] = calc_stat_iqr(filtered_roi)
    features_dict['stat_range'] = calc_stat_range(filtered_roi)
    features_dict['stat_mad'] = calc_stat_mad(filtered_roi)
    features_dict['stat_rmad'] = calc_stat_rmad(filtered_roi)
    features_dict['stat_medad'] = calc_stat_medad(filtered_roi)
    features_dict['stat_cov'] = calc_stat_cov(filtered_roi)
    features_dict['stat_qcod'] = calc_stat_qcod(filtered_roi)
    features_dict['stat_energy'] = calc_stat_energy(filtered_roi)
    features_dict['stat_rms'] = calc_stat_rms(filtered_roi)

    return features_dict.values()


patient_names = ['STS_' + str(e).zfill(3) for e in range(1, 52)]

for patient_name in tqdm(patient_names, desc='Patients'):
    for modality in ['CT', 'MR_T1', 'PET']:
        if modality == 'PET':
            res_mm = 3.0
        else:
            res_mm = 1.0
        print(f"Patient {patient_name}, modality {modality}, resolution {res_mm}.")
        img_path = f"data_sets/ibsi_validation/nifti/{patient_name}/{modality}_image.nii.gz"
        mask_path = f"data_sets/ibsi_validation/nifti/{patient_name}/{modality}_mask.nii.gz"
        # preprocess img
        roi = preprocess_img(img_path, mask_path, modality)
        # read the submission file
        df_submission = pd.read_csv('phase3/template.csv', sep=';')
        img_filter = None
        df_submission["ID_1"] = extract_features(img_filter, roi)
        img_filter = Mean(padding_type="constant", support=3, dimensionality="3D")
        df_submission["ID_2"] = extract_features(img_filter, roi)
        img_filter = LoG(padding_type="reflect", sigma_mm=3.0, cutoff=4.0, res_mm=res_mm, dimensionality="3D")
        df_submission["ID_3"] = extract_features(img_filter, roi)
        img_filter = Laws2(response_map="S5E5L5", padding_type="reflect", dimensionality="3D", rotation_invariance=True, pooling="max", energy_map=True, distance=5)
        df_submission["ID_4"] = extract_features(img_filter, roi)
        img_filter = Gabor(padding_type="reflect", res_mm=res_mm, sigma_mm=3.0, lambda_mm=3.0, gamma=1.0, theta=-5*np.pi/8, rotation_invariance=False, orthogonal_planes=False)
        df_submission["ID_5"] = extract_features(img_filter, roi)
        img_filter = Wavelets(wavelet_type="coif3", padding_type="reflect", response_map="LHH", decomposition_level=1, rotation_invariance=True)
        df_submission["ID_6"] = extract_features(img_filter, roi)

        # save results
        df_submission.to_csv(f"phase3/{patient_name}_{modality}.csv", index=False)
