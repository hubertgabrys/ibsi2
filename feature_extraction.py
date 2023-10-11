import numpy as np
from scipy.stats import iqr, skew, kurtosis


def calc_diag_n_voxel(arr):
    return np.count_nonzero(arr)


def calc_diag_n_voxel_interp_reseg(arr):
    return np.count_nonzero(~np.isnan(arr))


def calc_diag_mean_int_interp_reseg(arr):
    return np.nanmean(arr)


def calc_diag_max_int_interp_reseg(arr):
    return np.nanmax(arr)


def calc_diag_min_int_interp_reseg(arr):
    return np.nanmin(arr)


def calc_stat_mean(arr):
    return np.nanmean(arr)


def calc_stat_var(arr):
    return np.nanstd(arr) ** 2


def calc_stat_skew(arr):
    return skew(arr, axis=None, nan_policy='omit')


def calc_stat_kurt(arr):
    return kurtosis(arr, axis=None, nan_policy='omit')


def calc_stat_median(arr):
    return np.nanmedian(arr)


def calc_stat_min(arr):
    return np.nanmin(arr)


def calc_stat_p10(arr):
    return np.nanpercentile(arr, 10)


def calc_stat_p90(arr):
    return np.nanpercentile(arr, 90)


def calc_stat_max(arr):
    return np.nanmax(arr)


def calc_stat_iqr(arr):
    return iqr(arr, nan_policy='omit')


def calc_stat_range(arr):
    return np.nanmax(arr) - np.nanmin(arr)


def calc_stat_mad(arr):
    return np.nanmean(np.absolute(arr - np.nanmean(arr)))


def calc_stat_rmad(arr):
    arr2 = arr.copy()
    p10 = calc_stat_p10(arr2)
    p90 = calc_stat_p90(arr2)
    ind = np.where((arr2 < p10) | (arr2 > p90))
    arr2[ind] = np.nan
    return calc_stat_mad(arr2)


def calc_stat_medad(arr):
    return np.nanmean(np.absolute(arr-np.nanmedian(arr)))


def calc_stat_cov(arr):
    return np.nanstd(arr) / np.nanmean(arr)


def calc_stat_qcod(arr):
    p25 = np.nanpercentile(arr, 25)
    p75 = np.nanpercentile(arr, 75)
    return (p75 - p25) / (p75 + p25)


def calc_stat_energy(arr):
    return np.nansum(arr ** 2)


def calc_stat_rms(arr):
    return np.sqrt(np.nanmean(arr ** 2))


def extract_phase2_features(data_oryg, data_reseg, data_filtered):
    features_dict = {}
    features_dict['diag_n_voxel'] = calc_diag_n_voxel(data_oryg)
    features_dict['diag_n_voxel_interp_reseg'] = calc_diag_n_voxel_interp_reseg(data_reseg)
    features_dict['diag_mean_int_interp_reseg'] = calc_diag_mean_int_interp_reseg(data_reseg)
    features_dict['diag_max_int_interp_reseg'] = calc_diag_max_int_interp_reseg(data_reseg)
    features_dict['diag_min_int_interp_reseg'] = calc_diag_min_int_interp_reseg(data_reseg)
    features_dict['stat_mean'] = calc_stat_mean(data_filtered)
    features_dict['stat_var'] = calc_stat_var(data_filtered)
    features_dict['stat_skew'] = calc_stat_skew(data_filtered)
    features_dict['stat_kurt'] = calc_stat_kurt(data_filtered)
    features_dict['stat_median'] = calc_stat_median(data_filtered)
    features_dict['stat_min'] = calc_stat_min(data_filtered)
    features_dict['stat_p10'] = calc_stat_p10(data_filtered)
    features_dict['stat_p90'] = calc_stat_p90(data_filtered)
    features_dict['stat_max'] = calc_stat_max(data_filtered)
    features_dict['stat_iqr'] = calc_stat_iqr(data_filtered)
    features_dict['stat_range'] = calc_stat_range(data_filtered)
    features_dict['stat_mad'] = calc_stat_mad(data_filtered)
    features_dict['stat_rmad'] = calc_stat_rmad(data_filtered)
    features_dict['stat_medad'] = calc_stat_medad(data_filtered)
    features_dict['stat_cov'] = calc_stat_cov(data_filtered)
    features_dict['stat_qcod'] = calc_stat_qcod(data_filtered)
    features_dict['stat_energy'] = calc_stat_energy(data_filtered)
    features_dict['stat_rms'] = calc_stat_rms(data_filtered)
    return features_dict