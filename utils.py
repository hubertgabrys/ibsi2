import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
import SimpleITK as sitk
import contextlib
import joblib


class BaseImage:
    def __init__(self):
        self.image = None
        self.direction = None
        self.origin = None
        self.shape = None
        self.pixel_array = None
        self.spacing = None

    def _update_image_fields(self, image):
        self.image = image
        self.pixel_array = sitk.GetArrayFromImage(self.image)
        self.shape = self.image.GetSize()
        self.spacing = self.image.GetSpacing()
        self.origin = self.image.GetOrigin()
        self.direction = self.image.GetDirection()

    def run_diagnostics(self):
        print(f"Shape: {self.shape}")
        print(f"Spacing: {self.spacing}")
        print(f"Origin: {self.origin}")
        print(f"Direction: {self.direction}")
        print(f"Image size x: {self.shape[1] * self.spacing[1]:.3f} mm")
        print(f"Image size y: {self.shape[0] * self.spacing[0]:.3f} mm")
        print(f"Image size z: {self.shape[2] * self.spacing[2]:.3f} mm")
        print(f"Image dimension x: {self.shape[1]}")
        print(f"Image dimension y: {self.shape[0]}")
        print(f"Image dimension z: {self.shape[2]}")
        print(f"Pixel size x: {self.spacing[1]:.3f} mm")
        print(f"Pixel size y: {self.spacing[0]:.3f} mm")
        print(f"Pixel size z: {self.spacing[2]:.3f} mm")
        print(f"Mean intensity: {np.nanmean(self.pixel_array):.0f}")
        print(f"Min intensity: {np.nanmin(self.pixel_array):.0f}")
        print(f"Max intensity: {np.nanmax(self.pixel_array):.0f}")

    def read_nifti(self, nifti_path):
        # read image
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(nifti_path)
        image = reader.Execute()

        # cast int to float
        cast_filter = sitk.CastImageFilter()
        cast_filter.SetOutputPixelType(sitk.sitkFloat64)
        image = cast_filter.Execute(image)

        # fill out the fields
        self._update_image_fields(image)

    def interpolate(self, new_spacing, method=sitk.sitkLinear):
        # calculate new shape
        new_shape = np.array(self.shape) * (self.spacing / np.array(new_spacing))
        new_shape = np.ceil(new_shape).astype(int)
        new_shape = [int(s) for s in new_shape]

        # calculate new origin
        def get_new_origin(axis=0):
            """https://arxiv.org/pdf/1612.07003.pdf QCY4"""
            n_a = self.shape[axis]
            s_a = self.spacing[axis]
            s_b = new_spacing[axis]
            n_b = np.ceil((n_a * s_a) / s_b)
            x_b = self.origin[axis] + (s_a * (n_a - 1) - s_b * (n_b - 1)) / 2
            return x_b

        new_origin = [get_new_origin(axis) for axis in range(3)]

        # resample
        resample = sitk.ResampleImageFilter()
        resample.SetSize(new_shape)
        resample.SetOutputSpacing(new_spacing)
        resample.SetOutputOrigin(new_origin)
        resample.SetOutputDirection(self.direction)
        resample.SetInterpolator(method)
        resampled_image = resample.Execute(self.image)
        self._update_image_fields(resampled_image)

    def round_intensities(self):
        self.image = sitk.Round(self.image)


class Image(BaseImage):
    def __init__(self, modality):
        super().__init__()
        self.modality = modality


class Mask(BaseImage):
    def __init__(self):
        super().__init__()


class ROI:
    def __init__(self, image, mask, range_l=-np.inf, range_h=np.inf):
        self.image = sitk.GetArrayFromImage(image.image)
        self.image = self._reorient_image(self.image)
        self.morphological_mask = sitk.GetArrayFromImage(mask.image)
        self.morphological_mask = self._reorient_image(self.morphological_mask)
        self.intensity_mask = self._get_intensity_mask()
        self.range_l = range_l
        self.range_h = range_h
        self.resegmented_mask = self._get_resegmented_mask()

    def _reorient_image(self, arr):
        arr = arr.transpose(2, 1, 0)
        arr = np.rot90(arr, axes=(0, 1))
        arr = np.flipud(arr)
        return arr

    def _get_intensity_mask(self):
        ind = np.where(self.morphological_mask == 0)
        intensity_mask = self.image.copy()
        intensity_mask[ind] = np.nan
        return intensity_mask

    def _get_resegmented_mask(self):
        resegmented_mask = self.intensity_mask.copy()
        ind_l = np.where(resegmented_mask < self.range_l)
        ind_h = np.where(resegmented_mask > self.range_h)
        resegmented_mask[ind_l] = np.nan
        resegmented_mask[ind_h] = np.nan
        return resegmented_mask

    def run_diagnostics(self):
        print(f"Intensity mask voxel count: {np.count_nonzero(~np.isnan(self.intensity_mask))}")
        print(f"Morphological mask voxel count: {np.count_nonzero(self.morphological_mask)}")
        print(f"Intensity mask mean intensity: {np.nanmean(self.intensity_mask):.1f}")
        print(f"Intensity mask min intensity: {np.nanmin(self.intensity_mask)}")
        print(f"Intensity mask max intensity: {np.nanmax(self.intensity_mask)}")


def nifti_loader(phantom_name):
    """Function to load nii/gz files"""
    phantom_path = f"data_sets/ibsi_2_digital_phantom/nifti/{phantom_name}/image/{phantom_name}.nii.gz"
    dataset = nib.load(phantom_path)
    data = dataset.get_fdata()
    data = np.flipud(np.rot90(data, axes=(0, 1)))
    return data


def plot_slice(img, z_idx=32, cmap='gray', title=""):
    plt.figure(figsize=(9, 9))
    plt.title(title)
    z_slice = img[:, :, z_idx]
    plt.imshow(z_slice, interpolation="nearest", cmap=cmap)
    plt.show(block=False)


def save_img(filtered_data, filename):
    filtered_data = np.rot90(np.flipud(filtered_data), axes=(1, 0))
    filtered_image = nib.Nifti1Image(filtered_data, affine=np.eye(4))
    filtered_image.header['pixdim'][1:4] = [2, 2, 2]
    nib.save(filtered_image, filename)


def print_min_max(filtered_data, test_id=''):
    print(test_id)
    print(f"{np.nanmin(filtered_data):.4f}")
    print(f"{np.nanmax(filtered_data):.4f}")
    print('')


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument.
    source: https://stackoverflow.com/a/58936697/3859823
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
