import numpy as np

import utils
from filters import Mean, LoG, Laws, Gabor, Wavelets

dataset_directory = "data_sets/ibsi_2_digital_phantom/nifti/"


# 1.a.1
# IBSI zero padding is equivalent to SciPy "constant" padding with cval=0.
test_id = "1a1"
data = utils.nifti_loader(phantom_name="checkerboard")
mean_filter = Mean(padding_type="constant", support=15, dimensionality="3D")
filtered_data = mean_filter.filter(data)
utils.print_min_max(filtered_data, test_id=test_id)
utils.plot_slice(filtered_data, title=test_id)
utils.save_img(filtered_data, filename=f"phase1/{test_id}.nii.gz")

# 1.a.2
test_id = "1a2"
data = utils.nifti_loader(phantom_name="checkerboard")
mean_filter = Mean(padding_type="nearest", support=15, dimensionality="3D")
filtered_data = mean_filter.filter(data)
utils.print_min_max(filtered_data, test_id=test_id)
utils.plot_slice(filtered_data, title=test_id)
utils.save_img(filtered_data, filename=f"phase1/{test_id}.nii.gz")

# 1.a.3
# IBSI periodic padding is equivalent to SciPy "wrap" padding.
test_id = "1a3"
data = utils.nifti_loader(phantom_name="checkerboard")
mean_filter = Mean(padding_type="wrap", support=15, dimensionality="3D")
filtered_data = mean_filter.filter(data)
utils.print_min_max(filtered_data, test_id=test_id)
utils.plot_slice(filtered_data, title=test_id)
utils.save_img(filtered_data, filename=f"phase1/{test_id}.nii.gz")

# 1.a.4
# IBSI mirror padding is equivalent to SciPy "reflect" padding.
test_id = "1a4"
data = utils.nifti_loader(phantom_name="checkerboard")
mean_filter = Mean(padding_type="reflect", support=15, dimensionality="3D")
filtered_data = mean_filter.filter(data)
utils.print_min_max(filtered_data, test_id=test_id)
utils.plot_slice(filtered_data, title=test_id)
utils.save_img(filtered_data, filename=f"phase1/{test_id}.nii.gz")

# 1.b.1
test_id = "1b1"
data = utils.nifti_loader(phantom_name="impulse")
mean_filter = Mean(padding_type="constant", support=15, dimensionality="2D")
filtered_data = mean_filter.filter(data)
utils.print_min_max(filtered_data, test_id=test_id)
utils.plot_slice(filtered_data, title=test_id)
utils.save_img(filtered_data, filename=f"phase1/{test_id}.nii.gz")

# 2.a
test_id = "2a"
data = utils.nifti_loader(phantom_name="impulse")
log = LoG(padding_type="constant", sigma_mm=3, cutoff=4, padding_constant=0.0, res_mm=2, dimensionality="3D")
filtered_data = log.filter(data)
utils.print_min_max(filtered_data, test_id=test_id)
utils.plot_slice(filtered_data, title=test_id)
utils.save_img(filtered_data, filename=f"phase1/{test_id}.nii.gz")

# 2.b
test_id = "2b"
data = utils.nifti_loader(phantom_name="checkerboard")
log = LoG(padding_type="reflect", sigma_mm=5.0, cutoff=4.0, padding_constant=0.0, res_mm=2.0, dimensionality="3D")
filtered_data = log.filter(data)
utils.print_min_max(filtered_data, test_id=test_id)
utils.plot_slice(filtered_data, title=test_id)
utils.save_img(filtered_data, filename=f"phase1/{test_id}.nii.gz")

# 2.c
test_id = "2c"
data = utils.nifti_loader(phantom_name="checkerboard")
log = LoG(padding_type="reflect", sigma_mm=5.0, cutoff=4.0, padding_constant=0.0, res_mm=2.0, dimensionality="2D")
filtered_data = log.filter(data)
utils.print_min_max(filtered_data, test_id=test_id)
utils.plot_slice(filtered_data, title=test_id)
utils.save_img(filtered_data, filename=f"phase1/{test_id}.nii.gz")

# 3.a.1
test_id = "3a1"
data = utils.nifti_loader(phantom_name="impulse")
laws = Laws(response_map="E5L5S5", padding_type="constant", dimensionality="3D")
filtered_data = laws.filter(data)
utils.print_min_max(filtered_data, test_id=test_id)
utils.plot_slice(filtered_data, z_idx=31, title=test_id)
utils.save_img(filtered_data, filename=f"phase1/{test_id}.nii.gz")

# 3.a.2
test_id = "3a2"
resp_map = "E5L5S5"
data = utils.nifti_loader(phantom_name="impulse")
laws = Laws(response_map="E5L5S5", padding_type="constant", dimensionality="3D", rotation_invariance=True, pooling="max")
filtered_data = laws.filter(data)
utils.print_min_max(filtered_data, test_id=test_id)
utils.plot_slice(filtered_data, z_idx=32, title=test_id)
utils.save_img(filtered_data, filename=f"phase1/{test_id}.nii.gz")

# 3.a.3
test_id = "3a3"
data = utils.nifti_loader(phantom_name="impulse")
laws = Laws(response_map="E5L5S5", padding_type="constant", dimensionality="3D", rotation_invariance=True, pooling="max")
filtered_data = laws.filter(data)
energy_map = laws.get_energy_map(filtered_data, distance=7)
utils.print_min_max(filtered_data, test_id=test_id)
utils.plot_slice(energy_map, z_idx=32, title=test_id)
utils.save_img(energy_map, filename=f"phase1/{test_id}.nii.gz")

# 3.b.1
test_id = "3b1"
data = utils.nifti_loader(phantom_name="checkerboard")
laws = Laws(response_map="E3W5R5", padding_type="reflect", dimensionality="3D")
filtered_data = laws.filter(data)
utils.print_min_max(filtered_data, test_id=test_id)
utils.plot_slice(filtered_data, z_idx=31, title=test_id)
utils.save_img(filtered_data, filename=f"phase1/{test_id}.nii.gz")

# 3.b.2
test_id = "3b2"
data = utils.nifti_loader(phantom_name="checkerboard")
laws = Laws(response_map="E3W5R5", padding_type="reflect", dimensionality="3D", rotation_invariance=True, pooling="max")
filtered_data = laws.filter(data)
utils.print_min_max(filtered_data, test_id=test_id)
utils.plot_slice(filtered_data, z_idx=31, title=test_id)
utils.save_img(filtered_data, filename=f"phase1/{test_id}.nii.gz")

# 3.b.3
test_id = "3b3"
data = utils.nifti_loader(phantom_name="checkerboard")
laws = Laws(response_map="E3W5R5", padding_type="reflect", dimensionality="3D", rotation_invariance=True, pooling="max")
filtered_data = laws.filter(data)
energy_map = laws.get_energy_map(filtered_data, distance=7)
utils.print_min_max(energy_map, test_id=test_id)
utils.plot_slice(energy_map, z_idx=31, title=test_id)
utils.save_img(energy_map, filename=f"phase1/{test_id}.nii.gz")

# 3.c.1
test_id = "3c1"
data = utils.nifti_loader(phantom_name="checkerboard")
laws = Laws(response_map="L5S5", padding_type="reflect", dimensionality="2D")
filtered_data = laws.filter(data)
utils.print_min_max(filtered_data, test_id=test_id)
utils.plot_slice(filtered_data, z_idx=31, title=test_id)
utils.save_img(filtered_data, filename=f"phase1/{test_id}.nii.gz")

# 3.c.2
test_id = "3c2"
data = utils.nifti_loader(phantom_name="checkerboard")
laws = Laws(response_map="L5S5", padding_type="reflect", dimensionality="2D", rotation_invariance=True, pooling="max")
filtered_data = laws.filter(data)
utils.print_min_max(filtered_data, test_id=test_id)
utils.plot_slice(filtered_data, z_idx=31, title=test_id)
utils.save_img(filtered_data, filename=f"phase1/{test_id}.nii.gz")

# 3.c.3
test_id = "3c3"
data = utils.nifti_loader(phantom_name="checkerboard")
laws = Laws(response_map="L5S5", padding_type="reflect", dimensionality="2D", rotation_invariance=True, pooling="max")
filtered_data = laws.filter(data)
energy_map = laws.get_energy_map(filtered_data, distance=7)
utils.print_min_max(energy_map, test_id=test_id)
utils.plot_slice(energy_map, z_idx=31, title=test_id)
utils.save_img(energy_map, filename=f"phase1/{test_id}.nii.gz")

# 4.a.1
test_id = "4a1"
data = utils.nifti_loader(phantom_name="impulse")
gabor = Gabor(padding_type="constant", res_mm=2.0, sigma_mm=10.0, lambda_mm=4.0, gamma=1/2, theta=np.pi/3)
filtered_data = gabor.filter(data)
utils.print_min_max(filtered_data, test_id=test_id)
utils.plot_slice(filtered_data, z_idx=31, title=test_id)
utils.save_img(filtered_data, filename=f"phase1/{test_id}.nii.gz")

# 4.a.2
test_id = "4a2"
data = utils.nifti_loader(phantom_name="impulse")
gabor = Gabor(padding_type="constant", res_mm=2.0, sigma_mm=10.0, lambda_mm=4.0, gamma=1/2, theta=np.pi/4,
              rotation_invariance=True)
filtered_data = gabor.filter(data)
utils.print_min_max(filtered_data, test_id=test_id)
utils.plot_slice(filtered_data, z_idx=31, title=test_id)
utils.save_img(filtered_data, filename=f"phase1/{test_id}.nii.gz")

# 4.b.1
test_id = "4b1"
data = utils.nifti_loader(phantom_name="sphere")
gabor = Gabor(padding_type="reflect", res_mm=2.0, sigma_mm=20.0, lambda_mm=8.0, gamma=5/2, theta=5*np.pi/4)
filtered_data = gabor.filter(data)
utils.print_min_max(filtered_data, test_id=test_id)
utils.plot_slice(filtered_data, z_idx=31, title=test_id)
utils.save_img(filtered_data, filename=f"phase1/{test_id}.nii.gz")

# 4.b.2
test_id = "4b2"
data = utils.nifti_loader(phantom_name="sphere")
gabor = Gabor(padding_type="reflect", res_mm=2.0, sigma_mm=20.0, lambda_mm=8.0, gamma=5/2, theta=np.pi/8,
              rotation_invariance=True)
filtered_data = gabor.filter(data)
utils.print_min_max(filtered_data, test_id=test_id)
utils.plot_slice(filtered_data, z_idx=31, title=test_id)
utils.save_img(filtered_data, filename=f"phase1/{test_id}.nii.gz")

# 5.a.1
test_id = "5a1"
data = utils.nifti_loader(phantom_name="impulse")
wavelets = Wavelets(wavelet_type="db2", padding_type="constant", response_map="LHL", decomposition_level=1,
                    rotation_invariance=False)
filtered_data = wavelets.filter(data)
utils.print_min_max(filtered_data, test_id=test_id)
utils.plot_slice(filtered_data, z_idx=31, title=test_id)
utils.save_img(filtered_data, filename=f"phase1/{test_id}.nii.gz")

# 5.a.2
test_id = "5a2"
data = utils.nifti_loader(phantom_name="impulse")
wavelets = Wavelets(wavelet_type="db2", padding_type="constant", response_map="LHL", decomposition_level=1,
                    rotation_invariance=True)
filtered_data = wavelets.filter(data)
utils.print_min_max(filtered_data, test_id=test_id)
utils.plot_slice(filtered_data, z_idx=31, title=test_id)
utils.save_img(filtered_data, filename=f"phase1/{test_id}.nii.gz")

# 6.a.1
test_id = "6a1"
data = utils.nifti_loader(phantom_name="sphere")
wavelets = Wavelets(wavelet_type="coif1", padding_type="wrap", response_map="HHL", decomposition_level=1,
                    rotation_invariance=False)
filtered_data = wavelets.filter(data)
utils.print_min_max(filtered_data, test_id=test_id)
utils.plot_slice(filtered_data, z_idx=31, title=test_id)
utils.save_img(filtered_data, filename=f"phase1/{test_id}.nii.gz")

# 6.a.2
test_id = "6a2"
data = utils.nifti_loader(phantom_name="sphere")
wavelets = Wavelets(wavelet_type="coif1", padding_type="wrap", response_map="HHL", decomposition_level=1,
                    rotation_invariance=True)
filtered_data = wavelets.filter(data)
utils.print_min_max(filtered_data, test_id=test_id)
utils.plot_slice(filtered_data, z_idx=31, title=test_id)
utils.save_img(filtered_data, filename=f"phase1/{test_id}.nii.gz")

# 7.a.1
test_id = "7a1"
data = utils.nifti_loader(phantom_name="checkerboard")
wavelets = Wavelets(wavelet_type="haar", padding_type="reflect", response_map="LLL", decomposition_level=2,
                    rotation_invariance=True)
filtered_data = wavelets.filter(data)
utils.print_min_max(filtered_data, test_id=test_id)
utils.plot_slice(filtered_data, z_idx=31, title=test_id)
utils.save_img(filtered_data, filename=f"phase1/{test_id}.nii.gz")

# 7.a.2
test_id = "7a2"
data = utils.nifti_loader(phantom_name="checkerboard")
wavelets = Wavelets(wavelet_type="haar", padding_type="reflect", response_map="HHH", decomposition_level=2,
                    rotation_invariance=True)
filtered_data = wavelets.filter(data)
utils.print_min_max(filtered_data, test_id=test_id)
utils.plot_slice(filtered_data, z_idx=31, title=test_id)
utils.save_img(filtered_data, filename=f"phase1/{test_id}.nii.gz")
