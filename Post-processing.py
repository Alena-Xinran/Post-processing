import os
import nibabel as nib
import numpy as np
from scipy.ndimage import label

def remove_small_components(data, spacing, min_radius=1.5):
    """
    Remove connected components with radius smaller than min_radius from the 3D image data.
    :param data: 3D numpy array of the image data
    :param spacing: Tuple of spacing values in (x, y, z) direction
    :param min_radius: Minimum radius of connected components to keep (in millimeters)
    :return: Cleaned 3D image data with small components removed
    """
    # Label connected components
    labeled_array, num_features = label(data)
    
    # Calculate the volume of each connected component
    component_sizes = np.bincount(labeled_array.ravel())
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    component_volumes = component_sizes * voxel_volume
    
    # Calculate the minimum volume based on the minimum radius
    min_volume = (4/3) * np.pi * (min_radius ** 3)
    
    # Create a mask to remove small components
    mask = component_volumes >= min_volume
    mask[0] = 0  # Ensure the background is not included
    cleaned_data = mask[labeled_array]
    
    return cleaned_data

def process_nifti_file(file_path, min_radius=1.5):
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        spacing = img.header.get_zooms()

        # Remove small connected components
        cleaned_data = remove_small_components(data, spacing, min_radius)

        # Check if the processed image is all zero
        if np.any(cleaned_data > 0):
            return cleaned_data, img.affine, img.header
        else:
            return None, None, None
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None

def check_intersection(tumor_data, organ_data):
    # Check if there is any intersection between tumor_data and organ_data
    return np.any((tumor_data > 0) & (organ_data > 0))

def delete_existing_new_files(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('_new.nii.gz'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted existing file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

def main():
    base_dir = '/home/lccai/lxr/DiffTumor/dataset/output2'

    # Delete existing _new.nii.gz files
    delete_existing_new_files(base_dir)
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('_tumor.nii.gz'):
                tumor_file_path = os.path.join(root, file)
                
                # Determine the corresponding organ file
                if 'liver_tumor' in file:
                    organ_file_path = os.path.join(root, 'liver.nii.gz')
                elif 'pancreas_tumor' in file:
                    organ_file_path = os.path.join(root, 'pancreas.nii.gz')
                elif 'kidney_tumor' in file:
                    organ_file_path = os.path.join(root, 'kidney.nii.gz')
                else:
                    print(f"Unknown organ for file: {file}")
                    continue
                
                if os.path.exists(organ_file_path):
                    organ_img = nib.load(organ_file_path)
                    organ_data = organ_img.get_fdata()
                    
                    # Load tumor file and check intersection
                    tumor_img = nib.load(tumor_file_path)
                    tumor_data = tumor_img.get_fdata()
                    
                    if check_intersection(tumor_data, organ_data):
                        processed_data, affine, header = process_nifti_file(tumor_file_path)

                        if processed_data is not None:
                            output_file_path = os.path.join(root, file.replace('.nii.gz', '_new77777.nii.gz'))
                            new_img = nib.Nifti1Image(processed_data, affine, header)
                            nib.save(new_img, output_file_path)
                            print(f"Processed and saved: {output_file_path}")
                        else:
                            print(f"No mask found in: {tumor_file_path}")
                    else:
                        print(f"No intersection found for: {tumor_file_path}")
                else:
                    print(f"Organ file not found for: {tumor_file_path}")

if __name__ == "__main__":
    main()
