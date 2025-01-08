import nibabel as nib
import matplotlib.pyplot as plt
import os
import pydicom
from PIL import Image
import numpy as np

dir_path = "nii files"
output_dir="dcm_files"
os.makedirs(output_dir, exist_ok=True)
for f in os.listdir(dir_path):
    if f.endswith(".nii"):
        if os.path.isfile((os.path.join(dir_path,f))):
            file_path=os.path.join(dir_path,f)
            img = nib.load(file_path)
            data = img.get_fdata()

            output_dir = f.split(".")[0]
            os.makedirs(output_dir, exist_ok=True)

            for i in range(data.shape[2]):
                plt.imshow(data[:, :, i], cmap="gray")
                plt.axis('off')
                output_path = os.path.join(output_dir, f"slice_{i:03d}.png")
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                plt.close()

    elif f.endswith(".dcm"):
        if os.path.isfile((os.path.join(dir_path,f))):
            file_path=os.path.join(dir_path,f)
            dicom_data = pydicom.dcmread(file_path)
            image_array = dicom_data.pixel_array
            image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255
            image_array = image_array.astype(np.uint8)
            output_image = Image.fromarray(image_array)
            output_image.save(output_dir+"/"+f.split(".")[0]+".png")

print(f"Convertire terminata!")

