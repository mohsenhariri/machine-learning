import matplotlib.pyplot as plt
from pydicom import dcmread

DICOM_file_path = (
    "data_augmentation_dicom/data/100002/01-02-1999-NLST-LSS-55322/1-0OPAGELSPLUSD3602.512080.00.11.5-35858/1-100.dcm"
)


DICOM_file = dcmread(DICOM_file_path)


# plt.imshow(DICOM_file.pixel_array, cmap=plt.cm.gray)
# plt.show()
# print(DICOM_file)


arr = DICOM_file.pixel_array
plt.imshow(arr, cmap="gray")
plt.show()
