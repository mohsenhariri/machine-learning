from zipfile import ZipFile
import tarfile

from io import BytesIO

file_name = "./data/ImageNet/imagenet-object-localization-challenge.zip"
file_n = "./data/ImageNet/imagenet-object-localization-challenge.zip/imagenet_object_localization_patched2019.tar.gz"

# opening the zip file in READ mode
# with ZipFile(file_name, "r") as zfile:
#     # printing all the contents of the zip file
#     # zfile.printdir()
#     zfiledata = BytesIO(zfile.read("imagenet_object_localization_patched2019.tar.gz"))
#     print(zfiledata)
#     with tarfile.open()



# with tarfile.open(file_n, "w:gz") as f:
#   pass


# with ZipFile(file_name, "r") as zfile:
#     # printing all the contents of the zip file
#     # zfile.printdir()
#     zfiledata = BytesIO(zfile.read("imagenet_object_localization_patched2019.tar.gz"))
#     print(zfiledata)
#     with tarfile.open()

