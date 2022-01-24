from data_augmentation.ds_builder_2builder import train_loader
import torchvision
import matplotlib.pyplot as plt

sample_dataset_batch = next(iter(train_loader))
sample_input_batch = sample_dataset_batch[0]
sample_label_batch = sample_dataset_batch[1]

img_grid = torchvision.utils.make_grid(sample_input_batch)
# print(img_grid.size())
# print(img_grid.permute(1, 2, 0).size())
plt.imshow(img_grid.permute(1, 2, 0))
"""
about permute: 
https://stackoverflow.com/questions/51329159/how-can-i-generate-and-display-a-grid-of-images-in-pytorch-with-plt-imshow-and-t
torchvision.utils.make_grid() returns a tensor which contains the grid of images. But the channel dimension has to be moved to the end since that's what matplotlib recognizes
"""
plt.show()