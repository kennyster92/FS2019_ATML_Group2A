from torch.utils.data import Dataset
from skimage import img_as_float


class MelanomaDataset(Dataset):

    def __init__(self, image, label, transform=None):
        self.image = image  # our image
        self.label = label  # our diagnosy
        self.transform = transform

    def __getitem__(self, index):
        # Anything could go here, e.g. image loading from file or a different structure
        # must return image and center
        sel_image = img_as_float(self.image[index])
        sel_label = self.label[index].astype('float64')

        if self.transform is not None:
            sel_image = self.transform(sel_image)

        return sel_image, sel_label  # return 2 tensors

    def __len__(self):
        return len(self.image)  # return how many images and center we have
