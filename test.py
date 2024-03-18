import glob
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from PIL import Image
import torchvision
import model
#-----------------------------------------------------------------------------------------------------------------------
def populate_train_list(lowlight_images_path):
    image_list_lowlight = glob.glob(lowlight_images_path + "*.png")
    train_list = image_list_lowlight
    return train_list
class ImageDataset(Dataset):

    def __init__(self, test_path, resize = None):
        self.test_list = populate_train_list(test_path)
        self.data_list = self.test_list
        self.resize = resize
        print("Total testing examples:", len(self.data_list))
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(self.resize),
            transforms.ToTensor()
        ])
        img_name = self.data_list[idx]
        image = Image.open(img_name)
        image = tf(image)
        return image

if __name__ == "__main__":
    test_dataset = ImageDataset('test_data/', resize=((288, 384)))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    for idx, image in enumerate(test_loader):
        model_test = model.MMFF_NET().cuda()
        model_test.load_state_dict(torch.load(r'Epoch??.pth'))
        image = image.cuda()
        _, _, H, W = image.shape
        print(image.shape)
        enhance_img, r, Fu = model_test(image)
        img = torchvision.utils.make_grid(enhance_img).cpu().numpy()
        torchvision.utils.save_image(enhance_img,'kkk'+'%d.png'% (idx), padding = 0)
