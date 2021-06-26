from PIL import Image
import PIL
import torch
import torchvision.transforms as transforms
from io import BytesIO
import sys
sys.path.append('model_cyclegan')


class ImageProcessing:
    def __init__(self, new_size, device):
        self.new_size = new_size
        self.device = device
        self.image_size = None

        self.loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.unloader = transforms.ToPILImage()

    def image_loader(self, image_name):
        image = Image.open(image_name)
        self.image_size = image.size
        image = PIL.ImageOps.pad(image, (self.new_size, self.new_size))
        image = self.loader(image).unsqueeze(0)

        return image.to(self.device, torch.float)

    def get_image(self, tensor):
        image = tensor[0].cpu().clone()
        image = image * 0.5 + 0.5
        image = self.unloader(image)
        image = PIL.ImageOps.fit(image, (self.image_size[0], self.image_size[1]))

        return image


def run_gan(wts_path, image_name):
    model_path = wts_path
    model = torch.load(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_processing = ImageProcessing(new_size=512, device=device)
    image = image_processing.image_loader(image_name)

    model.eval()
    output = model.test(image)

    new_image = image_processing.get_image(output)

    # transform PIL image to send to telegram
    bio = BytesIO()
    bio.name = 'output.jpeg'
    new_image.save(bio, 'JPEG')
    bio.seek(0)

    return bio
