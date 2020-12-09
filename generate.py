import os

import torch
import torch.nn as nn

from skimage.io import imsave
from skimage.util import img_as_ubyte

import argparse
import progressbar

# set-up parser
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="generator_model.tar", help="path to the model parameters")
parser.add_argument("--output_path", default="generated", help="path of the output")
parser.add_argument("--n_images", type=int, default=100, help="number of images to generate")
parser.add_argument("--batch_size", type=int, default=10, help="size of the batches")

args = vars(parser.parse_args())

# set-up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Image contrasting
def contrasting(image):
    """Linear contrast correction

        Parameters
        ----------
        image: Tensor
            batch of images to contrast

        Returns
        -------
        image: Tensor
            corrected image
    """

    image *= 255

    R = image[:, :, :, 0]
    G = image[:, :, :, 1]
    B = image[:, :, :, 2]

    Y = 0.2126*R + 0.7152*G + 0.0722*B
    U = -0.0999*R - 0.3360*G + 0.4360*B
    V = 0.6150*R - 0.5586*G - 0.0563*B

    new_y = Y.view(Y.shape[0], -1)

    x_min = torch.min(new_y, dim=1)[0]
    x_max = torch.max(new_y, dim=1)[0]

    out = (new_y-x_min[:, None])*255/(x_max-x_min)[:, None]

    Y = out.view(Y.shape)

    R = (Y + 1.2803*V).unsqueeze(3)
    G = (Y - 0.2148*U - 0.3805*V).unsqueeze(3)
    B = (Y + 2.1279*U).unsqueeze(3)

    output = torch.cat((R, G, B), 3)
    output = torch.clamp(output, 0, 255)/255

    return output


# model constants
latent_size = 128
channels = 3

# Generator
generator = nn.Sequential(
    # in: latent_size x 1 x 1

    nn.ConvTranspose2d(latent_size, 1024, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(1024),
    nn.ReLU(True),
    # out: 1024 x 4 x 4

    nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    # out: 512 x 8 x 8

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    # out: 256 x 16 x 16

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # out: 128 x 32 x 32

    nn.ConvTranspose2d(128, channels, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh())
    # out: 3 x 64 x 64

generator.to(device)

# initialize models and optimizers with trained parameters
if torch.cuda.is_available():
    generator.load_state_dict(torch.load(args["model_path"]))
else:
    generator.load_state_dict(torch.load(args["model_path"], map_location=device))

# set-up generator model
generator.to(device)
generator.eval()

# check whether the output directory created
os.makedirs(args["output_path"], exist_ok=True)

# set up the progress bar
widgets = ["Generating: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=args["n_images"], widgets=widgets).start()

# number of saved image in its output name
saved_num = 0

# saving loop
for batch_num in range(args["n_images"]//args["batch_size"]):

    # form random noise vector
    latent_vec = torch.randn(args["batch_size"], latent_size, 1, 1).to(device)
    # generate images from noise
    fake_images = generator(latent_vec).cpu().detach().permute(0, 2, 3, 1)
    # image normalization from [-1,1] to [0,1]
    fake_images = fake_images*0.5 + 0.5
    # image contrasting
    fake_images = contrasting(fake_images)
    # image conversion
    fake_images = img_as_ubyte(fake_images)
    # saving
    for image_num in range(args["batch_size"]):
        saved_num += 1
        imsave(os.path.join(args["output_path"], f"{saved_num}.jpg"), fake_images[image_num])
        pbar.update(saved_num)

pbar.finish()




