from options.test_options import TestOptions
from models import create_model
import torch

import sys
sys.path.append('model_cyclegan')


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)
    if opt.eval:
        model.eval()

    torch.save(model, '../summer2winter.pth')
