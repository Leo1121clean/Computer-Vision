from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
import cv2
import heapq
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing function for Lite-Mono models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)

    parser.add_argument('--load_weights_folder', type=str,
                        help='path of a pretrained model to use',
                        )

    parser.add_argument('--model', type=str,
                        help='name of a pretrained model to use',
                        default="lite-mono",
                        choices=[
                            "lite-mono",
                            "lite-mono-small",
                            "lite-mono-tiny",
                            "lite-mono-8m"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")

    return parser.parse_args()


class depth_estimator(object):
    def __init__(self,
                 model="lite-mono-8m",
                 load_weights_folder=os.path.expanduser("~") + "/Lite-Mono/lite-mono-8m_1024x320/"
                 ) -> None:

        assert load_weights_folder is not None, \
            "You must specify the --load_weights_folder parameter"

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print("-> Loading model from ", load_weights_folder)
        encoder_path = os.path.join(load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)
        decoder_dict = torch.load(decoder_path)

        # extract the height and width of image that this model was trained with
        self.feed_height = encoder_dict['height']
        self.feed_width = encoder_dict['width']

        # LOADING PRETRAINED MODEL
        print("   Loading pretrained encoder")
        self.encoder = networks.LiteMono(model=model,
                                         height=self.feed_height,
                                         width=self.feed_width)

        model_dict = self.encoder.state_dict()
        self.encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

        self.encoder.to(self.device)
        self.encoder.eval()

        print("Loading pretrained decoder")
        self.depth_decoder = networks.DepthDecoder(self.encoder.num_ch_enc, scales=range(3))
        depth_model_dict = self.depth_decoder.state_dict()
        self.depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

        self.depth_decoder.to(self.device)
        self.depth_decoder.eval()

    def test_simple(self, args):
        # FINDING INPUT IMAGES
        if os.path.isfile(args.image_path) and not args.test:
            # Only testing on a single image
            paths = [args.image_path]
            output_directory = os.path.dirname(args.image_path)
        elif os.path.isdir(args.image_path):
            # Searching folder for images
            paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
            output_directory = args.image_path
        else:
            raise Exception("Can not find args.image_path: {}".format(args.image_path))

        print("-> Predicting on {:d} test images".format(len(paths)))

        # PREDICTING ON EACH IMAGE IN TURN
        with torch.no_grad():
            for idx, image_path in enumerate(paths):

                if image_path.endswith("_disp.jpg"):
                    # don't try to predict disparity for a disparity image!
                    continue

                # Load image and preprocess
                input_image = pil.open(image_path).convert('RGB')
                original_width, original_height = input_image.size
                input_image = input_image.resize((self.feed_width, self.feed_height), pil.LANCZOS)
                input_image = transforms.ToTensor()(input_image).unsqueeze(0)

                # PREDICTION
                input_image = input_image.to(self.device)
                features = self.encoder(input_image)
                outputs = self.depth_decoder(features)

                disp = outputs[("disp", 0)]

                disp_resized = torch.nn.functional.interpolate(
                    disp, (original_height, original_width), mode="bilinear", align_corners=False)

                # Saving numpy file
                output_name = os.path.splitext(os.path.basename(image_path))[0]
                # output_name = os.path.splitext(image_path)[0].split('/')[-1]
                scaled_disp, depth = disp_to_depth(disp_resized, 2, 100)
                depth = depth.squeeze().cpu().numpy()

                name_dest_npy = os.path.join(output_directory, "{}_depth.npy".format(output_name))
                np.save(name_dest_npy, depth)

                # Saving colormapped depth image
                disp_resized_np = disp_resized.squeeze().cpu().numpy()
                vmax = np.percentile(disp_resized_np, 95)
                normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
                mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                im = pil.fromarray(colormapped_im)

                name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
                im.save(name_dest_im)

                print("   Processed {:d} of {:d} images - saved predictions to:".format(
                    idx + 1, len(paths)))
                print("   - {}".format(name_dest_im))
                print("   - {}".format(name_dest_npy))

                


        print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    grasp_pose = depth_estimator()
    grasp_pose.test_simple(args)

