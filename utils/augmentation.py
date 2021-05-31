"""
Some code in this file is from https://github.com/TengdaHan/DPC
"""

import collections
import math
import cv2
import numbers
import random

import numpy as np
import torchvision
import torchvision.transforms.functional as F
from PIL import ImageOps, Image
from torchvision import transforms


class Padding:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, img):
        return ImageOps.expand(img, border=self.pad, fill=0)

''' Supports both PIL and numpy arrays '''
class Scale:
    def __init__(self, size, interpolation=Image.NEAREST, asnumpy=False):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
        self.asnumpy = asnumpy

    def __call__(self, imgmap):
        # assert len(imgmap) > 1 # list of images
        img1 = imgmap[0]
        if isinstance(self.size, int):
            if self.asnumpy:
               w, h = img1.shape[1], img1.shape[0]
            else:
               w, h = img1.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return imgmap
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                if self.asnumpy:
                    return [cv2.resize(i, (ow, oh), cv2.INTER_NEAREST) for i in imgmap]
                else:
                    return [i.resize((ow, oh), self.interpolation) for i in imgmap]
            else:
                oh = self.size
                ow = int(self.size * w / h)
                if self.asnumpy:
                    return [cv2.resize(i, (ow, oh), cv2.INTER_NEAREST) for i in imgmap]
                else:
                    return [i.resize((ow, oh), self.interpolation) for i in imgmap]
        else:
            if self.asnumpy:
                return [cv2.resize(i, (self.size, self.size), cv2.INTER_NEAREST) for i in imgmap]
            else:
                return [i.resize(self.size, self.interpolation) for i in imgmap]

''' Supports both PIL and numpy arrays '''
class CenterCrop:
    def __init__(self, size, consistent=True, asnumpy=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.asnumpy = asnumpy

    def __call__(self, imgmap):
        img1 = imgmap[0]
        if self.asnumpy:
           w, h = img1.shape[1], img1.shape[0]
        else:
           w, h = img1.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        if self.asnumpy:
           return [i[y1:y1+th, x1:x1+tw] for i in imgmap]
        else:
           return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]


class RandomCropWithProb:
    def __init__(self, size, p=0.8, consistent=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.consistent = consistent
        self.threshold = p

    def __call__(self, imgmap):
        img1 = imgmap[0]
        w, h = img1.size
        if self.size is not None:
            th, tw = self.size
            if w == tw and h == th:
                return imgmap
            if self.consistent:
                if random.random() < self.threshold:
                    x1 = random.randint(0, w - tw)
                    y1 = random.randint(0, h - th)
                else:
                    x1 = int(round((w - tw) / 2.))
                    y1 = int(round((h - th) / 2.))
                return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]
            else:
                result = []
                for i in imgmap:
                    if random.random() < self.threshold:
                        x1 = random.randint(0, w - tw)
                        y1 = random.randint(0, h - th)
                    else:
                        x1 = int(round((w - tw) / 2.))
                        y1 = int(round((h - th) / 2.))
                    result.append(i.crop((x1, y1, x1 + tw, y1 + th)))
                return result
        else:
            return imgmap


class RandomCrop:
    def __init__(self, size, consistent=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.consistent = consistent

    def __call__(self, imgmap, flowmap=None):
        img1 = imgmap[0]
        w, h = img1.size
        if self.size is not None:
            th, tw = self.size
            if w == tw and h == th:
                return imgmap
            if not flowmap:
                if self.consistent:
                    x1 = random.randint(0, w - tw)
                    y1 = random.randint(0, h - th)
                    return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]
                else:
                    result = []
                    for i in imgmap:
                        x1 = random.randint(0, w - tw)
                        y1 = random.randint(0, h - th)
                        result.append(i.crop((x1, y1, x1 + tw, y1 + th)))
                    return result
            elif flowmap is not None:
                assert (not self.consistent)
                result = []
                for idx, i in enumerate(imgmap):
                    proposal = []
                    for j in range(3):  # number of proposal: use the one with largest optical flow
                        x = random.randint(0, w - tw)
                        y = random.randint(0, h - th)
                        proposal.append([x, y, abs(np.mean(flowmap[idx, y:y + th, x:x + tw, :]))])
                    [x1, y1, _] = max(proposal, key=lambda x: x[-1])
                    result.append(i.crop((x1, y1, x1 + tw, y1 + th)))
                return result
            else:
                raise ValueError('wrong case')
        else:
            return imgmap

''' Supports both PIL and numpy arrays '''
class RandomSizedCrop:
    def __init__(self, size, crop_area=(0.5, 1), interpolation=Image.BILINEAR, consistent=True, p=1.0, force_inside=False, asnumpy=False):
        self.size = size
        self.interpolation = interpolation
        self.consistent = consistent
        self.threshold = p
        self.crop_area = crop_area
        self.force_inside = force_inside
        self.asnumpy = asnumpy

    def random_crop_box(self, img):
        if self.asnumpy:
            w = img.shape[1]
            h = img.shape[0]
        else:
            w = img.size[0]
            h = img.size[1]

        # x_c and y_c describe the crop area center as values between 0 and 1.
        # w_c and h_c describe the crop lengths as width and height values in pixel.
        w_c, h_c, x_c, y_c = random_image_crop_square(min_area_n=self.crop_area[0],
                                                      max_area_n=self.crop_area[1],
                                                      image_height=h, image_width=w,
                                                      force_inside=self.force_inside)

        # Upper left corner has to be calculated in pixels. Always at least half of image.
        x_l = math.floor(min(max(0, x_c * w - w_c / 2), w))
        y_u = math.floor(min(max(0, y_c * h - h_c / 2), h))

        return int(x_l), int(y_u), int(x_l + w_c), int(y_u + h_c)

    def __call__(self, imgmap):
        img1 = imgmap[0]
        if random.random() < self.threshold:  # do RandomSizedCrop
            if self.consistent:
                c_box = self.random_crop_box(img1)
                if self.asnumpy:
                   imgmap = [i[c_box[1]:c_box[3], c_box[0]:c_box[2]] for i in imgmap]
                   for i in imgmap: assert (i.shape[1] == c_box[2] - c_box[0] and i.shape[0] == c_box[3] - c_box[1])
                else:
                   imgmap = [i.crop(c_box) for i in imgmap]
                   for i in imgmap: assert (i.size == (c_box[2] - c_box[0], c_box[3] - c_box[1]))
            else:
                imgmap = [i.crop(self.random_crop_box(i)) for i in imgmap]
            if self.asnumpy:
                imgmap = [cv2.resize(i, (self.size, self.size), cv2.INTER_NEAREST) for i in imgmap]
            else:
                imgmap = [i.resize((self.size, self.size), self.interpolation) for i in imgmap]

        else:  # don't do RandomSizedCrop, do CenterCrop
            crop = CenterCrop(self.size, asnumpy=self.asnumpy)
            return crop(imgmap)

        return imgmap


class RandomHorizontalFlip:
    def __init__(self, consistent=True, p=None, command=None):
        self.consistent = consistent
        if p is not None:
            self.threshold = p
        if command == 'left':
            self.threshold = 0
        elif command == 'right':
            self.threshold = 1
        else:
            self.threshold = 0.5



    def __call__(self, imgmap):
        if self.consistent:
            if random.random() < self.threshold:
                return [i.transpose(Image.FLIP_LEFT_RIGHT) for i in imgmap]
            else:
                return imgmap
        else:
            result = []
            for i in imgmap:
                if random.random() < self.threshold:
                    result.append(i.transpose(Image.FLIP_LEFT_RIGHT))
                else:
                    result.append(i)
            assert len(result) == len(imgmap)
            return result


class RandomGray:
    '''Actually it is a channel splitting, not strictly grayscale images'''

    def __init__(self, consistent=True, p=0.5):
        self.consistent = consistent
        self.p = p  # probability to apply grayscale

    def __call__(self, imgmap):
        if self.consistent:
            if random.random() < self.p:
                return [self.grayscale(i) for i in imgmap]
            else:
                return imgmap
        else:
            result = []
            for i in imgmap:
                if random.random() < self.p:
                    result.append(self.grayscale(i))
                else:
                    result.append(i)
            assert len(result) == len(imgmap)
            return result

    def grayscale(self, img):
        channel = np.random.choice(3)
        np_img = np.array(img)[:, :, channel]
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
        return img


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image. --modified from pytorch source code
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, consistent=True, p=1.0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.consistent = consistent
        self.threshold = p

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = torchvision.transforms.Compose(transforms)

        return transform

    def __call__(self, imgmap):
        if random.random() < self.threshold:  # do ColorJitter
            if self.consistent:
                transform = self.get_params(self.brightness, self.contrast,
                                            self.saturation, self.hue)
                return [transform(i) for i in imgmap]
            else:
                result = []
                for img in imgmap:
                    transform = self.get_params(self.brightness, self.contrast,
                                                self.saturation, self.hue)
                    result.append(transform(img))
                return result
        else:  # don't do ColorJitter, do nothing
            return imgmap

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


''' Supports both PIL and numpy arrays '''
class RandomRotation:
    def __init__(self, consistent=True, degree=15, p=1.0, asnumpy=False):
        self.consistent = consistent
        self.degree_high = degree if isinstance(degree, numbers.Number) else degree[1]
        self.degree_low = -degree if isinstance(degree, numbers.Number) else degree[0]
        self.threshold = p
        self.asnumpy = asnumpy

    ''' Rotation method for numpy images '''
    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def __call__(self, imgmap):
        if random.random() < self.threshold:  # do RandomRotation
            if self.consistent:
                deg = np.random.randint(self.degree_low, self.degree_high, 1)[0]
                if self.asnumpy:
                    return [self.rotate_image(i, deg) for i in imgmap]
                else:
                    return [i.rotate(deg, expand=True) for i in imgmap]
            else:
                if self.asnumpy:
                    return [self.rotate_image(i, np.random.randint(self.degree_low, self.degree_high, 1)[0]) for i in imgmap]
                else:
                    return [i.rotate(np.random.randint(self.degree_low, self.degree_high, 1)[0], expand=True) for i in imgmap]

        else:  # don't do RandomRotation, do nothing
            return imgmap


class ToTensor:
    def __call__(self, imgmap):
        totensor = transforms.ToTensor()
        return [totensor(i) for i in imgmap]


class Normalize:
    def __init__(self, mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)):
        self.mean = mean
        self.std = std

    def __call__(self, imgmap):
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        return [normalize(i) for i in imgmap]


class Denormalize:
    def __init__(self, mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)):
        self.mean = mean
        self.std = std
        self.inv_mean = [-mean[i] / std[i] for i in range(3)]
        self.inv_std = [1 / i for i in std]

    def __call__(self, imgmap):
        # TODO: make decision based on input type instead of the extra method for tensors.
        normalize = transforms.Normalize(mean=self.inv_mean, std=self.inv_std)
        return [normalize(i) for i in imgmap]

    def denormalize(self, img):
        return transforms.Normalize(mean=self.inv_mean, std=self.inv_std)(img)


def random_image_crop_square(min_area_n=0.4, max_area_n=1, image_width=150, image_height=150, force_inside=False):
    """
    This follows the conventions of https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops.html#nvidia.dali.ops.Crop
    Especially considering the meaning of crop_pos_x_norm and crop_pos_y_norm.
    """
    image_shorter = min(image_height, image_width)
    image_longer = max(image_height, image_width)

    # First find square crop length.
    total_area = image_shorter * image_shorter

    min_crop_length = math.ceil(math.sqrt(min_area_n * total_area))
    max_crop_length = math.floor(math.sqrt(max_area_n * total_area))

    min_crop_length = min(max(min_crop_length, 1.), image_shorter)
    max_crop_length = min(max_crop_length, image_shorter)

    crop_length = np.random.uniform(min_crop_length, max_crop_length)

    crop_length_x = crop_length
    crop_length_y = crop_length

    # Second, find upper left corner position. Normal distributed around center.
    crop_pos_x_norm = min(max(np.random.normal(loc=0.5, scale=1. / 6), 0.), 1.)  # Normal distributed between 0 and 1.
    crop_pos_y_norm = min(max(np.random.normal(loc=0.5, scale=1. / 6), 0.), 1.)  # Normal distributed between 0 and 1.

    if force_inside:
        center_x = crop_pos_x_norm * image_width
        center_y = crop_pos_y_norm * image_height

        if center_x - (crop_length / 2) < 0:
            crop_pos_x_norm = (crop_length / 2.) / image_width  # Leftmost without going outside.
        if center_x + (crop_length / 2) > image_width:
            crop_pos_x_norm = ((image_width - crop_length) / 2.) / image_width  # rightmost without going outside.

        if center_y - (crop_length / 2) < 0:
            crop_pos_y_norm = (crop_length / 2.) / image_height  # upmost without going outside.
        if center_y + (crop_length / 2) > image_height:
            crop_pos_y_norm = ((image_height - crop_length) / 2.) / image_height  # lowermost without going outside.


    return crop_length_x, crop_length_y, crop_pos_x_norm, crop_pos_y_norm
