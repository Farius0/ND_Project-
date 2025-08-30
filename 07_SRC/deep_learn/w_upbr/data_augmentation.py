import numpy as np
import torch
import torchvision.transforms.functional as F

from numpy import random
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import transforms as T


def build_transforms(transforms_):
    transforms = []
    for transform in transforms_:
        transforms.append(transform)
    return Compose(transforms)


class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, **kwargs):
        for t in self.transforms:
            img = t(img, **kwargs)
        return img

    # def __repr__(self):
    #     format_string = self.__class__.__name__ + '('
    #     for t in self.transforms:
    #         format_string += '\n'
    #         format_string += '    {0}'.format(t)
    #     format_string += '\n)'
    #     return format_string

class ToTensor:
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    """

    def __call__(self, sample):
        return {'image': F.to_tensor(sample['image']),
                'label': torch.as_tensor(np.array(sample['label']), dtype=torch.int64)}

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToRGB:

    def __call__(self, sample):
        if sample['image'].shape[0] == 1:
            sample['image'] = sample['image'].repeat(3, 1, 1)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ConvertImageDtype(torch.nn.Module):

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, sample):
        sample['image'] = F.convert_image_dtype(sample['image'], self.dtype)
        sample['label'] = F.convert_image_dtype(sample['label'], self.dtype)
        return sample


class ToPILImage:

    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, sample):
        sample['image'] = F.to_pil_image(sample['image'], self.mode)
        sample['label'] = F.to_pil_image(sample['label'], self.mode)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        if self.mode is not None:
            format_string += 'mode={0}'.format(self.mode)
        format_string += ')'
        return format_string


# 2D transforms
class Normalize(T.Normalize):

    def __init__(self, mean, std, inplace=False):
        super().__init__(mean, std, inplace)

    def forward(self, sample):
        sample['image'] = F.normalize(sample['image'], self.mean, self.std, self.inplace)
        return sample


class Resize(T.Resize):

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        super().__init__(size, interpolation)

    def forward(self, sample):
        sample['image'] = F.resize(sample['image'], self.size, InterpolationMode.BILINEAR)
        sample['label'] = F.resize(sample['label'], self.size, InterpolationMode.NEAREST)
        return sample


class CenterCrop(T.CenterCrop):

    def __init__(self, size):
        super().__init__(size)

    def forward(self, sample):
        sample['image'] = F.center_crop(sample['image'], self.size)
        sample['label'] = F.center_crop(sample['label'], self.size)
        return sample


class Pad(T.Pad):

    def __init__(self, padding, fill=0, padding_mode="constant"):
        super().__init__(padding, fill, padding_mode)

    def forward(self, sample):
        sample['label'] = F.pad(sample['image'], self.padding, self.fill, self.padding_mode)
        sample['label'] = F.pad(sample['label'], self.padding, self.fill, self.padding_mode)
        return sample


class RandomCrop(T.RandomCrop):

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)

    def forward(self, sample):
        img = sample['image']
        label = sample['label']
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        width, height = img.size

        # width, height = F._get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        sample['image'] = F.crop(img, i, j, h, w)
        sample['label'] = F.crop(label, i, j, h, w)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(self.size, self.padding)


class RandomFlip(torch.nn.Module):

    def __init__(self, p=0.5, direction='horizontal'):
        super().__init__()
        assert 0 <= p <= 1
        assert direction in ['horizontal', 'vertical', None], 'direction should be horizontal, vertical or None'
        self.p = p
        self.direction = direction

    def forward(self, sample):
        if torch.rand(1) < self.p:
            img, label = sample['image'], sample['label']
            if self.direction == 'horizontal':
                sample['image'] = F.hflip(img)
                sample['label'] = F.hflip(label)
            elif self.direction == 'vertical':
                sample['image'] = F.vflip(img)
                sample['label'] = F.vflip(label)
            else:
                if torch.rand(1) < 0.5:
                    sample['image'] = F.hflip(img)
                    sample['label'] = F.hflip(label)
                else:
                    sample['image'] = F.vflip(img)
                    sample['label'] = F.vflip(label)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomResizedCrop(T.RandomResizedCrop):

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=InterpolationMode.BILINEAR):
        super().__init__(size, scale, ratio, interpolation)


    def forward(self, sample):
        img, mask = sample['image'], sample['label']
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        sample['image'] = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        sample['label'] = F.resized_crop(mask, i, j, h, w, self.size, self.interpolation)
        return sample


class RandomRotation(T.RandomRotation):

    def __init__(
            self,
            degrees,
            interpolation=InterpolationMode.NEAREST,
            expand=False,
            center=None,
            fill=0,
            p=0.5,
            resample=None
    ):
        super().__init__(degrees, interpolation, expand, center, fill, resample)
        self.p = p

    def forward(self, sample):
        if torch.rand(1) > self.p:
            return sample
        img, label = sample['image'], sample['label']
        fill = self.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F._get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]
        label_fill = self.fill
        if isinstance(label, torch.Tensor):
            if isinstance(label_fill, (int, float)):
                label_fill = [float(label_fill)] * F._get_image_num_channels(label)
            else:
                label_fill = [float(f) for f in label_fill]
        angle = self.get_params(self.degrees)
        sample['image'] = F.rotate(img, angle, self.resample, self.expand, self.center, fill)
        sample['label'] = F.rotate(label, angle, self.resample, self.expand, self.center, label_fill)
        return sample


class RandomRotation90(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p=p

    def forward(self, sample):
        if torch.rand(1) < self.p:
            rot_times = random.randint(0, 4)
            sample['image'] = torch.rot90(sample['image'], rot_times, [1, 2])
            sample['label'] = torch.rot90(sample['label'], rot_times, [1, 2])
        return sample


class RandomErasing(T.RandomErasing):

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        super().__init__(p, scale, ratio, value, inplace)

    def forward(self, sample):
        if torch.rand(1) < self.p:
            img, label = sample['image'], sample['label']
            # cast self.value to script acceptable type
            if isinstance(self.value, (int, float)):
                value = [self.value, ]
            elif isinstance(self.value, str):
                value = None
            elif isinstance(self.value, tuple):
                value = list(self.value)
            else:
                value = self.value

            if value is not None and not (len(value) in (1, img.shape[-3])):
                raise ValueError(
                    "If value is a sequence, it should have either a single value or "
                    "{} (number of input channels)".format(img.shape[-3])
                )

            x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=value)
            sample['image'] = F.erase(img, x, y, h, w, v, self.inplace)
            sample['label'] = F.erase(label, x, y, h, w, v, self.inplace)
        return sample


class GaussianBlur(T.GaussianBlur):

    def __init__(self, kernel_size, sigma=(0.1, 2.0), p=0.5):
        super().__init__(kernel_size, sigma)
        self.p = p

    def forward(self, sample):
        if torch.rand(1) < self.p:
            sigma = self.get_params(self.sigma[0], self.sigma[1])
            sample['image'] = F.gaussian_blur(sample['image'], self.kernel_size, [sigma, sigma])
        return sample


class RandomGrayscale(T.RandomGrayscale):

    def __init__(self, p=0.1):
        super().__init__(p)

    def forward(self, sample):
        if torch.rand(1) < self.p:
            img = sample['image']
            if len(img.shape) == 4:
                img = img.permute(3, 0, 1, 2).contiguous()
                if img.size(1) == 1:
                    img = img.repeat(1, 3, 1, 1)
            num_output_channels = F._get_image_num_channels(img)
            img = F.rgb_to_grayscale(img, num_output_channels=num_output_channels)
            if len(img.shape) == 4:
                img = img.permute(1, 2, 3, 0).contiguous()
                img = img[0].unsqueeze(0)
            sample['image'] = img
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.p)


class ColorJitter(T.ColorJitter):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=1.):
        super().__init__(brightness, contrast, saturation, hue)
        self.p = p

    def forward(self, sample):
        if torch.rand(1) < self.p:
            img = sample['image']
            if len(img.size) == 4:
                img = img.permute(3, 0, 1, 2).contiguous()
            elif img.size[0] == 1:
                img = img.repeat(3, 1, 1)
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
                self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = F.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = F.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = F.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = F.adjust_hue(img, hue_factor)

            if len(img.size) == 4:
                img = img.permute(1, 2, 3, 0).contiguous()
                img = img[0].unsqueeze(0)
            sample['image'] = img
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string





