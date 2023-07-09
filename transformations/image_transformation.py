"""
This module defines classes of image transformations that can be applied to images.
Each also defines a unique identifier string and a parameterization object to ensure consistency between runs.
"""

from .transformation import *
import numpy as np
from typing import Type
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates
from imgaug import augmenters as iaa
from abc import ABC, abstractmethod

class ImgAugTransformation(Transformation, ABC):
    """
    Applies an imgaug transformation to an image
    https://imgaug.readthedocs.io/en/latest/source/overview
    """
    def __init__(self, deterministic: bool = True, seed: Optional[int] = None, override: Optional[int] = None):
        """
        Initialize a new instance of the transformation class
        :param deterministic: Whether to output the same result for every call
        :param seed: The seed to use for the random number generator
        :param override: Sets all values of the parameterization to the given value
        """
        super().__init__(seed, override)
        self.deterministic = deterministic
        self.aug = self.get_augmentation(self.param)

    @abstractmethod
    def get_augmentation(self, parameterization: Parameterization):
        """
        :param parameterization: The parameterization to use for the transformation
        :return: The imgaug augmentation object
        """
        raise NotImplementedError
    
    def transform(self, input_data: np.ndarray, parameterization: Parameterization):
        """
        :param input_data: The data to transform
        :param parameterization: The parameterization to use for the transformation
        :return: The transformed data
        """
        if input_data.ndim == 4:
            return self.aug(images=input_data)
        elif input_data.ndim == 3:
            return self.aug(image=input_data)
        else:
            raise ValueError("Input data must be 3 or 4 dimensional")
    
class SequentialImgAugTransformation(ImgAugTransformation):
    """
    Constructs a sequential imgaug transformation from a list of imgaug transformation classes
    """
    def __init__(self, transformation_types: List[Type[ImgAugTransformation]], deterministic: bool = True, seed: Optional[int] = None, override: Optional[int] = None):
        self.transformation_types = transformation_types
        self.transformations = [transformation_type(deterministic, seed, override) for transformation_type in transformation_types]
        super().__init__(deterministic, seed, override)

    @property
    def id(self):
        id = "Sequence"
        for transformation in self.transformations:
            id += f"+{transformation.id}"
        return id
    
    @property
    def param_ids(self):
        return [] # This itself is not parameterized, but the transformations it contains are

    def get_augmentation(self, parameterization: Parameterization):
        transformation_augmentations = [transformation.aug for transformation in self.transformations]
        return iaa.Sequential(transformation_augmentations)

class CutoutImgAugTransformation(ImgAugTransformation):
    """
    Applies the imgaug cutout transformation to an image
    https://imgaug.readthedocs.io/en/latest/source/overview/arithmetic.html#cutout
    """
    @property
    def id(self):
        return "ImgAugCutout"
    
    @property
    def param_ids(self):
        return [
            ("nb_iterations", IntRangeGenerator(1, 5)),
            ("size", FloatRangeGenerator(0.1, 0.5)),
            ("fill_mode", ChoiceGenerator(["constant", "gaussian"])),
            ("cval", IntRangeGenerator(0, 255)),
            ("fill_per_channel", BooleanGenerator()),
        ]
    
    def get_augmentation(self, parameterization: Parameterization):
        return iaa.Cutout(
            seed=self.seed if self.deterministic else None,
            nb_iterations=parameterization["nb_iterations"],
            size=parameterization["size"],
            fill_mode=parameterization["fill_mode"],
            cval=parameterization["cval"]
        )
    
class MotionBlurImgAugTransformation(ImgAugTransformation):
    """
    Applies the imgaug motion blur transformation to an image
    https://imgaug.readthedocs.io/en/latest/source/overview/blur.html#motionblur
    """
    @property
    def id(self):
        return "ImgAugMotionBlur"
    
    @property
    def param_ids(self):
        return [
            ("k", IntRangeGenerator(5, 20)),
            ("angle", FloatRangeGenerator(0, 360)),
        ]
    
    def get_augmentation(self, parameterization: Parameterization):
        return iaa.MotionBlur(
            seed=self.seed if self.deterministic else None,
            k=parameterization["k"],
            angle=parameterization["angle"]
        )
    
class DropoutImgAugTransformation(ImgAugTransformation):
    """
    Applies the imgaug dropout transformation to an image
    https://imgaug.readthedocs.io/en/latest/source/overview/arithmetic.html#dropout
    """
    @property
    def id(self):
        return "ImgAugDropout"
    
    @property
    def param_ids(self):
        return [
            ("p", FloatRangeGenerator(0.05, 0.5)),
            ("per_channel", ChoiceGenerator([True, False])),
        ]
    
    def get_augmentation(self, parameterization: Parameterization):
        return iaa.Dropout(
            seed=self.seed if self.deterministic else None,
            p=parameterization["p"],
            per_channel=parameterization["per_channel"]
        )

class ElasticTransformationImgAugTransformation(ImgAugTransformation):
    """
    Applies the imgaug elastic transformation to an image
    https://imgaug.readthedocs.io/en/latest/source/overview/geometric.html#elastictransformation
    """
    @property
    def id(self):
        return "ImgAugElasticTransformation"
    
    @property
    def param_ids(self):
        return [
            ("alpha", FloatRangeGenerator(5, 200)),
            ("sigma", FloatRangeGenerator(0.5, 10)),
        ]
    
    def get_augmentation(self, parameterization: Parameterization):
        return iaa.ElasticTransformation(
            seed=self.seed if self.deterministic else None,
            alpha=parameterization["alpha"],
            sigma=parameterization["sigma"]
        )

class JigsawImgAugTransformation(ImgAugTransformation):
    """
    Applies the imgaug jigsaw transformation to an image
    https://imgaug.readthedocs.io/en/latest/source/overview/geometric.html#jigsaw
    """
    @property
    def id(self):
        return "ImgAugJigsaw"
    
    @property
    def param_ids(self):
        return [
            ("nb_rows", IntRangeGenerator(2, 10)),
            ("nb_cols", IntRangeGenerator(2, 10)),
        ]
    
    def get_augmentation(self, parameterization: Parameterization):
        return iaa.Jigsaw(
            seed=self.seed if self.deterministic else None,
            nb_rows=parameterization["nb_rows"],
            nb_cols=parameterization["nb_cols"]
        )

class AffinePolarWarpingImgAugTransformation(ImgAugTransformation):
    """
    Applies the imgaug affine polar warping transformation to an image
    https://imgaug.readthedocs.io/en/latest/source/overview/geometric.html#affinepolarwarping
    """
    @property
    def id(self):
        return "ImgAugAffinePolarWarping"
    
    @property
    def param_ids(self):
        return [
            ("x", FloatRangeGenerator(-0.4, 0.4)),
            ("y", FloatRangeGenerator(-0.4, 0.4)),
        ]
    
    def get_augmentation(self, parameterization: Parameterization):
        return iaa.WithPolarWarping(
            iaa.Affine(
                seed=self.seed if self.deterministic else None,
                translate_percent={"x": parameterization["x"], "y": parameterization["y"]},
            ),
            seed=self.seed if self.deterministic else None,
        )

class CannyImgAugTransformation(ImgAugTransformation):
    """
    Applies the imgaug canny transformation to an image
    https://imgaug.readthedocs.io/en/latest/source/overview/edges.html#canny
    """
    @property
    def id(self):
        return "ImgAugCanny"
    
    @property
    def param_ids(self):
        return [
            ("alpha", FloatRangeGenerator(0.8, 1)),
            ("threshold_min", IntRangeGenerator(100 - 40, 100 + 40)),
            ("threshold_max", IntRangeGenerator(200 - 40, 200 + 40)),
        ]
    
    def get_augmentation(self, parameterization: Parameterization):
        return iaa.Canny(
            seed=self.seed if self.deterministic else None,
            alpha=parameterization["alpha"],
            hysteresis_thresholds=(parameterization["threshold_min"], parameterization["threshold_max"]),
        )

imgaug_transformations = [
    CutoutImgAugTransformation,
    MotionBlurImgAugTransformation,
    DropoutImgAugTransformation,
    ElasticTransformationImgAugTransformation,
    JigsawImgAugTransformation,
    AffinePolarWarpingImgAugTransformation,
    CannyImgAugTransformation,
]

imgaug_transformation_map = {
    "ImgAugCutout": CutoutImgAugTransformation,
    "ImgAugMotionBlur": MotionBlurImgAugTransformation,
    "ImgAugDropout": DropoutImgAugTransformation,
    "ImgAugElasticTransformation": ElasticTransformationImgAugTransformation,
    "ImgAugJigsaw": JigsawImgAugTransformation,
    "ImgAugAffinePolarWarping": AffinePolarWarpingImgAugTransformation,
    "ImgAugCanny": CannyImgAugTransformation,
}

class GaussianBlurTransformation(Transformation):
    """
    A transformation that applies a Gaussian blur to an image
    Parameterized by the standard deviation of the Gaussian kernel and the kernel size
    """
    def to_odd(self, x, max_value=13):
        return int(x * (max_value // 2 + 1)) * 2 + 1

    @property
    def id(self):
        return "GaussianBlur"
    
    @property
    def param_ids(self):
        return [
            ("sigma", FloatRangeGenerator(2, 6)),
            ("kernel_size", ApplicationGenerator(self.to_odd))  # Generate a random odd number between 1 and 13
        ]
    
    def transform(self, input_data: np.ndarray, parameterization: Parameterization):
        """
        :param input_data: The data to transform
        :param parameterization: The parameterization to use for the transformation
        :return: The transformed data
        """
        return cv2.GaussianBlur(
            input_data,
            (parameterization["kernel_size"], parameterization["kernel_size"]),
            parameterization["sigma"]
        )
    

class MedianBlurTransformation(Transformation):
    """
    A transformation that applies a median blur to an image
    Parameterized by the kernel size
    """
    def to_odd(self, x, max_value=7):
        return int(x * (max_value // 2 + 1)) * 2 + 1
    
    @property
    def id(self):
        return "MedianBlur"
    
    @property
    def param_ids(self):
        return [
            ("kernel_size", ApplicationGenerator(self.to_odd))  # Generate a random odd number between 1 and 7
        ]
    
    def transform(self, input_data: np.ndarray, parameterization: Parameterization):
        """
        :param input_data: The data to transform
        :param parameterization: The parameterization to use for the transformation
        :return: The transformed data
        """
        return cv2.medianBlur(input_data, parameterization["kernel_size"])
    

class AddNoiseTransformation(Transformation):
    """
    A transformation that adds Gaussian noise to an image
    Parameterized by the mean and standard deviation of the Gaussian noise
    """
    @property
    def id(self):
        return "AddNoise"

    @property
    def param_ids(self):
        return [
            ("mean", FloatRangeGenerator(10, 50)),
            ("std_dev", FloatRangeGenerator(10, 50))
        ]

    def transform(self, input_data: np.ndarray, parameterization: Parameterization):
        """
        :param input_data: The data to transform
        :param parameterization: The parameterization to use for the transformation
        :return: The transformed data
        """
        noise = np.random.normal(parameterization["mean"], parameterization["std_dev"], input_data.shape).astype(np.uint8)
        
        # Adding the noise to the input data
        noised_data = cv2.add(input_data, noise)
        return noised_data


class ErosionTransformation(Transformation):
    """
    A transformation that applies erosion to an image.
    Parameterized by the size of the erosion kernel.
    """
    @property
    def id(self):
        return "Erosion"
    
    @property
    def param_ids(self):
        return [
            ("kernel_size", IntRangeGenerator(1, 10))
        ]
    
    def transform(self, input_data: np.ndarray, parameterization: Parameterization):
        """
        :param input_data: The data to transform
        :param parameterization: The parameterization to use for the transformation
        :return: The transformed data
        """
        kernel = np.ones((parameterization["kernel_size"], parameterization["kernel_size"]),np.uint8)
        return cv2.erode(input_data, kernel, iterations = 1)
    

class DilationTransformation(Transformation):
    """
    A transformation that applies dilation to an image.
    Parameterized by the size of the dilation kernel.
    """
    @property
    def id(self):
        return "Dilation"
    
    @property
    def param_ids(self):
        return [
            ("kernel_size", IntRangeGenerator(1, 10))
        ]
    
    def transform(self, input_data: np.ndarray, parameterization: Parameterization):
        """
        :param input_data: The data to transform
        :param parameterization: The parameterization to use for the transformation
        :return: The transformed data
        """
        kernel = np.ones((parameterization["kernel_size"], parameterization["kernel_size"]),np.uint8)
        return cv2.dilate(input_data, kernel, iterations = 1)
    

class ThresholdingTransformation(Transformation):
    """
    A transformation that applies thresholding to an image
    Parameterized by the threshold value and the max value
    """
    @property
    def id(self):
        return "Thresholding"
    
    @property
    def param_ids(self):
        return [
            ("threshold", FloatRangeGenerator(50, 255)),
            ("max_val", FloatRangeGenerator(50, 255))
        ]
    
    def transform(self, input_data: np.ndarray, parameterization: Parameterization):
        """
        :param input_data: The data to transform
        :param parameterization: The parameterization to use for the transformation
        :return: The transformed data
        """
        _, thresh = cv2.threshold(input_data, parameterization["threshold"], parameterization["max_val"], cv2.THRESH_BINARY)
        return thresh


class PerspectiveTransformation(Transformation):
    """
    A transformation that applies a perspective transform to an image
    Parameterized by four points in the image that form a quadrilateral
    """
    @property
    def id(self):
        return "PerspectiveTransform"
    
    @property
    def param_ids(self):
        return [
            ("top_left", PointGenerator([(-0.2, 0.4), (-0.2, 0.4)])),
            ("top_right", PointGenerator([(0.6, 1.2), (-0.2, 0.4)])),
            ("bottom_left", PointGenerator([(-0.2, 0.4), (0.6, 1.2)])),
            ("bottom_right", PointGenerator([(0.6, 1.2), (0.6, 1.2)]))
        ]
    
    def transform(self, input_data: np.ndarray, parameterization: Parameterization):
        """
        :param input_data: The data to transform
        :param parameterization: The parameterization to use for the transformation
        :return: The transformed data
        """
        rows, cols, _ = input_data.shape

        top_left = (parameterization["top_left"][0] * cols, parameterization["top_left"][1] * rows)
        top_right = (parameterization["top_right"][0] * cols, parameterization["top_right"][1] * rows)
        bottom_left = (parameterization["bottom_left"][0] * cols, parameterization["bottom_left"][1] * rows)
        bottom_right = (parameterization["bottom_right"][0] * cols, parameterization["bottom_right"][1] * rows)

        pts1 = np.float32([[top_left[0], top_left[0]],
                           [top_right[0], top_right[1]],
                            [bottom_left[0], bottom_left[1]],
                            [bottom_right[0], bottom_right[1]]])

        pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        # cv2.circle(input_data, (int(cols/2), int(rows/2)), 10, (255, 0, 0), -1)
        # cv2.circle(input_data, (int(top_left[0]), int(top_left[1])), 10, (255, 0, 0), -1)
        # cv2.circle(input_data, (int(top_right[0]), int(top_right[1])), 10, (255, 0, 0), -1)
        # cv2.circle(input_data, (int(bottom_left[0]), int(bottom_left[1])), 10, (255, 0, 0), -1)
        # cv2.circle(input_data, (int(bottom_right[0]), int(bottom_right[1])), 10, (255, 0, 0), -1)

        img = cv2.warpPerspective(input_data, matrix, (cols, rows))
        
        return img


class ElasticTransformation(Transformation):
    """
    A transformation that applies an elastic deformation to an image
    Parameterized by the alpha (scaling factor) and sigma (elasticity coefficient)
    """
    @property
    def id(self):
        return "ElasticTransform"
    
    @property
    def param_ids(self):
        return [
            ("alpha", FloatRangeGenerator(0.5, 3)),
            ("sigma", FloatRangeGenerator(0.1, 0.5))
        ]
    
    def transform(self, input_data: np.ndarray, parameterization: Parameterization):
        """
        :param input_data: The data to transform
        :param parameterization: The parameterization to use for the transformation
        :return: The transformed data
        """
        raise "Broken"
        alpha = parameterization["alpha"]
        sigma = parameterization["sigma"]

        random_state = np.random.RandomState(None)

        shape = input_data.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z+dz, (-1, 1))

        distored_image = map_coordinates(input_data, indices, order=1, mode='reflect')
        return distored_image.reshape(input_data.shape)


class FisheyeDistortionTransformation(Transformation):
    """
    A transformation that applies a fisheye distortion to an image.
    Parameterized by the scale of the distortion.

    Stolen from https://github.com/Gil-Mor/iFish/blob/master/fish.py
    """
    def __init__(self, max_radius=1.0, **kwargs):
        self.max = max_radius
        self.transformation_hash = -1
        self.transformation_map = None
        super().__init__(**kwargs)

    @property
    def id(self):
        return "FisheyeDistortion"
    
    @property
    def param_ids(self):
        return [
            ("scale", FloatRangeGenerator(0.1, self.max))  # Above 1 produces strange effects, but we want very strange transformations
        ]
    
    def get_fish_xn_yn(self, source_x, source_y, radius, distortion):
        """
        Get normalized x, y pixel coordinates from the original image and return normalized 
        x, y pixel coordinates in the destination fished image.
        :param distortion: Amount in which to move pixels from/to center.
        As distortion grows, pixels will be moved further from the center, and vice versa.
        """

        if 1 - distortion*(radius**2) == 0:
            return source_x, source_y

        return source_x / (1 - (distortion*(radius**2))), source_y / (1 - (distortion*(radius**2)))
    
    def generate_pixel_map(self, width, height, distortion_coefficient):
        """
        Generate pixel map for distortion
        :return: Pixel map as a 2D array of (x, y) tuples
        """
        pixel_map = np.zeros((width, height, 2), dtype=int)

        # easier calcultion if we traverse x, y in dst image
        for x in range(width):
            for y in range(height):

                # normalize x and y to be in interval of [-1, 1]
                xnd, ynd = float((2*x - width)/width), float((2*y - height)/height)

                # get xn and yn distance from normalized center
                rd = (xnd**2 + ynd**2)**0.5

                # new normalized pixel coordinates
                xdu, ydu = self.get_fish_xn_yn(xnd, ynd, rd, distortion_coefficient)

                # convert the normalized distorted xdn and ydn back to image pixels
                xu, yu = int(((xdu + 1)*width)/2), int(((ydu + 1)*height)/2)

                # store the original coordinates in the pixel map
                pixel_map[x, y] = [xu, yu]

        return pixel_map
    
    def distort_image(self, img, width, height, transformation_map):
        """
        Distort the image using the pre-computed pixel map
        :param img: Image to distort
        :return: Distorted image
        """
        # Check and convert image to correct color channels
        # If input image is only BW or RGB convert it to RGBA
        # So that output 'frame' can be transparent.
        if len(img.shape) == 2:
            # Duplicate the one BW channel twice to create Black and White
            # RGB image (For each pixel, the 3 channels have the same value)
            bw_channel = np.copy(img)
            img = np.dstack((img, bw_channel))
            img = np.dstack((img, bw_channel))
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = np.dstack((img, np.full((width, height), 255)))

        # Prepare array for dst image
        dstimg = np.zeros_like(img)

        # For every pixel in the destination image, copy from the corresponding
        # pixel in the source image, as indicated by the pixel map
        for x in range(width):
            for y in range(height):
                xu, yu = transformation_map[x, y]

                # if new pixel is in bounds copy from source pixel to destination pixel
                if 0 <= xu < width and 0 <= yu < height:
                    dstimg[x][y] = img[xu][yu]

        # Convert back to RGB
        if len(img.shape) == 3 and img.shape[2] == 4:
            dstimg = dstimg[:, :, :3]

        return dstimg.astype(np.uint8)
    
    def transform(self, input_data: np.ndarray, parameterization: Parameterization):
        """
        :param input_data: The data to transform
        :param parameterization: The parameterization to use for the transformation
        :return: The transformed data
        """
        w, h, _ = input_data.shape
        distortion = parameterization["scale"]
        transformation_hash = (w, h, distortion)
        if transformation_hash != self.transformation_hash or self.transformation_map is None:
            # Then we need to recompute the pixel map
            print(f"Recomputing pixel map... (old hash: {self.transformation_hash}, new hash: {transformation_hash})")
            self.transformation_map = self.generate_pixel_map(w, h, distortion)
            self.transformation_hash = transformation_hash
        return self.distort_image(input_data, w, h, self.transformation_map)


transformation_classes = [
    GaussianBlurTransformation,
    MedianBlurTransformation,
    AddNoiseTransformation,
    ErosionTransformation,
    DilationTransformation,
    # ThresholdingTransformation,
    PerspectiveTransformation,
    # ElasticTransformation
    # FisheyeDistortionTransformation
    *imgaug_transformations
]

transformation_name_map = {
    "gaussian": GaussianBlurTransformation,
    "median": MedianBlurTransformation,
    "noise": AddNoiseTransformation,
    "erosion": ErosionTransformation,
    "dilation": DilationTransformation,
    "perspective": PerspectiveTransformation,
    # "fisheye": FisheyeDistortionTransformation  # Very slow. Needs a smarter implementation
    **imgaug_transformation_map
}