"""
This module defines classes of image transformations that can be applied to images.
Each also defines a unique identifier string and a parameterization object to ensure consistency between runs.
"""

from .transformation import *
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates

class GaussianBlurTransformation(Transformation):
    """
    A transformation that applies a Gaussian blur to an image
    Parameterized by the standard deviation of the Gaussian kernel and the kernel size
    """
    def to_odd(self, x, max_value=13):
        return int(x * (max_value // 2 + 1)) * 2 + 1

    @property
    def id():
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


    def fish(self, img, distortion_coefficient):
        """
        :type img: numpy.ndarray
        :param distortion_coefficient: The amount of distortion to apply.
        :return: numpy.ndarray - the image with applied effect.
        """

        # If input image is only BW or RGB convert it to RGBA
        # So that output 'frame' can be transparent.
        w, h = img.shape[0], img.shape[1]
        if len(img.shape) == 2:
            # Duplicate the one BW channel twice to create Black and White
            # RGB image (For each pixel, the 3 channels have the same value)
            bw_channel = np.copy(img)
            img = np.dstack((img, bw_channel))
            img = np.dstack((img, bw_channel))
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = np.dstack((img, np.full((w, h), 255)))

        # prepare array for dst image
        dstimg = np.zeros_like(img)

        # floats for calcultions
        w, h = float(w), float(h)

        # easier calcultion if we traverse x, y in dst image
        for x in range(len(dstimg)):
            for y in range(len(dstimg[x])):

                # normalize x and y to be in interval of [-1, 1]
                xnd, ynd = float((2*x - w)/w), float((2*y - h)/h)

                # get xn and yn distance from normalized center
                rd = (xnd**2 + ynd**2)**0.5

                # new normalized pixel coordinates
                xdu, ydu = self.get_fish_xn_yn(xnd, ynd, rd, distortion_coefficient)

                # convert the normalized distorted xdn and ydn back to image pixels
                xu, yu = int(((xdu + 1)*w)/2), int(((ydu + 1)*h)/2)

                # if new pixel is in bounds copy from source pixel to destination pixel
                if 0 <= xu and xu < img.shape[0] and 0 <= yu and yu < img.shape[1]:
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
        distortion = parameterization["scale"]
        return self.fish(input_data, distortion)


transformation_classes = [
    GaussianBlurTransformation,
    MedianBlurTransformation,
    AddNoiseTransformation,
    ErosionTransformation,
    DilationTransformation,
    # ThresholdingTransformation,
    PerspectiveTransformation,
    # ElasticTransformation
    FisheyeDistortionTransformation
]