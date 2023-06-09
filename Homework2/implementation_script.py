import utils
import numpy as np

import itertools  # for Cartesian product


def nonMaxSuprression(img, d=5):
    """
    Given an image set all values to 0 that are not
    the maximum in its (2d+1,2d+1)-window

    Parameters
    ----------
    img : ndarray
        an image
    d : int
        for each pixel consider the surrounding (2d+1,2d+1)-window

    Returns
    -------
    result : ndarray

    """
    rows, cols = img.shape
    result = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            # no padding, shrink window at the edges
            low_y = max(0, i - d)
            low_x = max(0, j - d)
            high_y = min(rows, i + d)
            high_x = min(cols, j + d)

            window = img[low_y:high_y, low_x:high_x]

            max_val = window.max()
            if img[i, j] == max_val:
                result[i, j] = max_val

    return result


def calcBinaryMask(img, thresh=0.3):
    """
    Compute the gradient of an image and compute a binary mask
    based on the threshold. Corresponds to O^B in the slides.

    Parameters
    ----------
    img : ndarray
        an image
    thresh : float
        A threshold value. The default is 0.3.

    Returns
    -------
    binary : ndarray
        A binary image.

    """

    grad = utils.calcDirectionalGrad(img)
    threshold_value = np.max(grad) * thresh
    binary_mask = abs(grad) > threshold_value

    return binary_mask


def correlation(img, template):
    """
    Compute a correlation of gradients between an image and a template.

    Note:
    You should use the formula in the slides using the fourier transform.
    Then you are guaranteed to succeed.

    However, you can also compute the correlation directly.
    The resulting image must have high positive values at positions
    with high correlation.

    Parameters
    ----------
    img : ndarray
        a grayscale image
    template : ndarray
        a grayscale image of the template

    Returns
    -------
    ndarray
        an image containing the correlation between image and template gradients.
    """

    # (1.1) compute gradient of the image
    img_grad = utils.calcDirectionalGrad(img)
    img_r, img_c = img_grad.shape
    # (1.2) and the template
    template_grad = utils.calcDirectionalGrad(template)
    tmp_r, tmp_c = template_grad.shape

    # from 46th slide
    # (2) normalize template gradient
    template_sum = np.sum(abs(template_grad))
    O_i = template_grad / template_sum

    # (3.1) calculate binary mask and filter gradients
    O_b = calcBinaryMask(template)
    T = O_b * O_i

    # (3.2) coping template into larger scale frame and shifting
    T_padded = np.pad(T, ((0, img_r - tmp_r), (0, img_c - tmp_c)), constant_values=0)
    T_ready = utils.circularShift(T_padded, tmp_c // 2, tmp_r // 2)

    # Debug code - to see whether template is correctly shifted (spoiler - it is :)
    # utils.show(abs(T_ready))
    # exit()

    # (4) calculate correlation
    corr = np.real(
        np.fft.ifft2(
            np.fft.fft2(img_grad) * np.conjugate(np.fft.fft2(T_ready))
        )
    )

    return corr


def GeneralizedHoughTransform(img, template, angles, scales):
    """
    Compute the generalized hough transform. Given an image and a template.

    Parameters
    ----------
    img : ndarray
        A query image
    template : ndarray
        a template image
    angles : list[float]
        A list of angles provided in degrees
    scales : list[float]
        A list of scaling factors

    Returns
    -------
    hough_table : list[(correlation, angle, scaling)]
        The resulting hough table is a list of tuples.
        Each tuple contains the correlation and the corresponding combination
        of angle and scaling factors of the template.

        Note the order of these values.
    """
    result = []
    for (angle, scale) in itertools.product(angles, scales):
        modified_template = utils.rotateAndScale(template, angle, scale)
        corr = correlation(img, modified_template)
        result.append((corr, angle, scale))
    return result
