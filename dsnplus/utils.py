
import numpy as np
import keras.utils as image
import keras.applications.inception_v3 as inception_v3

def crop2square(img):
    short_side = min(img.size)
    horizontal_padding = (img.size[0] - short_side) / 2
    vertical_padding = (img.size[1] - short_side) / 2
    return img.crop(
    (horizontal_padding, vertical_padding,
        img.size[0] - horizontal_padding, img.size[1] - vertical_padding))

def preprocess_image(image_path):
    img = crop2square(image.load_img(image_path)).resize((512, 512))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = inception_v3.preprocess_input(x)
    return x

def get_simplified_score(scores):
    """
    Get AREDS simplified severity score from drusen size, pigmentary abnormality, and advanced AMD.

    Args:
        scores: a dict of individual risk factors

    Returns:
        a score of 0-5
    """
    def has_adv_amd(score):
        return True if score == 1 else False

    def has_pigment(score):
        return True if score == 1 else False

    def has_large_drusen(score):
        return True if score == 2 else False

    def has_intermediate_drusen(score):
        return True if score == 1 else False

    score = 0
    if has_adv_amd(scores['amd_left']):
        score += 5
    if has_adv_amd(scores['amd_right']):
        score += 5
    if has_pigment(scores['pigment_left']):
        score += 1
    if has_pigment(scores['pigment_right']):
        score += 1
    if has_large_drusen(scores['drusen_left']):
        score += 1
    if has_large_drusen(scores['drusen_right']):
        score += 1
    if has_intermediate_drusen(scores['drusen_left']) \
            and has_intermediate_drusen(scores['drusen_right']):
        score += 1

    return 5 if score >= 5 else score