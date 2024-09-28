''' These classes pre-load the model, so if we process '''

def DSNPlusDrusen(model='areds'):
    """
    Load and instantiate drusen model

    Args:
        model: One of 'areds1' (pre-training on AREDS1),
              or the path to the model file to be loaded.

    Returns:
        A Keras model instance.
    """
    if model == 'areds':
        model = get_file(
            'drusen_model.h5',
            DRUSEN_PATH,
            cache_dir='models',
            file_hash=DRUSEN_MD5
        )
    logging.info('Loading the model: %s', model)
    return models.load_model(model)