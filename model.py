"""
Usage:
    model.py -i --model_folder=<str> --image_folder=<str> --input_file=<str> --output_file=<str>
    model.py -t --model_path=<str> --image_folder=<str> --input_file=<str> --risk_factor=<str>

Options:
    -h --help              Show this help message
    -i --inference         Evaluate the model
    -t --train             Continue training the model
    --model_folder=<str>   Path where the model is saved
    --model_path=<str>     File path of model
    --image_folder=<str>   Path where the images are saved
    --input_file=<str>     Input data file of patients, image paths, and labels (risk factors) 
    --output_file=<str>    Output file name
    --risk_factor=<str>    Risk factor to train: drusen, pigment, amd
"""

import multiprocessing
import sys 
import os 
import numpy as np
import pandas as pd
from docopt import docopt
from tensorflow.keras.models import load_model
from dsnplus.utils import get_simplified_score, preprocess_image
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers, callbacks
from dsnplus.data_generator import DataGenerator
from tensorflow.keras.optimizers import Adam


def run_inference_final_score(model_folder, image_folder, input_file, output_file, risk_factors=['drusen', 'pigment', 'amd']):
    df = pd.read_csv(input_file).groupby('PATID')

    scores = pd.DataFrame(index=df.groups.keys()) # Initialize with unique PATIDs

    for risk_factor in risk_factors:  # For each model
        model_path = os.path.join(model_folder, risk_factor+'.h5')
        model = load_model(model_path)

        print(f"\nRunning inference for {risk_factor}")

        for patid, data in df:
            left_eye = data[data['EYE'] == 'L']
            right_eye = data[data['EYE'] == 'R']
            if left_eye.empty or right_eye.empty:
                print(f"Skipping {patid}: Missing left or right eye image")
                continue

            x_left = os.path.join(image_folder, left_eye['pathname'].values[0])
            x_right = os.path.join(image_folder, right_eye['pathname'].values[0])

            x_left = preprocess_image(x_left)
            left_pred = model.predict(x_left)
            left_score = np.argmax(left_pred, axis=1)[0]
            right_score = np.argmax(model.predict(preprocess_image(x_right)), axis=1)[0]

            scores.loc[patid, f"{risk_factor}_L"] = left_score
            scores.loc[patid, f"{risk_factor}_R"] = right_score
    
    final_scores = {} # patid => final score
    for patid, data in scores.iterrows():
        simplified_score = get_simplified_score(data)
        final_scores[str(patid)] = simplified_score
        final_scores[str(patid)] = {'simplified_score_PRED': simplified_score}

        for risk_factor in risk_factors:
            final_scores[str(patid)][f"{risk_factor}_L_PRED"] = data[f"{risk_factor}_L"]
            final_scores[str(patid)][f"{risk_factor}_R_PRED"] = data[f"{risk_factor}_R"]
    
    final_scores_df = pd.DataFrame.from_dict(final_scores, orient='index').reset_index().rename(columns={'index': 'PATID'})
    final_scores_df.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")

def train(model_path, image_folder, input_file, risk_factor, batch_size=16):
    ''' 
    Load and continue training model
    '''

    train_data, valid_data, n_classes = prep_instances(input_file, image_folder)

    cpu_count = multiprocessing.cpu_count()
    workers = max(int(cpu_count / 3), 1)

    model = load_model(model_path)
    earlystop = callbacks.EarlyStopping(monitor='val_loss', min_delta=K.epsilon(), patience=5, verbose=1, mode="min")
    best_checkpoint = callbacks.ModelCheckpoint(
          str(model_path), save_best_only=True, save_weights_only=False,
          monitor='val_loss', mode='min', verbose=1)
    
    train_generator = DataGenerator(train_data, n_classes, batch_size, risk_factor, shuffle=True)
    valid_generator = DataGenerator(valid_data, n_classes, batch_size, risk_factor, shuffle=False)
    train_chunk_number = train_generator.get_epoch_num()

    print(f"Info: Training model {model_path}")
    model.fit(train_generator, 
       use_multiprocessing=True, 
       workers=workers, 
       steps_per_epoch=train_chunk_number,
       callbacks=[earlystop, best_checkpoint], 
       epochs=50, 
       validation_data=valid_generator,
       validation_steps=valid_generator.get_epoch_num(), 
       verbose=1,)

def prep_instances(train_file, image_folder, val_size=0.8, shuffle=True):
    """
    Read the dataset, split into training and validation sets, and find number of classes.
    Args:
        train_file(str): File path of training dataset
        val_size(float): Between 0.0 and 1.0; proportion of the dataset to include in the validation split.
        shuffle(bool): Whether or not to shuffle the data before splitting.
        parent(str): Data directory
    Returns:
        DataFrame: Training, validation data
        int: Number of classes
    """
    df = pd.read_csv(train_file)
    df['pathname'] = df['pathname'].apply(lambda x: os.path.join(image_folder, x))
    n_classes = len(df.iloc[:, 1].unique())

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    split_size = int(len(df) * val_size)
    train_data = df.iloc[:split_size]#.values.tolist()
    valid_data = df.iloc[split_size:]#.values.tolist()

    return train_data, valid_data, n_classes


if __name__ == "__main__":
    argv = docopt(__doc__, sys.argv[1:])

    model_folder = argv['--model_folder']
    image_folder = argv['--image_folder']
    input_file = argv['--input_file']

    if argv['--inference']: 
        output_file = argv['--output_file']
        print(f"Running inference with output file: {output_file}")
        run_inference_final_score(model_folder, image_folder, input_file, output_file)

    elif argv['--train']:
        print("Continuing training")
        model_path = argv['--model_path']
        risk_factor = argv['--risk_factor']
        train(model_path, image_folder, input_file, risk_factor)
