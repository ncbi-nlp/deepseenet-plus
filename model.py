"""
Usage:
    model.py -i --model_folder=<str> --image_folder=<str> --input_file=<str> --output_file=<str>
    model.py -t --model_path=<str> --image_folder=<str> --input_file=<str> --risk_factor=<str>

Options:
    -h --help              Show this help message
    -i --inference         Evaluate the model
    -t --train             Continue training the model
    --model_folder=<str>   Path where the model is saved
    --model_path=<str>     File path of trained model
    --image_folder=<str>   Path where the images are saved
    --input_file=<str>     Input file of patients, image path names
    --output_file=<str>    Output file name
    --risk_factor=<str>    Risk factor to train: drusen, pigment, amd
"""
# RUN INFERENCE:
#   python model.py -i --model_folder=../models --image_folder=../example_images --input_file=../ex_input_file.csv --output_file=ex_output.csv
# CONTINUE TRAINING:
#   python model.py -t --model_path=../models/amd.h5 --image_folder=../example_images --input_file=../ex_input_file.csv --risk_factor=amd 

'''
pull models and utils from dsnplus folder

- data_generator.py

these load save .h5 model files for each risk factor to do inference (NOT train)
- model_amd.py
- model_drusen.py
- model_pigment.py

- model_risk_factor.py -- idk what this does. looks like it just makes a wholenew model? used in examples/train.py...
- model_simplified.py

- utils.py

In DeepSeeNet, examples folder has scripts to do inference OR train. These will be combined in model.py.
`model.py -inference` should produce scores for each risk factor AND output a final severity score.

- run_inference_risk_factors(): inference on each risk factor
- predict_final_score(): combine for final score. 

    - DSN's `predict_simplified_score.py` creates instance of DeepSeeNetSimplifiedScore, which creates instances of DeepSeeNetDrusen, DeepSeeNetPigment, DeepSeeNetAdvancedAMD.
        - In run_inference_risk_factors(), I can mimic DeepSeeNetSimplifiedScore, but also print out/save intermediate risk factor predictions
            - this is prob better because then I can do continual training
        - OR use the 2 functions I already have: run_inference_risk_factors() run on L/R/risk_factor and save each, then call predict_final_score()

- DLN doesn't have patid? is just consecutive, so they just append predictions. 
    DSN's predict_simplified_score is for one patient (one (left_eye, right_eye) pair) at a time

'''

import multiprocessing
import sys 
import os 
import numpy as np
import pandas as pd
from docopt import docopt
# from keras.models import load_model
from tensorflow.keras.models import load_model
from dsnplus.utils import get_simplified_score, preprocess_image
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers, callbacks
from dsnplus.data_generator import DataGenerator
from tensorflow.keras.optimizers import Adam


def run_inference_final_score(model_folder, image_folder, input_file, output_file, risk_factors=['drusen', 'pigment', 'amd']):
    
    #?? is there a standard for what input_file should contain? how to get L, R for same patient
    
    # scores = pd.DataFrame(columns=['PATID']+risk_factors) # patid, drusen, pigment, amd => #, (left score, right score), (left score, right score), (left score, right score)

    df = pd.read_csv(input_file).groupby('PATID')

    # Initialize with unique PATIDs
    scores = pd.DataFrame(index=df.groups.keys()) #columns=[f"{risk_factor}_left" for risk_factor in risk_factors] + [f"{risk_factor}_right" for risk_factor in risk_factors]
    

    for risk_factor in risk_factors:  # basically "for each model"
        model_path = os.path.join(model_folder, risk_factor+'.h5')
        model = load_model(model_path)

        scores[f"{risk_factor}_left"] = np.nan  # initialize columns for left and right eye
        scores[f"{risk_factor}_right"] = np.nan

        print(f"\nRunning inference for {risk_factor}")
        print(f"Model path: {model_path}")
        print("scores df:\n", scores)

        for patid, data in df:
            left_eye = data[data['EYE'] == 'L']
            right_eye = data[data['EYE'] == 'R']
            if left_eye.empty or right_eye.empty:
                print(f"Skipping {patid}: Missing left or right eye image")
                continue

            x_left = os.path.join(image_folder, left_eye['pathname'].values[0])
            x_right = os.path.join(image_folder, right_eye['pathname'].values[0])
            print("x_left\t", x_left, "| x_right\t", x_right)

            x_left = preprocess_image(x_left)
            left_pred = model.predict(x_left)
            left_score = np.argmax(left_pred, axis=1)[0]
            right_score = np.argmax(model.predict(preprocess_image(x_right)), axis=1)[0]
            print("left_score\t", left_score, "| right_score\t", right_score)

            scores.loc[patid, f"{risk_factor}_left"] = left_score
            scores.loc[patid, f"{risk_factor}_right"] = right_score

    print("Scores df after all risk factors:\n", scores, "\n")
    
    final_scores = {} # patid => final score
    for patid, data in scores.iterrows():
        simplified_score = get_simplified_score(data)
        final_scores[str(patid)] = simplified_score
        print(f"Final score for {patid}: {simplified_score}")
    
    print("Final scores df:\n", final_scores)
    final_scores_df = pd.DataFrame(list(final_scores.items()), columns=['PATID', 'simplified_score_PRED'])
    final_scores_df.to_csv(output_file, index=False)

# df has predictions for each risk factor
    #?? also save? in case only want risk factor, or error in final score 
    # df.to_csv(output_file, index=False)


# class DataLoader(object):                 // a class seems unecessary. used prep_instances instead
#     def __init__(self):
#         log.info('data loader created')
#     def load_instances(self, file_path):
#         df = pd.read_csv(file_path)
#         df = df.sample(frac=1.0, replace=False, random_state=seed)
#         return df

def train(model_path, image_folder, input_file, risk_factor, batch_size=2): #!! 16
    ''' Continue training model.
    input_file(str): training data file '''

    train_data, valid_data, n_classes = prep_instances(input_file, image_folder)

    cpu_count = multiprocessing.cpu_count()
    workers = max(int(cpu_count / 3), 1)

    model = load_model(model_path)
    earlystop = callbacks.EarlyStopping(monitor='val_loss', min_delta=K.epsilon(), patience=5, verbose=1, mode="min")
    best_checkpoint = callbacks.ModelCheckpoint(
          str(model_path), save_best_only=True, save_weights_only=False,
          monitor='val_loss', mode='min', verbose=1)
    # optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, clipnorm=0.001, epsilon=0.001, amsgrad=False)
    
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
    Read the dataset.
    Args:
        train_file(str): File path of training dataset
        val_size(float): Between 0.0 and 1.0; proportion of the dataset to include in the validation split.
        shuffle(bool): Whether or not to shuffle the data before splitting.
        parent(str): Data directory
    Returns:
        !!
        int: Number of classes
    """
    df = pd.read_csv(train_file)
    print("df before mod\n", df)
    df['pathname'] = df['pathname'].apply(lambda x: os.path.join(image_folder, x))
    print("df after mod\n", df)
    n_classes = len(df.iloc[:, 1].unique())

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    split_size = int(len(df) * val_size)
    train_data = df.iloc[:split_size]#.values.tolist()
    valid_data = df.iloc[split_size:]#.values.tolist()

    return train_data, valid_data, n_classes


if __name__ == "__main__":
    print("running main")
    
    try:
        argv = docopt(__doc__, sys.argv[1:])
        print(argv)  # Check if arguments are parsed correctly
    except SystemExit as e:
        print(f"Error parsing arguments: {e}")
        sys.exit(1)

    model_folder = argv['--model_folder']
    image_folder = argv['--image_folder']
    input_file = argv['--input_file']

    if argv['--inference']:  # Check both short and long forms
        output_file = argv['--output_file']
        print(f"Running inference with output file: {output_file}")
        run_inference_final_score(model_folder, image_folder, input_file, output_file)

    elif argv['--train']:
        print("Continuing training")
        model_path = argv['--model_path']
        risk_factor = argv['--risk_factor']
        train(model_path, image_folder, input_file, risk_factor)
