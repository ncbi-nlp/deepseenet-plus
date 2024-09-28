"""
Usage:
    model_classify.py -i --model_folder=<str> --image_folder=<str> --input_file=<str> --output_file=<str>
    model_classify.py -t --model_folder=<str> --image_folder=<str> --input_file=<str>

Options:
    -h --help              Show this help message
    -i --inference         Evaluate the model
    -t --train             Continue training the model
    --model_folder=<str>   Path where the model is saved
    --image_folder=<str>   Path where the images are saved
    --input_file=<str>     Input file of patients, image path names
    --output_file=<str>    Output file name
"""
# python model.py -i --model_folder=../models --image_folder=../example_images --input_file=../ex_input_file.csv --output_file=ex_output.csv

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



continual training
- load a partially trained model
'''

import sys 
import os 
import numpy as np
import pandas as pd
from docopt import docopt
from keras.models import load_model
from dsnplus.utils import get_simplified_score, preprocess_image


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


def run_inference_risk_factors(model_folder, image_folder, input_file, output_file, risk_factors=['drusen', 'pigment', 'amd']):
    ''' Basically from DSN, combine examples/predict_drusen with similar predict_pigment, predict_amd
    predicts these risk factors per image (doesn't care about l/r)'''

    df = pd.read_csv(input_file)
    # for each risk factor model, run inference
    for risk_factor in risk_factors:
        model_path = os.path.join(model_folder, risk_factor+'.h5')
        model = load_model(model_path)
        # Image filenames are in image_folder. Do one image at a time (just inference)
        predictions = []
        image_files = [file for file in os.listdir(model_folder) if file.endswith('.h5')]
        for image_file in image_files:
            image = preprocess_image(os.path.join(image_folder, image_file))
            pred =  model.predict([image])
            predictions.append(pred[0][0]) #?? argmax? examine

        df[risk_factor+'_PRED'] = predictions

# df has predictions for each risk factor
    #?? also save? in case only want risk factor, or error in final score 
    # df.to_csv(output_file, index=False)



def continue_train():
    # load model
    # run train?? may need to refernce other files for actual training, such as inception_amd2.py, inception_drusen.py etc (combine these into one train_model.py file?)
    pass

#*To continue training, specify generators and call model.fit. This uses the same code as in the train_model function, EXCEPT do not recompile the model.


if __name__ == "__main__":
    print("running main")
    
    # argv = docopt(__doc__, sys.argv[1:])
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


