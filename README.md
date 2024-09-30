# Towards Accountable AI-Assisted Eye Disease Diagnosis: Workflow Design, External Validation, and Continual Learning

This repository provides related codes, data, and models for the paper titled 'Towards Accountable AI-Assisted Eye Disease Diagnosis: Workflow Design, External Validation, and Continual Learning'.

## Instructions to set up
### Environments
Have python3.8+ and Tensorflow 2.9.1 + installed.

### Clone the repository
git clone https://github.com/ncbi-nlp/deepseenet-plus.git
cd deepseenet-plus

### Install the required libraries
pip install -r requirements.txt

## Models
Please download the models from [here](https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/deeplensnet/models.zip).

## Inference
Inference can be performed with the following command. Replace options with the correct paths. 
This will grade scores for each risk factor, as well as a final simplified severity score.
```
python model.py -i --model_folder=models/ --image_folder=image_set/ --input_file=input_files.csv --output_file=output_file.csv
```
To use the example data, run the following, replacing `models/` with the model folder.
```
python model.py -i --model_folder=models/ --image_folder=example_images/ --input_file=example_input_files.csv --output_file=example_output.csv
```
## Continue training
A saved risk factor model can be further trained. Specify the targeted risk factor with `risk_factor` (either "drusen", "pigment", or "amd").
```
python model.py -t --model_path=model.h5 --image_folder=image_set/ --input_file=input_files.csv --risk_factor=drusen/pigment/amd
```
This will load the model from `model_path`, read images and labels from `input_file`, train the model, and save the latest best model.

## NCBI's Disclaimer
This tool shows the results of research conducted in the [Computational Biology Branch](https://www.ncbi.nlm.nih.gov/research/), [NCBI](https://www.ncbi.nlm.nih.gov/home/about). 

The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely on the basis of information produced on this website. NIH does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced on this website, please see a health care professional. 

More information about [NCBI's disclaimer policy](https://www.ncbi.nlm.nih.gov/home/about/policies.shtml) is available.

About [text mining group](https://www.ncbi.nlm.nih.gov/research/bionlp/).

## For Research Use Only
The performance characteristics of this product have not been evaluated by the Food and Drug Administration and is not intended for commercial use or purposes beyond research use only. 

## Acknowledgement
This research was supported in part by the Intramural Research Program of the National Eye Institute, National Institutes of Health, Department of Health and Human Services, Bethesda, Maryland, and the National Center for Biotechnology Information, National Library of Medicine, National Institutes of Health. The sponsor and funding organization participated in the design and conduct of the study; data collection, management, analysis, and interpretation; and the preparation, review and approval of the manuscript.
The views expressed herein are those of the authors and do not reflect the official policy or position of Walter Reed National Military Medical Center, Madigan Army Medical Center, Joint Base Andrews, the U.S. Army Medical Department, the U.S. Army Office of the Surgeon General, the Department of the Air Force, the Department of the Army/Navy/Air Force, Department of Defense, the Uniformed Services University of the Health Sciences or any other agency of the U.S. Government. Mention of trade names, commercial products, or organizations does not imply endorsement by the U.S. Government.


