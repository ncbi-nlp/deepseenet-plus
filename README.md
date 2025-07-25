# AI Workflow, External Validation, and Development in Eye Disease Diagnosis

This repository provides related codes, data, and models for the paper titled `AI Workflow, External Validation, and Development in Eye Disease Diagnosis <[https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2836426]'.


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
To use the example data provided, run the following, replacing `models/` with the model folder.
```
python model.py -i --model_folder=models/ --image_folder=examples/example_images/ --input_file=examples/example_input_file.csv --output_file=examples/example_output.csv
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

## Citing Our Work
=================

*  Chen, Q*., Keenan, T.D*., Agron, E., Allot, A., Guan, E., Duong, B., Elsawy, A., Hou, B., Xue, C., Bhandari, S. and Broadhead, G., 2025. AI Workflow, External Validation, and Development in Eye Disease Diagnosis. JAMA Network Open, 8(7), pp.e2517204-e2517204.



## Acknowledgement
This research was supported by R00LM014024 and the NIH Intramural Research Program of National Library of Medicine and National Eye Institute. Dr. Mehta would also like to acknowledge a departmental unrestricted grant by Research to Prevent Blindness. The views do not reflect the official policy or position of the U.S. Army Medical Department, the U.S. Army Office of the Surgeon General, the Department of the Army, Department of Defense, the Uniformed Services University of the Health Sciences, or any other agency of the U.S. Government.


