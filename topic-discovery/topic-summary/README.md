# Text Analysis
This repo contains summary extraction from a group of texts. 

## Packages
#### Python
Tested on python3.8

Install python packages : 
```
pip install -r requirements.txt
```
Download Spacy model : 
```
python -m spacy download en_core_web_sm
```

### Summary Extraction
```
python get_summary.py --data_path <DATA_PATH>     // files after preprocessing
                      --output_path <OUTPUT_PATH> // json file
```