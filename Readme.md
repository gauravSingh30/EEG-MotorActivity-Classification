
## Authors 
- [Gaurav Singh](https://github.com/gauravSingh30)

# Motor Activity Classification
The objective is to classifiy the motor activity into 4 classes based on the EEG signal

# Usage
All the jupyter notebook contains the corresponding code with results .
Run the jupyter notebook sequentially to generate the result.

# Data
Dataset can be downloaded from here: [Dataset](https://drive.google.com/drive/folders/1IQkCOOLI9zsaEzijcjKSFQO1N4ofKfQd?usp=sharing)

## File Content

-	ProjectEEG1DCNN.ipynb - contains jupyter notebook for 2d CNN architecture
-	ProjectEEGCNN2D.ipynb - contains jupyter notebook for 2d CNN architecture
-	ProjectEEGCNN1D+LSTM.ipynb - contains jupyter notebook for 1d CNN + LSTM architecture
-   ProjectEEGCNN1D+GRU.ipynb - contains jupyter notebook for 1d CNN + GRU architecture
-   ProjectEEG_1dunet.ipynb - contains jupyter notebook for 1d Unet architecture
-   ProjectEEGCNN2D_Analysis.ipynb - contains jupyter notebook containing the analysis done for the best architecture
-   CWT-2DCNN.ipynb - contains jupyter notebook containing notebook for 2D CNN with CWT transformation
-   CroppingandEnsembling.ipynb - contains jupyter notebook containing notebook for 2D CNN with ensembling and cropping as pre processing
- helper.py contains the ncessary pre-processing functions 
- load_data.py contains data loaders 
