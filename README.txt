Some Essential Notes to Utilise The System.
---------------------------------------------

The file in .ipynb format therefore must be opened in a notebook such as; Jupyter Notebook or Google Colabatory.
For the system to fully function as expected, use Google Colabatory.

** The `VERSION - BACK-UP - EmotionRecognition` file is only to be used if the marker does not have access to a Notebook, hence the format of the file is .py. However, the system is designed to be compiled on Google Colabatory **

Getting Started
----------------
The Emotion Recognition system can be found in the 'EmotionRecognition.ipynb' file.
- For the system to work the 'Colab Notebooks' folder must be uploaded to the root directory of the users Google Drive.
- The user will then be prompt to mount their Google Drive storage to the system; this is essential for the system to function.


Training the Model
------------------
The entire dataset is not included in the submission to prevent the folder from being large.
- Pickle files have been used instead; one file contains all the image matrixes and the other contains all the attached image labels.
- The pickle files require much less storage than the entire database.
- The system can skip the augmentation/normalizing the dataset steps due to the pickle file. Execute the code from the pickle loading in point.
- Begining to train the Model would result in a Comprehensive Testing of thirty-two combinations - the optimum model can found in the `Model/` directory.
- The performance of the model can be visualised once training has been completed.

Testing the Model
-----------------
The Live-data allows for the model to be tested on high-quality unseen data. There is one candidate:
	- Jo
Each candidate has expressed six emotions; joy, anger, fear, surprised, sadness and disgust.
The image files are stored in a directory of directories. The file path should be:
- `/content/drive/My Drive/Media/Colab Notebooks/Live-Testing-Set/{Candidates Name}/{Emotion}/{Candidates Name + _ + Emotion}.JPG`
The structure is simple and each image is stored with the same naming convention of the Name of the Candidate and the emotion being expressed, with a underscore seperating the two values.
- Every emotion/image is in lowercase, whereas each name has a capital letter.
- The model will make predictions on the Live-data - providing percentages for the emotions detected.


Comparing Models
----------------
To prevent users from having to carry out the Comprehensive Testing of the model to visualise the results in TensorBoard, the .Log files have been saved in the `Logs/` directory.
- The files can be visualised in TensorBoard.
- There are thirty-two combinations.


Disclaimer
----------------
The `Model/` directory only contains the best performing model as the entire folder would be in excess of 1GB. Therefore, the models can be retrieved by:
- Loading the pickle files.
- Executing the CNN Model Architecture cell, then the script will iterate through each model architecture, saving the model to the `Model/` directory.
