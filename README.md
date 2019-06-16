
# Human Emotion Recognition
## Summary

Deep learning human emotion recognition system developed with the Keras API utilising Google Collaboratory's data science workspace similar to the Jupyter Notebook, each active session (virtual machine) has free access to a powerful dedicated Graphics processing unit (GPU) and twelve gigabytes of Random-Access Memory (RAM).

The  deep  learning  AI  system  utilises  a  convolutional  neural  network  (CNN)  to  identify  the  six universally  recognised  emotions  of;  joy,  anger,  sadness,  disgust,  fear  and  surprised.  Supervised learning is used to teach the CNN the correct conclusions and predictions for each emotion. The trained CNN will then be able to make accurate predictions on new, unseen data.

There is not an accepted theory for identifying an emotion. As a result, emotions are believed to be inherently non-scientific due to the ambiguity that comes with emotional expression. Emotions  are  a  state  of  physiological  arousal;  therefore, the AI  system would  need  to identify emotions the same way humans do, based on human behaviour-posture and facial expressions.



### Augmentation

Gathering datasets large enough to train the neural network to be accurate and robust is an expensive task due to the limited complete facial emotion recognition datasets available. Small datasets can lead to the network either overfitting or underfitting. Augmenting the training data should ensure that the networks will generalize better when exposed to new images that are not in the training set, with a minimised risk of overfitting.

- The  dataset  has  been  uniquely  augmented  seventeen  times,  therefore expanding  the  dataset from 1770 unique images to 30090 unique images



### How to use

- Upload the GitHub repository to your Google Drive.

- Navigate to Google Collaboratory, configure your account,
- Execute Application 
- Disclaimer - Only a select amount of images have been inserted as GitHub only allows for files under 100mb, therefore, you must find your own dataset, there are hundreds of open source datasets. A number of test images exist in repository already but for better accuracy; increase the dataset.

## Training

The CNN model utilises a validation split as a performance measure. The training set is split; 80% of the  training  set  is allocated  for training the  model and  20%  is allocated for  testing the  model. The testing data contains 6018image samples. The large dataset allows for the model’s generalisation to be analysed based on the validation accuracy and loss values. Due to a large number of images being tested (testing data), providing a visual representation of the accuracy of the model’s predictions on unseen data.

The model contains two arguments that prompt call-back methods to be executed during training and validation (Keras, 2019); TensorBoard and EarlyStopping.




