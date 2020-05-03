# Recognising-HandWritten-Digits-with-CNN
The following project is used to recognize the handwritten digits with the help of Convolutional Neural Network

# Required
* Python 3.6 or higher
* Pycharm Editor (Download the community version from JetBrains Official Website)

__[IMPORTANT] TensorFlow is used as backend__
Install tensorflow gloablly using pip command in in Command Prompt using the command:-
pip install tensorflow
-> tap Y to install all the required directories

To check python is correctly installed on your system
1. Open Command Propmpt
1. Enter the command python --version
1. You must see a version number as follows 3.6.7 or other

# Creating and Configuring the Project

-->Fire UP the PyCharm and create the project in the specific directory. Create a virtual Environement using the pipenv option.
-->Choose the the project directory in location tab.
-->Check the option saying import the global packages.
-->Click on create Button.
[Wait for the editor the generate the skeleton of the project and indexing]

Go to File -> Settings ->Project ->Project Interpreter
Click on the + sign next to list of installed packages
A new window will pop up
Install the following dependencies and API 
* Keras
* MatplotLib
* Numpy 
* Pillow
* scikit-learn
* pandas

[Wait for all the dependencies to install and configuring the skeleeton and indexing]

# Step to launch the project
1. Clone the project and unzip all the file to the directory of project.
1. First Run the 'train_models.py' to train the Neural Network  
__%ignore the warning everytime as those are for running the tensorflow-GPU version. CPU will be enough to handle the processing%__

1. Run the 'save_final_model.py' to save to model to the directory
__%This will create a file name final_model.h5' in the directory&__

1. Run the 'evaluatin_final_model.py'
__%This will ensure that the saved model is working perfectly with the TESTING DATA twith good enough accuracy%__

1. Finally Run the 'prediction.py' to test the model on the 'sample_image.png' 

__If you want to test a different image.__
--> Save the image file in the project directory.    
--> Go to the 'prediction.py' and in the run_example function change the name of the image in the argument of load_image function to the name of the image.
