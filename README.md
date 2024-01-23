# License plate recognition

## Description
A Dutch plate recognition written in Python using only image processing techniques and no machine learning. The process involves:
- Plate localization
- Character segmentation
- Character recognition
- Adjusting the output

The system works well on videos of Dutch license plates up to about 10m distance.

## Input
You run your video by putting under ``dataset/trainingvideo.avi``.  

## Output
You can see the output under ``dataset/Output.csv``. 

## Project setup
The shell script ``evaluator.sh`` is used for running the project and calculating scores.  
This file initially runs your ``main.py`` file, followed by ``evaluation.py``. 
Do not modify either one of ``evaluator.sh`` or ``evaluation.py`` if you want to see proper outputs.

Rest of the project:
- ``CaptureFrame_Process.py`` for reading the input video
- ``Localization.py`` for figuring out the location of the plate in a frame
- ``Recognize.py`` for figuring out what characters are in a plate
- ``helpers/`` additional methods to help you get started (you do not have to use them, and are there for inspiration)
- ``requirements.txt`` if you want to use additional Python packages make sure to add them here as well

