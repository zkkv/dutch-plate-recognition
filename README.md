# License plate recognition template

This is the base template for the License Plate Recognition project from the CSE2225 Image Processing course.

The goal of the project is: given an input video you should recognize the license plates.  

## Input
You can see an example video under ``dataset/dummytestvideo.avi``.  
We recommend splitting the video into a training set and a testing set, and only using the testing part as a way to calculate your expected score.   
This is important because for grading we use a totally different video.

## Output
You can see an example of output under ``dataset/sampleOutput.csv``.  
Your output file should be in the same format.

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
- ``.gitlab-ci.yml`` Gitlab pipeline file

Pipeline:  
If you want to see your score you can change the file in ``.gitlab-ci.yml`` to run on the ``trainingvideo.avi``. 
We do not recommend always having this uncommented because running it on the full video makes the pipeline significantly slower.

