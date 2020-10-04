# skyfall
Project Skyfall - Mobile interface segmentation & code generation

## Overview
This was an (non successful) attempt to create a sketch->code solution. While i wasn't going to open source this i thought it might be useful to someone at some point.


## Idea
The idea was simple, take human made sketches and convert them to code. To begin with, i started with trying to recognize mobile screen sketches and convert them to react navive components.

*Note: There are more sophisticated methods trying to tackle this problem. My approach was simpler and more personal, recognize custom components and genrate code set by the user. Something like ITTT.*

## Stack

- Python
- OpenCV2 
- Numpy
- TensorFlow + Keras
- PyQt5



## Implementation

The app is built using python and PyQt5 for the UI. App has 2 modes:

### Training

In this mode we can introduce new sketches and components to recognize. Works as follows:
1. Show a sketch that contains X numbers of components ie. 10 toolbars
2. We use OpenCV to automatically segment the sketch into components
3. We label components
4. Components are fed into a convolutional neural network for training using Keras
5. Model is trained and ready for usage

### Recognition

In this mode we can show the app a sketch (using previously trained models), recognize components and generate code. Works as follows:
1. Show a sketch of components we have trained for.
2. We do a prediction using our pre-trained models. i.e [toolbar, text, text, round button, bottom bar]
3. Feed the output into a component parser.
4. Parser ouputs platform specific code i.e react native app

## Why it didn't work

At the time of implementation (2018) image recognition was good with 95%+ accuracy, IF you had a large enough dataset. Model accuracy using few examples was ok but not great with a lot of false positives which would make any practical application of this app frustrating.
The intent was to introduce an easy way to define a sketch style (using few human examples) and use that to generate common code set by the user. For that to work accuracy was paramount otherwise the effort would not be worth it.


