from enum import Enum

class DisplaySize(Enum):
  SMALL = 0
  MEDIUM = 1
  BIG = 2

class CropType(Enum):
  NONE = 0
  MOBILE = 1 # Asssumes we're processing a mobile screen
  TRAINING = 2 # Assumes we're dealing with training images


class ConverterType(Enum):
  NONE = 0
  REACT = 1 # React native