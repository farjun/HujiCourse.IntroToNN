from enum import Enum

class AlexnetLayers(Enum):
  conv1 = 0
  conv2 = 1
  conv3 = 2
  conv4 = 3
  conv5 = 5


class NeuronChoice(object):
  def __init__(self,layer:str, index=None, filter=None, row = None, col = None):
    self.layer = layer
    self.filter = filter
    self.row = row
    self.col = col
    self.index = index

    #sanity check
    if layer.startswith("conv") and (filter is None or row is None or col is None):
      raise ValueError("conv layer neurons need to have filter, row and col")

    if layer.startswith("dense") and (index is None):
      raise ValueError("dense layer neurons need to have index")


