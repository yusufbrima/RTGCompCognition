from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import librosa
import tensorflow as tf
import librosa.display
from utils import Preprocessing
from inaSpeechSegmenter import Segmenter
from inaSpeechSegmenter.export_funcs import seg2csv, seg2textgrid
import config
from  pathlib  import Path 
import random


np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

#We are initializing the experiment being analyzed
exp = config.experiment["NAME"]



if __name__ == "__main__":
    """ This script executes the preprocessing routines for the dataset. This includes applying a segmentation algorithm
     inaSpeechSegmenter, as well as filtering out 10% of the frequencies below the Nyquist frequency in each audio file"""
    
    pp =  Preprocessing(config.data['INPUT_PATH'])
    # pp.convert()

    df =  pp.get_files()

    pp.clean() #This function executes the preprocessing which saves the output files in config.data['OUT_PATH']