import numpy as np
import pandas as pd
from pyvi import ViTokenizer
import glob
from collections import Counter
from string import punctuation
paths = glob.glob("./comment/*.txt")
comments = []
for path in paths :
  with open(path,encoding="utf-8") as file:
      text= file.read()
      text_lower = text.lower()
      text_token = ViTokenizer.tokenize(text_lower)
      comments.append(text_token)
  file.close()