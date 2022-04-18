import numpy as np
from abc import ABC
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score, roc_curve, auc