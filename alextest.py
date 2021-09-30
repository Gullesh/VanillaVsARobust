import sys
import os
sys.path.append('/home/mallet/Desktop/sam')
os.chdir('/home/mallet/Desktop/sam')
from sam.utils import load_madry_model

mdl = load_madry_model(arch='madry_alexnet', my_attacker=True)