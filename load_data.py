import torch
import pandas as pd

def load_warfarin_data():
	warfarin_frame = pd.read_csv('data/warfarin.csv')
	print(warfarin_frame)

load_warfarin_data()