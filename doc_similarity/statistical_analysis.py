import pandas as pd 
import numpy as np

test_df = pd.read_csv(r'doc_similarity\data\test_post_processed.csv')

print(test_df["Tamanho do nódulo (mm)"].value_counts())

nodules_size = test_df["Tamanho do nódulo (mm)"].tolist()

nodules_size = np.array(nodules_size)
print(np.mean(nodules_size))

print(np.std(nodules_size))

