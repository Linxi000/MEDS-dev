import pandas as pd

df = pd.read_parquet("/inspire/hdd/project/embodied-multimodality/public/bwang/data/AIME_2024/aime-2024.parquet")
df['data_source'] = "DigitalLearningGmbH/MATH-lighteval"

df.to_parquet("/inspire/hdd/project/embodied-multimodality/public/bwang/data/AIME_2024/aime-2024-math.parquet", index=False)