import polars as pl
import numpy as np

# read csv file
df = pl.read_csv("data/raw/train.csv")

# shuffle the DataFrame (version-agnostic)
df = df.with_columns(pl.lit(np.random.rand(df.height)).alias("_rand")).sort("_rand").drop("_rand")

# calculate split index for 80/20 split
split_idx = int(0.8 * df.height)

train_df = df[:split_idx]
test_df = df[split_idx:]

# save to csv files
train_df.write_csv("data/processed/train.csv")
test_df.write_csv("data/processed/test.csv")