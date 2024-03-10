import pandas as pd


def solve_file(filename):
    df = pd.read_csv(filename, delimiter=";", names=["City", "Temp"])
    df = df.groupby("City").agg({"Temp": ["min", "mean", "max"]})
    df = df.sort_index()
    return df
