import pandas as pd


def load_data(filepath: str):
    """
    Loads WineQT dataset, cleans columns,
    creates binary target and returns X, y, df.
    """
    df = pd.read_csv(filepath)

    df.columns = (df.columns
                  .str.lower()
                  .str.strip()
                  .str.replace(" ", "_"))

    df = df.drop(columns=['id'])
    df['target'] = (df['quality'] >= 6).astype(int)

    X = df.drop(columns=['quality', 'target'])
    y = df['target']

    return X, y, df


if __name__ == "__main__":
    X, y, df = load_data('../data/WineQT.csv')
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Bad wine:  {(y == 0).sum()}")
    print(f"Good wine: {(y == 1).sum()}")