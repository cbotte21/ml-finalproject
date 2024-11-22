import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def convert(df):
    """
    Converts dataframe to vw-readable format
    """
    vw_data = [f"{row['rating']} |user {row['userId']} |movie {row['movieId']}" for _, row in df.iterrows()]
    return vw_data

def split(file, n_ratings=5):
    """
    Converts a csv file into two dataframes, one as testing, the other as training+validation
    :param file: ratings.csv
    :param n_ratings: number of ratings from each user to remove
    :return: testing dataframe, training+validation dataframe
    """
    df = pd.read_csv(file)
    max_n_ratings = np.min(df.value_counts(subset="userId"))

    assert(n_ratings < max_n_ratings), (f"Number of ratings {n_ratings} to remove "
                                        f"is greater than a user's number of ratings {max_n_ratings}"
                                        )
    groups = df.groupby(["userId"])
    train_df = groups.tail(n_ratings)
    test_df = groups.
    print(groups.tail(n_ratings))

def main():
    split("ratings.csv")


if __name__ == "__main__":
    main()




