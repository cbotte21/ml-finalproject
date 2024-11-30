import numpy as np
import pandas as pd

def convert(df:pd.DataFrame):
    """
    Converts dataframe to vw-readable format
    """
    vw_data = (df['rating'].astype(str) + ' |user ' + df['userId'].astype(str) + ' |movie ' + df['movieId'].astype(str)).tolist()
    return vw_data

def split(df:pd.DataFrame, training_size:float=None, n_ratings:int=5, randomstate:int=None):
    """
    Splits the dataframe such that for each user, we remove n_ratings and place those ratings inside dataframe2
    :param df: dataframe
    :param n_ratings: number of ratings from each user to remove
    :param training_size: creates a split using a portion of each user's rating set
    :param randomstate: for reproducibility during model selection
    :return: training:dataframe, testing:dataframe
    """
    df = df.sample(frac=1, random_state=randomstate, ignore_index=True)
    users = df.groupby("userId", group_keys=False)
    if training_size is None:
        max_n_ratings = np.min(df.value_counts(subset="userId"))
        assert(n_ratings < max_n_ratings), (f"Number of ratings {n_ratings} to remove "
                                        f"is greater than some user's number of ratings {max_n_ratings}"
                                        )
        df1 = users.apply(lambda group: group.head(-n_ratings)).reset_index(drop=True)
        df2 = users.tail(n_ratings)
    else:
        df2 = users.apply(lambda group: group.head(-(int(len(group) * training_size))))
        df1 = users.apply(lambda group: group.tail(int(len(group) * training_size)))
    return df1, df2
