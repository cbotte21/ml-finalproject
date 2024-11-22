import vowpalwabbit
import pandas as pd
import math
from preprocessing import convert
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def main():
    df = pd.read_csv('ratings.csv').drop("timestamp", axis=1)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=1)
    training = convert(train_df)
    testing = convert(test_df)

    #initialize the model
    model = vowpalwabbit.Workspace(q=['um'], rank=15, l2=0.001, l=0.01, passes=5,
                                   arg_str='--decay_learning_rate 0.99 --power_t 0 --cache_file model.cache --initial_weight 0',
                                   P=20
    )
    for example in training:
        model.learn(example)
    model.finish()
    model.save("model.vw")

    predictions = []
    for example in testing:
        prediction = model.predict(example)
        predictions.append(prediction)

    rmse = math.sqrt(mean_squared_error(test_df['rating'], predictions))
    print(f'RMSE: {rmse}')
if __name__ == '__main__':
    main()
