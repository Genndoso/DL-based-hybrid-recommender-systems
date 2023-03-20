import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from deepctr.feature_column import SparseFeat, get_feature_names
from deepctr.models import NFM, DeepFM
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_csv():
    iteraction_df = pd.read_csv('./dataset/interactions_processed.csv')
    item_df = pd.read_csv('./dataset/items_processed.csv')
    user_df = pd.read_csv('./dataset/users_processed.csv')
    return iteraction_df, item_df, user_df

def get_data_df(iteraction_df, item_df, user_df):
    """ combine interaction, items and users"""
    row_array = []
    for _, row in tqdm(iteraction_df.iterrows()):
        curr_row = row.to_dict()

        # get the relavant user and item from our item and user csv
        curr_user_id = row['user_id']
        curr_item_id = row['item_id']
        user_info = user_df[user_df['user_id'] == curr_user_id].squeeze()
        item_info = item_df[item_df['item_id'] == curr_item_id].squeeze()

        # combining the dict
        curr_row = {**curr_row, **user_info}
        curr_row = {**curr_row, **item_info}
        row_array.append(curr_row)
    
    return pd.DataFrame(row_array)



if __name__ == "__main__":
    if not os.path.exists('./dataset/data.csv'):
        # combine our data if we haven't already
        iteraction_df, item_df, user_df = load_csv()
        
        target = iteraction_df['watched_pct']
        data_df = get_data_df(iteraction_df, item_df, user_df)
        data_df.to_csv('./dataset/data.csv')
    else:
        print('already preproccessed data, reading')
        data_df = pd.read_csv('./dataset/data.csv')


    sparse_features = ['item_id','user_id', 'age', 'income','sex','kids_flg',
                                   'content_type','title','title_orig',
                                    'genres','countries','for_kids',
                                    'age_rating','studios','directors','actors',
                                    'description','keywords','release_year_cat']

    target = ['watched_pct']
    for feature in sparse_features:
        lbe = LabelEncoder()
        data_df[feature] = lbe.fit_transform(data_df[feature])
    fixlen_feature_columns = [SparseFeat(feat, data_df[feat].nunique()) for feat in sparse_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    print('splitting data!')
    train, test = train_test_split(data_df, test_size = 0.2)
    train_model_input = {name:train[name].values for name in feature_names}
    test_model_input = {name:test[name].values for name in feature_names}

    print('Training!')
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression', dnn_hidden_units=(256, 128, 64))
    model.compile('rmsprop',loss='mse',metrics=['mse'])
    history = model.fit(train_model_input, train[target].values, batch_size=1500, epochs=1, verbose=True, validation_split=0.1)

    pred_ans = model.predict(test_model_input, batch_size=256)
    print(pred_ans)
    rmse = mean_squared_error(test[target].values, pred_ans, squared=False)
    print('test RMSE', rmse)