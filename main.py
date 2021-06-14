import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import xgboost as xgb
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--type', 
                        help='Get_feature, Train, Predict')

def get_relation(train_data, df, group, arithmetic, rename, merge_col):
    relationship = train_data.groupby(by=group, as_index=False).agg(arithmetic)
    relationship.columns = [val[0] if val[-1] == '' else '_'.join(val) for val in relationship.columns.values]
    relationship.rename(columns=rename, inplace=True)
    df = pd.merge(df, relationship, how='left', on=merge_col)
    return df

def get_feature():
    items = pd.read_csv("./data/items.csv")
    item_categories = pd.read_csv("./data/item_categories.csv")
    train = pd.read_csv("./data/sales_train.csv")
    test = pd.read_csv("./data/test.csv")

    # find outliear
    # # plt.scatter(train['item_price'], train['item_cnt_day'], color='red')
    # # plt.xlabel('item_price')
    # # plt.ylabel('item_cnt_day')
    # # plt.savefig('outlier.jpg')
    # # plt.show()

    # ax = sns.boxplot(x=train.item_price)
    # plt.savefig('./item_price_outliear.jpg')
    # plt.show()

    group = ['date_block_num', 'shop_id', 'item_id']

    # outliers

    train = train[train.item_price < 100000]
    train = train[train.item_cnt_day < 1001]

    median = train[(train.shop_id == 32) & (train.item_id == 2973) & (train.date_block_num == 4) & (
                train.item_price > 0)].item_price.median()
    train.loc[train.item_price < 0, 'item_price'] = median

    # median = train[(train.shop_id == 32) & (train.item_id == 2973) & (train.date_block_num == 4) & (train.item_price > 0)].item_price.median()
    # train.loc[train.item_price < 0, 'item_price'] = median

    train.loc[train.shop_id == 0, 'shop_id'] = 57
    test.loc[test.shop_id == 0, 'shop_id'] = 57
    train.loc[train.shop_id == 1, 'shop_id'] = 58
    test.loc[test.shop_id == 1, 'shop_id'] = 58
    train.loc[train.shop_id == 10, 'shop_id'] = 11
    test.loc[test.shop_id == 10, 'shop_id'] = 11

    test['date_block_num'] = 34

    # add feature
    # 在train中加入item_categories
    category = items[['item_id', 'item_category_id']].drop_duplicates()
    category.set_index(['item_id'], inplace=True)
    category = category.item_category_id
    train['category'] = train.item_id.map(category)

    # 對category name 編碼並加入category name
    item_categories['meta_category'] = item_categories.item_category_name.apply(lambda x: x.split(' ')[0])
    item_categories['meta_category'] = pd.Categorical(item_categories.meta_category).codes
    item_categories.set_index(['item_category_id'], inplace=True)
    meta_category = item_categories.meta_category
    train['meta_category'] = train.category.map(meta_category)

    # 對city編碼並加入city
    # shops['city'] = shops.shop_name.apply(lambda x: str.replace(x, '!', '')).apply(lambda x: x.split(' ')[0])
    # shops['city'] = pd.Categorical(shops['city']).codes
    # city = shops.city
    # train['city'] = train.shop_id.map(city)

    # 加入year and month
    year = pd.concat([train.date_block_num, train.date.apply(lambda x: int(x.split('.')[2]))], axis=1).drop_duplicates()
    year.set_index(['date_block_num'], inplace=True)
    year = year.date.append(pd.Series([2015], index=[34]))

    month = pd.concat([train.date_block_num, train.date.apply(lambda x: int(x.split('.')[1]))], axis=1).drop_duplicates()
    month.set_index(['date_block_num'], inplace=True)
    month = month.date.append(pd.Series([11], index=[34]))

    all_shops_items = []
    for block_num in train['date_block_num'].unique():
        unique_shops = train[train['date_block_num'] == block_num]['shop_id'].unique()
        unique_items = train[train['date_block_num'] == block_num]['item_id'].unique()
        all_shops_items.append(np.array(list(itertools.product([block_num], unique_shops, unique_items)), dtype='int32'))
    df = pd.DataFrame(np.vstack(all_shops_items), columns=group, dtype='int32')
    df = df.append(test, sort=True)
    # df['ID'] = df.ID.fillna(-1).astype('int32')
    df['year'] = df.date_block_num.map(year)
    df['month'] = df.date_block_num.map(month)
    df['category'] = df.item_id.map(category)
    df['meta_category'] = df.category.map(meta_category)
    # df['city'] = df.shop_id.map(city)


    # date_block_num, shop_id, item_id 和 銷售量的關係
    df = get_relation(train, df, group, {'item_cnt_day': ['sum']}, {'item_cnt_day_sum': 'target'}, group)

    # 月份、商品 和銷售量的關係
    df = get_relation(train, df, ['date_block_num', 'item_id'], {'item_cnt_day': ['sum']}, {'item_cnt_day_sum': 'target_item'}, ['date_block_num', 'item_id'])

    # 月份、商店 和銷售量的關係
    df = get_relation(train, df, ['date_block_num', 'shop_id'], {'item_cnt_day': ['sum']}, {'item_cnt_day_sum': 'target_shop'}, ['date_block_num', 'shop_id'])

    # 月份、商品種類 和銷售量的關係
    df = get_relation(train, df, ['date_block_num', 'category'], {'item_cnt_day': ['sum']}, {'item_cnt_day_sum': 'target_category'}, ['date_block_num', 'category'])

    # 月份、商品 和商品金額的關係
    df = get_relation(train, df, ['date_block_num', 'item_id'], {'item_price': ['mean', 'max']}, {'item_price_mean': 'target_price_mean', 'item_price_max': 'target_price_max'}, ['date_block_num', 'item_id'])


    df.fillna(0, inplace=True)
    df.to_pickle('./feature.pkl')
    print(df)

def train():
    df = pd.read_pickle("feature.pkl")
    train = df[df.date_block_num < 34]
    test = df[df.date_block_num == 34]


    # x_train, x_test, y_train, y_test = train_test_split(train.drop(['target'],  axis=1), train['target'], test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)

    Train = xgb.DMatrix(train.drop(['target'],  axis=1), label=train['target'])
    val_y = val['target']
    Val = xgb.DMatrix(val.drop(['target'],  axis=1), label=val['target'])
    Test = xgb.DMatrix(test.drop(['target'], axis=1))
    xgb_params = {
        'eval_metric': 'rmse',
        'lambda': '0.171', 
        'gamma': '0.124',
        'booster': 'gbtree', 
        'alpha': '0.170',
        'objective': 'reg:squarederror',
        'colsample_bytree': '0.715',
        'subsample': '0.874', 
        'silent': True,
        'min_child_weight': 26,
        'eta': '0.148',
        'max_depth': 6,
        'tree_method': 'gpu_hist', 
        'n_gpus': 1
    }
    model = xgb.train(xgb_params, Train, 1500, [(Train, 'Train'), (Val, 'Val')], early_stopping_rounds=50, verbose_eval=1)
    predict = model.predict(Val)
    model.save_model('test.model')
    predict = np.where(predict < 0, 0, predict)

    x_ax = range(len(predict))
    plt.plot(x_ax, val_y, label="original")
    plt.plot(x_ax, predict, label="predict")
    plt.legend()
    plt.savefig('./predict_result.jpg')
    plt.show()
    # mse = mean_squared_error(y_test, predict)

    # xgbr = xgb.XGBRegressor(verbosity=0, n_estimators=1)
    # print(xgbr)
    # xgbr.fit(x_train, y_train, eval_metric='rmse', verbose=True)

    # score = xgbr.score(x_train, y_train)
    # predict = xgbr.predict(x_test)
    # mse = mean_squared_error(y_test, predict)
    # print("mse: ", mse)
    # print("rmse: ", mse**(1/2.0))

    # x_ax = range(len(predict))
    # plt.plot(x_ax, y_test, label="original")
    # plt.plot(x_ax, predict, label="predict")
    # plt.legend()
    # plt.savefig('./predict_result.jpg')
    # plt.show()

    # test = test.drop(['target'], axis=1)
    # predict = xgbr.predict(test).clip(0, 20)
    # predict = np.around(predict, decimals=1)
    # id = np.arange(len(predict))
    # id = pd.DataFrame(id, columns=['ID'])
    # output = pd.concat([id, pd.DataFrame(predict, columns=['item_cnt_month'])], axis=1)
    # output.to_csv('submission_me.csv', index=False)
    # xgbr.save_model('test.model')

def predict():
    df = pd.read_pickle("feature.pkl")
    test = df[df.date_block_num == 34]
    test = xgb.DMatrix(test.drop(['target'], axis=1))
    bst = xgb.Booster()
    bst.load_model("test.model")
    predict = bst.predict(test)
    predict = np.around(predict, decimals=1)
    predict = np.where(predict < 0, 0, predict)
    id = np.arange(len(predict))
    id = pd.DataFrame(id, columns=['ID'])
    output = pd.concat([id, pd.DataFrame(predict, columns=['item_cnt_month'])], axis=1)
    output.to_csv('submission.csv', index=False)
    
    # xgbr = xgb.XGBRegressor()
    # xgbr.load_model("test.model")
    # predict = xgbr.predict(test).clip(0, 20)
    # predict = np.around(predict, decimals=1)
    # id = np.arange(len(predict))
    # id = pd.DataFrame(id, columns=['ID'])
    # output = pd.concat([id, pd.DataFrame(predict, columns=['item_cnt_month'])], axis=1)
    # output.to_csv('submission_me.csv', index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.type == "Get_feature":
        get_feature()
    elif args.type == "Train":
        train()
    else:
        predict()