import gc
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime


# 特征提取
def get_fea(sku_id, goodsale):
    fea = pd.DataFrame({'sku_id': sku_id.values})
    fea.reset_index(drop=True, inplace=True)

    fea = sku_price_fea(fea, goodsale)
    print(fea.shape[0], fea.shape[1] - 1)
    fea = sku_sale_num_fea(fea, goodsale)
    print(fea.shape[0], fea.shape[1] - 1)
    fea = last_day_fea(fea, goodsale)
    print(fea.shape[0], fea.shape[1] - 1)
    fea = goodsale_sku_slide_fea(fea, goodsale)
    print(fea.shape[0], fea.shape[1] - 1)

    del fea['sku_id']
    gc.collect()

    fea = fea.astype(np.float32)

    return fea


# sku价格特征
def sku_price_fea(fea, goodsale):
    goodsale['amount'] = goodsale['goods_num'] * goodsale['goods_price']
    amount_sum_df = goodsale.groupby('sku_id')['amount'].sum().reset_index(name='amount_sum')
    data = goodsale.groupby('sku_id')['goods_price'].agg(['max', 'min', 'mean', 'std']).reset_index()
    data.columns = ['sku_id', 'sku_price_max', 'sku_price_min', 'sku_price_mean', 'sku_price_std']
    fea = pd.merge(fea, amount_sum_df, on='sku_id', how='left')
    fea = pd.merge(fea, data, on='sku_id', how='left')
    fea = fea.fillna(0)
    del data
    del amount_sum_df
    gc.collect()
    return fea


# sku销量特征
def sku_sale_num_fea(fea, goodsale):
    data = goodsale.groupby('sku_id')['goods_price'].agg(['max', 'min', 'mean', 'median', 'sum', 'count']).reset_index()
    data.columns = ['sku_id', 'sku_sale_num_max', 'sku_sale_num_min', 'sku_sale_num_mean', 'sku_sale_num_median', 'sku_sale_num_sum', 'sku_sale_num_count']
    fea = pd.merge(fea, data, on='sku_id', how='left')
    fea['sku_sale_num_max_rank'] = fea['sku_sale_num_max'].rank()
    fea['sku_sale_num_min_rank'] = fea['sku_sale_num_min'].rank()
    fea['sku_sale_num_mean_rank'] = fea['sku_sale_num_mean'].rank()
    fea['sku_sale_num_median_rank'] = fea['sku_sale_num_median'].rank()
    fea['sku_sale_num_sum_rank'] = fea['sku_sale_num_sum'].rank()
    fea = fea.fillna(0)
    del data
    gc.collect()
    return fea


# 最后一天特征
def last_day_fea(fea, goodsale):
    sub_goodsale = goodsale[goodsale['data_date'] == goodsale['data_date'].max()]
    data = sub_goodsale.groupby('sku_id')['goods_num'].sum().reset_index(name='last_day_sku_sale_num_sum')
    fea = pd.merge(fea, data, on='sku_id', how='left')
    fea['last_day_sku_sale_num_sum_rank'] = fea['last_day_sku_sale_num_sum'].rank()
    fea = fea.fillna(0)
    del data
    gc.collect()
    return fea


# goodsale表sku滑窗
def goodsale_sku_slide_fea(fea, goodsale):
    for i in [2, 3, 5, 7, 9, 12, 14]:
        date_sort = sorted(list(set(goodsale['data_date'])))
        sub_goodsale = goodsale[goodsale['data_date'] >= date_sort[-i]]
        data = sub_goodsale.groupby('sku_id')['goods_num'].agg(['max', 'min', 'mean', 'median', 'sum']).reset_index()
        data.columns = ['sku_id', 'goodsale_sku_max_slide_' + str(i), 'goodsale_sku_min_slide_' + str(i),
                        'goodsale_sku_mean_slide_' + str(i),  'goodsale_sku_median_slide_' + str(i),
                        'goodsale_sku_sum_slide_' + str(i)]
        fea = pd.merge(fea, data, on=['sku_id'], how='left')
        fea['goodsale_sku_sum_slide_%s_rank_0' % str(i)] = fea['goodsale_sku_sum_slide_' + str(i)].rank()
        fea['goodsale_sku_max_slide_%s_rank_0' % str(i)] = fea['goodsale_sku_max_slide_' + str(i)].rank()
        fea['goodsale_sku_min_slide_%s_rank_0' % str(i)] = fea['goodsale_sku_min_slide_' + str(i)].rank()
        fea['goodsale_sku_mean_slide_%s_rank_0' % str(i)] = fea['goodsale_sku_mean_slide_' + str(i)].rank()
        fea['goodsale_sku_median_slide_%s_rank_0' % str(i)] = fea['goodsale_sku_median_slide_' + str(i)].rank()
        fea = fea.fillna(0)
        del data
        del sub_goodsale
        gc.collect()
    return fea


# 打标签
def get_label(goodsale_label, sku_id):
    label_df = pd.DataFrame({'sku_id': sku_id})
    date = sorted(list(set(goodsale_label['data_date'])))
    for i in range(5):
        data = goodsale_label[(goodsale_label['data_date'] >= date[i * 7]) & (goodsale_label['data_date'] <= date[i * 7 + 6])]
        data = data.groupby('sku_id')['goods_num'].sum().reset_index(name='goods_num')
        data = pd.DataFrame({'sku_id': data['sku_id'], 'week' + str(i + 1): data['goods_num']})
        label_df = pd.merge(label_df, data, on=['sku_id'], how='left')
    label_df.sort_values(by=['sku_id'], inplace=True)
    label_df.fillna(0, inplace=True)
    label_df.index = label_df['sku_id']
    del label_df['sku_id']
    gc.collect()
    return label_df


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    start_time = datetime.strptime(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '%Y-%m-%d %H:%M:%S')
    print('start time :', start_time)

    goodsale_a = pd.read_csv('./dataset/a/goodsale.csv', index_col=False, dtype={'goods_num': np.float32})
    goodsale_b = pd.read_csv('./dataset/b/goodsale.csv', index_col=False, dtype={'goods_num': np.float32})
    submit_example = pd.read_csv('./dataset/b/submit_example_2.csv', index_col=False)
    goodsale = pd.concat([goodsale_a, goodsale_b], ignore_index=True)
    del goodsale_a;del goodsale_b
    gc.collect()

    goodsale.goods_price = goodsale.goods_price.map(lambda x: float(str(x).replace(',', '')))
    goodsale.orginal_shop_price = goodsale.orginal_shop_price.map(lambda x: float(str(x).replace(',', '')))

    X_train = []
    y_train = []
    fea_regions = [[20170612, 20170810], [20170614, 20170812], [20170815, 20171013], [20170818, 20171016], [20170820, 20171018]]
    label_regions = [[20170925, 20171029], [20170927, 20171031], [20171128, 20180101], [20171201, 20180104], [20171203, 20180106]]
    for fea_region, label_region in zip(fea_regions, label_regions):
        print('train %s ing...' % str(fea_regions.index(fea_region) + 1))
        label = get_label(goodsale[(goodsale.data_date >= label_region[0]) & (goodsale.data_date <= label_region[1])],
                          list(set(goodsale[(goodsale.data_date >= fea_region[0]) & (goodsale.data_date <= fea_region[1])]['sku_id'])))
        y_train.append(label)
        X_train.append(get_fea(label.index,
                               goodsale[(goodsale.data_date >= fea_region[0]) & (goodsale.data_date <= fea_region[1])]))
    print('test ing...')
    fea_region = [20180116, 20180316]
    X_test = get_fea(submit_example.sku_id,
                     goodsale[(goodsale.data_date >= fea_region[0]) & (goodsale.data_date <= fea_region[1])])

    del goodsale
    gc.collect()

    X_train = pd.concat(X_train, ignore_index=True)
    y_train = pd.concat(y_train, ignore_index=True)

    print('X_train', X_train.shape)
    print('X_test', X_test.shape)

    print('model predict')
    params = {'boosting': 'gbdt', 'objective': 'regression', 'learning_rate': 0.1,  'max_depth': 7, 'num_leaves': 127,
              'min_child_weight': 25, 'lambda_l1': 0.5, 'lambda_l2': 0.2}
    result = pd.DataFrame({'sku_id': submit_example['sku_id']})
    del submit_example
    gc.collect()
    for i in [1, 2, 3, 4, 5]:
        print('week %s' % str(i))
        lgb_train = lgb.Dataset(X_train.values, y_train['week%s' % str(i)])
        gbm = lgb.train(params, lgb_train, num_boost_round=2500)
        result['week%s' % str(i)] = gbm.predict(X_test.values)
    result = result[result > 0].fillna(0)
    result.to_csv('zh_lgb_v2.csv', index=False)

    end_time = datetime.strptime(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '%Y-%m-%d %H:%M:%S')
    print('end time :', end_time)

    run_time = end_time - start_time
    print('run time :', run_time)
