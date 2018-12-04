import gc
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime


# 特征提取
def get_fea(sku_id, goodsinfo, goodsale, goods_sku_relation):
    fea = pd.DataFrame({'sku_id': sku_id.values})
    fea.reset_index(drop=True, inplace=True)

    fea = goods_sku_relation_fea(fea, goodsale, goods_sku_relation)
    print(fea.shape[0], fea.shape[1] - 1)
    fea = goodsinfo_fea(fea, goodsinfo, goods_sku_relation)
    print(fea.shape[0], fea.shape[1] - 1)
    fea = sku_price_fea(fea, goodsale)
    print(fea.shape[0], fea.shape[1] - 1)
    fea = sku_sale_num_fea(fea, goodsale)
    print(fea.shape[0], fea.shape[1] - 1)
    fea = last_day_fea(fea, goodsale, goods_sku_relation)
    print(fea.shape[0], fea.shape[1] - 1)
    fea = goodsale_sku_slide_fea(fea, goodsale, goods_sku_relation)
    print(fea.shape[0], fea.shape[1] - 1)

    del fea['sku_id']
    gc.collect()

    fea = fea.astype(np.float32)

    return fea


# goods_sku_relation表
def goods_sku_relation_fea(fea, goodsale, goods_sku_relation):
    data = goodsale[['goods_id', 'sku_id']].drop_duplicates()
    sku_num_df = data.groupby('goods_id')['sku_id'].count().reset_index(name='sku_num')
    data = pd.merge(goods_sku_relation, sku_num_df, on=['goods_id'], how='left')
    fea = pd.merge(fea, data, on=['sku_id'], how='left')
    fea = fea.fillna(0)
    del data
    del sku_num_df
    del fea['goods_id']
    gc.collect()
    return fea


# goodsinfo表
def goodsinfo_fea(fea, goodsinfo, goods_sku_relation):
    data = pd.merge(goods_sku_relation, goodsinfo, on=['goods_id'], how='left')
    fea = pd.merge(fea, data, on=['sku_id'], how='left')
    del fea['brand_id']
    del fea['goods_id']
    del fea['cat_level2_id']
    del fea['cat_level3_id']
    del fea['cat_level4_id']
    del fea['cat_level5_id']
    del fea['cat_level6_id']
    del fea['cat_level7_id']
    gc.collect()
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
def last_day_fea(fea, goodsale, goods_sku_relation):
    sub_goodsale = goodsale[goodsale['data_date'] == goodsale['data_date'].max()]
    data = sub_goodsale.groupby('sku_id')['goods_num'].sum().reset_index(name='last_day_sku_sale_num_sum')
    fea = pd.merge(fea, data, on='sku_id', how='left')
    fea['last_day_sku_sale_num_sum_rank'] = fea['last_day_sku_sale_num_sum'].rank()
    data = sub_goodsale.groupby('goods_id')['goods_num'].sum().reset_index(name='last_day_goods_sale_num_sum')
    data = pd.merge(goods_sku_relation, data, on=['goods_id'], how='left')
    fea = pd.merge(fea, data, on='sku_id', how='left')
    fea = fea.fillna(0)
    del data
    del fea['goods_id']
    gc.collect()
    return fea


# goodsale表sku滑窗
def goodsale_sku_slide_fea(fea, goodsale, goods_sku_relation):
    for i in [3, 5, 7, 14, 21, 35]:
        date_sort = sorted(list(set(goodsale['data_date'])))
        sub_goodsale = goodsale[goodsale['data_date'] >= date_sort[-i]]
        data = sub_goodsale.groupby('sku_id')['goods_num'].agg(['max', 'min', 'mean', 'median', 'sum']).reset_index()
        data.columns = ['sku_id', 'goodsale_sku_max_slide_' + str(i), 'goodsale_sku_min_slide_' + str(i),
                        'goodsale_sku_mean_slide_' + str(i),  'goodsale_sku_median_slide_' + str(i),
                        'goodsale_sku_sum_slide_' + str(i)]
        fea = pd.merge(fea, data, on=['sku_id'], how='left')
        goods_sale_df = sub_goodsale.groupby('goods_id')['goods_num'].sum().reset_index(name='goods_sale)num_sum')
        goods_sale_df = pd.merge(goods_sku_relation, goods_sale_df, on=['goods_id'], how='left')
        fea = pd.merge(fea, goods_sale_df, on=['sku_id'], how='left')
        fea['goodsale_sku_sum_slide_%s_rank_0' % str(i)] = fea['goodsale_sku_sum_slide_' + str(i)].rank()
        fea['goodsale_sku_max_slide_%s_rank_0' % str(i)] = fea['goodsale_sku_max_slide_' + str(i)].rank()
        fea['goodsale_sku_min_slide_%s_rank_0' % str(i)] = fea['goodsale_sku_min_slide_' + str(i)].rank()
        fea['goodsale_sku_mean_slide_%s_rank_0' % str(i)] = fea['goodsale_sku_mean_slide_' + str(i)].rank()
        fea['goodsale_sku_median_slide_%s_rank_0' % str(i)] = fea['goodsale_sku_median_slide_' + str(i)].rank()
        fea = fea.fillna(0)
        del data
        del fea['goods_id']
        del sub_goodsale
        del goods_sale_df
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

    goodsinfo = pd.read_csv('./dataset/b/goodsinfo.csv', index_col=False)
    goodsale = pd.read_csv('./dataset/b/goodsale.csv', index_col=False, dtype={'goods_num': np.float32})
    goods_sku_relation = pd.read_csv('./dataset/b/goods_sku_relation.csv', index_col=False)
    submit_example = pd.read_csv('./dataset/b/submit_example_2.csv', index_col=False)

    goodsale.goods_price = goodsale.goods_price.map(lambda x: float(str(x).replace(',', '')))
    goodsale.orginal_shop_price = goodsale.orginal_shop_price.map(lambda x: float(str(x).replace(',', '')))

    X_train = []
    y_train = []
    fea_regions = [[20170619, 20170807], [20170626, 20170814], [20170703, 20170821], [20170831, 20171019], [20170907, 20171026]]
    label_regions = [[20170921, 20171026], [20170928, 20171102], [20171005, 20171109], [20171203, 20180107], [20171210, 20180114]]
    for fea_region, label_region in zip(fea_regions, label_regions):
        print('train %s ing...' % str(fea_regions.index(fea_region) + 1))
        label = get_label(goodsale[(goodsale.data_date >= label_region[0]) & (goodsale.data_date <= label_region[1])],
                          list(set(goodsale[(goodsale.data_date >= fea_region[0]) & (goodsale.data_date <= fea_region[1])]['sku_id'])))
        y_train.append(label)
        X_train.append(get_fea(label.index,
                               goodsinfo,
                               goodsale[(goodsale.data_date >= fea_region[0]) & (goodsale.data_date <= fea_region[1])],
                               goods_sku_relation))
    print('test ing...')
    fea_region = [20180126, 20180316]
    X_test = get_fea(submit_example.sku_id,
                     goodsinfo,
                     goodsale[(goodsale.data_date >= fea_region[0]) & (goodsale.data_date <= fea_region[1])],
                     goods_sku_relation)

    del goodsinfo;del goodsale;del goods_sku_relation
    gc.collect()

    X_train = pd.concat(X_train, ignore_index=True)
    y_train = pd.concat(y_train, ignore_index=True)

    print('X_train', X_train.shape)
    print('X_test', X_test.shape)

    print('model predict')
    params = {'boosting': 'gbdt', 'objective': 'regression', 'learning_rate': 0.06,  'max_depth': 6, 'num_leaves': 31,
              'min_child_weight': 25, 'lambda_l1': 0.5, 'lambda_l2': 0.2, 'min_child_samples': 500}
    result = pd.DataFrame({'sku_id': submit_example['sku_id']})
    del submit_example
    gc.collect()
    for i in [1, 2, 3, 4, 5]:
        print('week %s' % str(i))
        lgb_train = lgb.Dataset(X_train.values, y_train['week%s' % str(i)])
        gbm = lgb.train(params, lgb_train, num_boost_round=2000)
        result['week%s' % str(i)] = gbm.predict(X_test.values)
    result = result[result > 0].fillna(0)
    result.to_csv('zh_lgb_v4.csv', index=False)

    end_time = datetime.strptime(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '%Y-%m-%d %H:%M:%S')
    print('end time :', end_time)

    run_time = end_time - start_time
    print('run time :', run_time)
