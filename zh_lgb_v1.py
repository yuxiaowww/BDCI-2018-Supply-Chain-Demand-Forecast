import gc
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime


# 特征提取
def get_fea(sku_id, goodsdaily, goodsinfo, goodsale, goods_sku_relation):
    fea = pd.DataFrame({'sku_id': sku_id.values})
    fea.reset_index(drop=True, inplace=True)

    fea = goodsdaily_fea(fea, goodsdaily, goods_sku_relation)
    print(fea.shape[0], fea.shape[1] - 1)
    fea = goodsinfo_fea(fea, goodsinfo, goods_sku_relation)
    print(fea.shape[0], fea.shape[1] - 1)
    # fea = sku_sale_day_num_fea(fea, goodsale)
    # print(fea.shape[0], fea.shape[1] - 1)
    # fea = sku_sale_last_day_distance_day_fea(fea, goodsale)
    # print(fea.shape[0], fea.shape[1] - 1)
    fea = sku_price_fea(fea, goodsale)
    print(fea.shape[0], fea.shape[1] - 1)
    fea = goodsale_sku_slide_fea(fea, goodsale)
    print(fea.shape[0], fea.shape[1] - 1)
    fea = goodsale_goods_slide_fea(fea, goodsale, goods_sku_relation)
    print(fea.shape[0], fea.shape[1] - 1)
    fea = goodsale_rank_fea(fea)
    print(fea.shape[0], fea.shape[1] - 1)
    fea = sku_price_mean_rank_fea(fea, goods_sku_relation)
    print(fea.shape[0], fea.shape[1] - 1)
    fea = same_goods_sku_slide_sum_rank_fea(fea, goods_sku_relation)
    print(fea.shape[0], fea.shape[1] - 1)
    # fea = sku_last_n_day_sale_num_fea(fea, goodsale)
    # print(fea.shape[0], fea.shape[1] - 1)

    fea = fea[sorted(fea.columns)]

    del fea['sku_id']
    gc.collect()

    fea = fea.astype(np.float32)

    return fea


# goodsdaily表
def goodsdaily_fea(fea, goodsdaily, goods_sku_relation):
    for i in [3, 5, 7, 14, 21, 27]:
        date_sort = sorted(list(set(goodsdaily['data_date'])))
        sub_goodsale = goodsdaily[goodsdaily['data_date'] >= date_sort[-i]]
        data = sub_goodsale.groupby('goods_id')['goods_click', 'cart_click', 'favorites_click', 'sales_uv'].agg(['max', 'min', 'mean', 'sum']).reset_index()
        data.columns = ['goods_id',
                        'goods_click_max_slide_' + str(i), 'goods_click_min_slide_' + str(i),
                        'goods_click_mean_slide_' + str(i), 'goods_click_sum_slide_' + str(i),
                        'cart_click_max_slide_' + str(i), 'cart_click_min_slide_' + str(i),
                        'cart_click_mean_slide_' + str(i), 'cart_click_sum_slide_' + str(i),
                        'favorites_click_max_slide_' + str(i), 'favorites_click_min_slide_' + str(i),
                        'favorites_click_mean_slide_' + str(i), 'favorites_click_sum_slide_' + str(i),
                        'sales_uv_max_slide_' + str(i), 'sales_uv_min_slide_' + str(i),
                        'sales_uv_mean_slide_' + str(i), 'sales_uv_sum_slide_' + str(i)]
        data = pd.merge(goods_sku_relation, data, on=['goods_id'], how='left')
        fea = pd.merge(fea, data, on=['sku_id'], how='left')
        del fea['goods_id']
        del data
        del sub_goodsale
        gc.collect()
        fea = fea.fillna(-999)
    return fea


# goodsinfo表
def goodsinfo_fea(fea, goodsinfo, goods_sku_relation):
    data = pd.merge(goods_sku_relation, goodsinfo, on=['goods_id'], how='left')
    fea = pd.merge(fea, data, on=['sku_id'], how='left')
    del fea['brand_id']
    del fea['goods_id']
    del fea['cat_level6_id']
    del fea['cat_level7_id']
    gc.collect()
    fea = fea.fillna(-999)
    return fea


# sku销售天数
def sku_sale_day_num_fea(fea, goodsale):
    sku_sale_day_num = goodsale['sku_id'].value_counts()
    sku_sale_day_num_df = pd.DataFrame({'sku_id': sku_sale_day_num.index, 'sku_sale_day_num': sku_sale_day_num})
    fea = pd.merge(fea, sku_sale_day_num_df, on='sku_id', how='left')
    fea = fea.fillna(0)
    del sku_sale_day_num
    del sku_sale_day_num_df
    gc.collect()
    return fea


# sku销售最后一天距离窗口最后一天的天数
def sku_sale_last_day_distance_day_fea(fea, goodsale):
    last_day = int(goodsale['data_date'].max())
    sku_sale_last_day = goodsale['data_date'].groupby([goodsale['sku_id']]).max()
    sku_sale_last_day_df = pd.DataFrame({'sku_id': sku_sale_last_day.index,
                                         'sku_sale_last_day_distance_day': last_day - sku_sale_last_day})
    fea = pd.merge(fea, sku_sale_last_day_df, on='sku_id', how='left')
    fea = fea.fillna(-999)
    del last_day
    del sku_sale_last_day
    del sku_sale_last_day_df
    gc.collect()
    return fea


# sku价格特征
def sku_price_fea(fea, goodsale):
    data = goodsale.groupby('sku_id')['goods_price'].agg(['max', 'min', 'mean', 'var']).reset_index()
    data.columns = ['sku_id', 'sku_price_max', 'sku_price_min', 'sku_price_mean', 'sku_price_var']
    fea = pd.merge(fea, data, on='sku_id', how='left')
    fea = fea.fillna(-999)
    del data
    gc.collect()
    return fea


# goodsale表sku滑窗
def goodsale_sku_slide_fea(fea, goodsale):
    for i in [3, 5, 7, 14, 21, 27]:
        date_sort = sorted(list(set(goodsale['data_date'])))
        sub_goodsale = goodsale[goodsale['data_date'] >= date_sort[-i]]
        data = sub_goodsale.groupby('sku_id')['goods_num'].agg(['max', 'min', 'mean', 'median', 'sum']).reset_index()
        data.columns = ['sku_id', 'goodsale_sku_max_slide_' + str(i), 'goodsale_sku_min_slide_' + str(i),
                        'goodsale_sku_mean_slide_' + str(i),  'goodsale_sku_median_slide_' + str(i),
                        'goodsale_sku_sum_slide_' + str(i)]
        fea = pd.merge(fea, data, on=['sku_id'], how='left')
        fea = fea.fillna(-999)
        del data
        del sub_goodsale
        gc.collect()
    return fea


# goodsale表goods滑窗
def goodsale_goods_slide_fea(fea, goodsale, goods_sku_relation):
    for i in [3, 5, 7, 14, 21, 27]:
        date_sort = sorted(list(set(goodsale['data_date'])))
        sub_goodsale = goodsale[goodsale['data_date'] >= date_sort[-i]]
        data = sub_goodsale.groupby('goods_id')['goods_num'].agg(['max', 'min', 'mean', 'sum']).reset_index()
        data.columns = ['goods_id', 'goodsale_goods_max_slide_' + str(i), 'goodsale_goods_min_slide_' + str(i),
                        'goodsale_goods_mean_slide_' + str(i), 'goodsale_goods_sum_slide_' + str(i)]
        data = pd.merge(goods_sku_relation, data, on=['goods_id'], how='left')
        fea = pd.merge(fea, data, on=['sku_id'], how='left')
        fea = fea.fillna(-999)
        del fea['goods_id']
        del sub_goodsale
        del data
        gc.collect()
    return fea


# goodsale表rank
def goodsale_rank_fea(fea):
    for i in [3, 5, 7, 14, 21, 27]:
        fea['goodsale_sku_sum_slide_%s_rank_0' % str(i)] = fea['goodsale_sku_sum_slide_' + str(i)].rank(ascending=True, method='min')
        fea['goodsale_sku_sum_slide_%s_rank_1' % str(i)] = fea['goodsale_sku_sum_slide_' + str(i)].rank(ascending=False, method='min')
        fea['goodsale_sku_max_slide_%s_rank_0' % str(i)] = fea['goodsale_sku_max_slide_' + str(i)].rank(ascending=True, method='min')
        fea['goodsale_sku_max_slide_%s_rank_1' % str(i)] = fea['goodsale_sku_max_slide_' + str(i)].rank(ascending=False, method='min')
        fea['goodsale_sku_min_slide_%s_rank_0' % str(i)] = fea['goodsale_sku_min_slide_' + str(i)].rank(ascending=True, method='min')
        fea['goodsale_sku_min_slide_%s_rank_1' % str(i)] = fea['goodsale_sku_min_slide_' + str(i)].rank(ascending=False, method='min')
        fea['goodsale_sku_mean_slide_%s_rank_0' % str(i)] = fea['goodsale_sku_mean_slide_' + str(i)].rank(ascending=True, method='min')
        fea['goodsale_sku_mean_slide_%s_rank_1' % str(i)] = fea['goodsale_sku_mean_slide_' + str(i)].rank(ascending=False, method='min')
        fea['goodsale_sku_median_slide_%s_rank_0' % str(i)] = fea['goodsale_sku_median_slide_' + str(i)].rank(ascending=True, method='min')
        fea['goodsale_sku_median_slide_%s_rank_1' % str(i)] = fea['goodsale_sku_median_slide_' + str(i)].rank(ascending=False, method='min')
    return fea


# 同一个goods中，sku平均价格均值排序
def sku_price_mean_rank_fea(fea, goods_sku_relation):
    fea = pd.merge(fea, goods_sku_relation, on='sku_id', how='left')
    sku_price_mean_df = fea[['goods_id', 'sku_id', 'sku_price_mean']]
    sku_price_mean_rank_0 = sku_price_mean_df.groupby('goods_id')['sku_price_mean'].rank(ascending=False, method='min')
    sku_price_mean_rank_1 = sku_price_mean_df.groupby('goods_id')['sku_price_mean'].rank(ascending=True, method='min')
    fea['sku_price_mean_rank_0'] = sku_price_mean_rank_0.values
    fea['sku_price_mean_rank_1'] = sku_price_mean_rank_1.values
    del fea['goods_id']
    del sku_price_mean_df
    del sku_price_mean_rank_0
    del sku_price_mean_rank_1
    gc.collect()
    return fea


# 同一个goods中，sku销售量滑窗排序
def same_goods_sku_slide_sum_rank_fea(fea, goods_sku_relation):
    for i in [3, 5, 7, 14, 21, 27]:
        fea = pd.merge(fea, goods_sku_relation, on='sku_id', how='left')
        sku_sale_num_mean_df = fea[['goods_id', 'sku_id', 'goodsale_sku_sum_slide_' + str(i)]]
        sku_sale_num_mean_rank_0 = sku_sale_num_mean_df.groupby('goods_id')['goodsale_sku_sum_slide_' + str(i)].rank(ascending=False, method='min')
        sku_sale_num_mean_rank_1 = sku_sale_num_mean_df.groupby('goods_id')['goodsale_sku_sum_slide_' + str(i)].rank(ascending=True, method='min')
        fea['same_goods_sku_slide_sum_%s_rank_0' % str(i)] = sku_sale_num_mean_rank_0.values
        fea['same_goods_sku_slide_sum_%s_rank_1' % str(i)] = sku_sale_num_mean_rank_1.values
        del fea['goods_id']
        del sku_sale_num_mean_df
        del sku_sale_num_mean_rank_0
        del sku_sale_num_mean_rank_1
        gc.collect()
    return fea


# 有销量的天数中，sku排第n天的销量数
def sku_last_n_day_sale_num_fea(fea, goodsale):
    group = goodsale['goods_num'].groupby([goodsale['sku_id'], goodsale['data_date']]).sum()
    group = group.unstack()
    group = group.fillna(-999)
    for day in [1, 2, 3, 5, 7, 10, 14]:
        sku_last_n_day_goods_num = group.apply(lambda x: sorted(x)[-day], axis=1).reset_index(name='sku_last_%s_day_goods_num' % str(day))
        fea = pd.merge(fea, sku_last_n_day_goods_num, on='sku_id', how='left')
        del sku_last_n_day_goods_num
        gc.collect()
    del group
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

    goodsdaily = pd.read_csv('./dataset/b/goodsdaily.csv', index_col=False, dtype={'goods_click': np.float32,
                                                                                'cart_click': np.float32,
                                                                                'favorites_click': np.float32,
                                                                                'sales_uv': np.float32,
                                                                                'onsale_days': np.float32})
    goodsinfo = pd.read_csv('./dataset/b/goodsinfo.csv', index_col=False)
    goodsale = pd.read_csv('./dataset/b/goodsale.csv', index_col=False, dtype={'goods_num': np.float32})
    goods_sku_relation = pd.read_csv('./dataset/b/goods_sku_relation.csv', index_col=False)
    submit_example = pd.read_csv('./dataset/b/submit_example_2.csv', index_col=False)

    goodsale.goods_price = goodsale.goods_price.map(lambda x: float(str(x).replace(',', '')))
    goodsale.orginal_shop_price = goodsale.orginal_shop_price.map(lambda x: float(str(x).replace(',', '')))

    X_train = []
    y_train = []
    fea_regions = [[20170301, 20170327], [20170920, 20171016]]
    label_regions = [[20170512, 20170615], [20171201, 20180104]]
    for fea_region, label_region in zip(fea_regions, label_regions):
        print('train %s ing...' % str(fea_regions.index(fea_region) + 1))
        label = get_label(goodsale[(goodsale.data_date >= label_region[0]) & (goodsale.data_date <= label_region[1])],
                          list(set(goodsale[(goodsale.data_date >= fea_region[0]) & (goodsale.data_date <= fea_region[1])]['sku_id'])))
        y_train.append(label)
        X_train.append(get_fea(label.index,
                               goodsdaily[(goodsdaily.data_date >= fea_region[0]) & (goodsdaily.data_date <= fea_region[1])],
                               goodsinfo,
                               goodsale[(goodsale.data_date >= fea_region[0]) & (goodsale.data_date <= fea_region[1])],
                               goods_sku_relation))
    print('test ing...')
    fea_region = [20180218, 20180316]
    X_test = get_fea(submit_example.sku_id,
                     goodsdaily[(goodsdaily.data_date >= fea_region[0]) & (goodsdaily.data_date <= fea_region[1])],
                     goodsinfo,
                     goodsale[(goodsale.data_date >= fea_region[0]) & (goodsale.data_date <= fea_region[1])],
                     goods_sku_relation)

    del goodsdaily;del goodsinfo;del goodsale;del goods_sku_relation
    gc.collect()

    X_train = pd.concat(X_train, ignore_index=True)
    y_train = pd.concat(y_train, ignore_index=True)

    print('X_train', X_train.shape)
    print('X_test', X_test.shape)

    print('model predict')
    params = {'boosting_type': 'gbdt', 'objective': 'regression', 'learning_rate': 0.06, 'max_depth': 6, 'num_leaves': 31,
              'min_child_weight': 25, 'lambda_l1': 0.1, 'lambda_l2': 0.2}
    result = pd.DataFrame({'sku_id': submit_example['sku_id']})
    del submit_example
    gc.collect()
    for i in [1, 2, 3, 4, 5]:
        print('week %s' % str(i))
        lgb_train = lgb.Dataset(X_train.values, y_train['week%s' % str(i)])
        gbm = lgb.train(params, lgb_train, num_boost_round=2000)
        result['week%s' % str(i)] = gbm.predict(X_test.values, num_iteration=gbm.best_iteration)
    result = result[result > 0].fillna(0)
    result.to_csv('zh_lgb_v1.csv', index=False)

    end_time = datetime.strptime(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '%Y-%m-%d %H:%M:%S')
    print('end time :', end_time)

    run_time = end_time - start_time
    print('run time :', run_time)
