import gc
import time
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime


# 特征提取
def get_fea(sku_id, goodsdaily, goodsinfo, goodsale, goods_sku_relation):
    fea = pd.DataFrame({'sku_id': sku_id.values})
    fea.reset_index(drop=True, inplace=True)

    fea = goodsdaily_fea(fea, goodsdaily, goods_sku_relation)
    # print(fea.shape[0], fea.shape[1] - 1)
    fea = goodsinfo_fea(fea, goodsinfo, goods_sku_relation)
    # print(fea.shape[0], fea.shape[1] - 1)
    fea = goodsale_sku_slide_fea(fea, goodsale)
    # print(fea.shape[0], fea.shape[1] - 1)
    fea = goodsale_goods_slide_fea(fea, goodsale, goods_sku_relation)
    # print(fea.shape[0], fea.shape[1] - 1)

    fea = fea[sorted(fea.columns)]

    del fea['sku_id']
    gc.collect()

    fea = fea.astype(np.float32)

    return fea


# goodsdaily表
def goodsdaily_fea(fea, goodsdaily, goods_sku_relation):
    for i in [1, 3, 7, 9, 12]:
        date_sort = sorted(list(set(goodsdaily['data_date'])))
        sub_goodsdaily = goodsdaily[goodsdaily['data_date'] >= date_sort[-i]]
        data = sub_goodsdaily.groupby('goods_id')['goods_click', 'cart_click', 'favorites_click', 'sales_uv'].agg(['max', 'min', 'mean', 'count', 'sum']).reset_index()
        data.columns = ['goods_id',
                        'goods_click_max_slide_' + str(i), 'goods_click_min_slide_' + str(i),
                        'goods_click_mean_slide_' + str(i), 'goods_click_count_slide_' + str(i),
                        'goods_click_sum_slide_' + str(i),
                        'cart_click_max_slide_' + str(i), 'cart_click_min_slide_' + str(i),
                        'cart_click_mean_slide_' + str(i), 'cart_click_count_slide_' + str(i),
                        'cart_click_sum_slide_' + str(i),
                        'favorites_click_max_slide_' + str(i), 'favorites_click_min_slide_' + str(i),
                        'favorites_click_mean_slide_' + str(i), 'favorites_click_count_slide_' + str(i),
                        'favorites_click_sum_slide_' + str(i),
                        'sales_uv_max_slide_' + str(i), 'sales_uv_min_slide_' + str(i),
                        'sales_uv_mean_slide_' + str(i), 'sales_uv_count_slide_' + str(i),
                        'sales_uv_sum_slide_' + str(i)]
        data = pd.merge(goods_sku_relation, data, on=['goods_id'], how='left')
        fea = pd.merge(fea, data, on=['sku_id'], how='left')
        del fea['goods_id']
        del data
        del sub_goodsdaily
        gc.collect()
        fea = fea.fillna(0)

    data = goodsdaily.groupby('goods_id')['goods_click', 'cart_click', 'favorites_click', 'sales_uv'].agg(['max', 'min', 'mean', 'sum']).reset_index()
    data.columns = ['goods_id',
                    'goods_click_max', 'goods_click_min', 'goods_click_mean', 'goods_click_sum',
                    'cart_click_max', 'cart_click_min', 'cart_click_mean', 'cart_click_sum',
                    'favorites_click_max', 'favorites_click_min', 'favorites_click_mean', 'favorites_click_sum',
                    'sales_uv_max', 'sales_uv_min', 'sales_uv_mean', 'sales_uv_sum']
    data = pd.merge(goods_sku_relation, data, on=['goods_id'], how='left')
    fea = pd.merge(fea, data, on=['sku_id'], how='left')
    del fea['goods_id']
    del data
    gc.collect()
    fea = fea.fillna(0)
    return fea


# goodsinfo表
def goodsinfo_fea(fea, goodsinfo, goods_sku_relation):
    data = pd.merge(goods_sku_relation, goodsinfo, on=['goods_id'], how='left')
    fea = pd.merge(fea, data, on=['sku_id'], how='left')
    del fea['brand_id']
    del fea['goods_id']
    del fea['cat_level1_id']
    del fea['cat_level2_id']
    del fea['cat_level3_id']
    del fea['cat_level4_id']
    del fea['cat_level5_id']
    del fea['cat_level6_id']
    del fea['cat_level7_id']
    gc.collect()
    fea = fea.fillna(-999)
    return fea


# goodsale表sku滑窗
def goodsale_sku_slide_fea(fea, goodsale):
    for i in [1, 2, 3, 5, 7, 9, 12]:
        date_sort = sorted(list(set(goodsale['data_date'])))
        sub_goodsale = goodsale[goodsale['data_date'] >= date_sort[-i]]
        data = sub_goodsale.groupby('sku_id')['goods_num'].agg(['max', 'mean', 'sum']).reset_index()
        data.columns = ['sku_id', 'goodsale_sku_max_slide_' + str(i),  'goodsale_sku_mean_slide_' + str(i),
                        'goodsale_sku_sum_slide_' + str(i)]
        fea = pd.merge(fea, data, on=['sku_id'], how='left')
        fea['goodsale_sku_sum_slide_%s_rank' % str(i)] = fea['goodsale_sku_sum_slide_' + str(i)].rank()
        fea['goodsale_sku_mean_slide_%s_rank' % str(i)] = fea['goodsale_sku_mean_slide_' + str(i)].rank()
        fea = fea.fillna(0)
        del data
        del sub_goodsale
        gc.collect()

    data = goodsale.groupby('sku_id')['goods_num'].agg(['max', 'mean', 'sum']).reset_index()
    data.columns = ['sku_id', 'goodsale_sku_max', 'goodsale_sku_mean', 'goodsale_sku_sum']
    fea = pd.merge(fea, data, on=['sku_id'], how='left')
    del data
    gc.collect()
    return fea


# goodsale表goods滑窗
def goodsale_goods_slide_fea(fea, goodsale, goods_sku_relation):
    for i in [1, 2, 3, 5, 7, 9, 12]:
        date_sort = sorted(list(set(goodsale['data_date'])))
        sub_goodsale = goodsale[goodsale['data_date'] >= date_sort[-i]]
        data = sub_goodsale.groupby('goods_id')['goods_num'].agg(['sum']).reset_index()
        data.columns = ['goods_id', 'goodsale_goods_sum_slide_' + str(i)]
        data = pd.merge(goods_sku_relation, data, on=['goods_id'], how='left')
        fea = pd.merge(fea, data, on=['sku_id'], how='left')
        fea['goodsale_goods_sum_slide_%s_rank' % str(i)] = fea['goodsale_goods_sum_slide_' + str(i)].rank()
        fea = fea.fillna(0)
        del fea['goods_id']
        del sub_goodsale
        del data
        gc.collect()

    data = goodsale.groupby('goods_id')['goods_num'].sum().reset_index(name='goodsale_goods_sum')
    data = pd.merge(goods_sku_relation, data, on=['goods_id'], how='left')
    fea = pd.merge(fea, data, on=['sku_id'], how='left')
    fea = fea.fillna(0)
    del fea['goods_id']
    del data
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
    # print('start time :', start_time)

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
    fea_regions = [[20170613, 20170811], [20170816, 20171014], [20170819, 20171017], [20170822, 20171020]]
    label_regions = [[20170926, 20171030], [20171129, 20180102], [20171202, 20180105], [20171205, 20180108]]
    for fea_region, label_region in zip(fea_regions, label_regions):
        # print('train %s ing...' % str(fea_regions.index(fea_region) + 1))
        label = get_label(goodsale[(goodsale.data_date >= label_region[0]) & (goodsale.data_date <= label_region[1])],
                          list(set(goodsale[(goodsale.data_date >= fea_region[0]) & (goodsale.data_date <= fea_region[1])]['sku_id'])))
        y_train.append(label)
        X_train.append(get_fea(label.index,
                               goodsdaily[(goodsdaily.data_date >= fea_region[0]) & (goodsdaily.data_date <= fea_region[1])],
                               goodsinfo,
                               goodsale[(goodsale.data_date >= fea_region[0]) & (goodsale.data_date <= fea_region[1])],
                               goods_sku_relation))
    # print('test ing...')
    fea_region = [20180116, 20180316]
    X_test = get_fea(submit_example.sku_id,
                     goodsdaily[(goodsdaily.data_date >= fea_region[0]) & (goodsdaily.data_date <= fea_region[1])],
                     goodsinfo,
                     goodsale[(goodsale.data_date >= fea_region[0]) & (goodsale.data_date <= fea_region[1])],
                     goods_sku_relation)

    del goodsdaily;del goodsinfo;del goodsale;del goods_sku_relation
    gc.collect()

    X_train = pd.concat(X_train, ignore_index=True)
    y_train = pd.concat(y_train, ignore_index=True)

    # print('X_train', X_train.shape)
    # print('X_test', X_test.shape)

    # print('model predict')
    params = {'booster': 'gbtree', 'objective': 'reg:linear', 'eval_metric': 'rmse', 'eta': 0.02, 'min_child_weight': 18, 'max_depth': 6,
              'lambda': 5, 'gamma': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.7, 'colsample_bylevel': 0.8}
    result = pd.DataFrame({'sku_id': submit_example['sku_id']})
    del submit_example
    gc.collect()
    for i in [1, 2, 3, 4, 5]:
        # print('week %s' % str(i))
        dtrain = xgb.DMatrix(X_train.values, label=y_train['week%s' % str(i)])
        dtest = xgb.DMatrix(X_test.values)
        bst = xgb.train(params, dtrain, num_boost_round=1000)
        result['week%s' % str(i)] = bst.predict(dtest)
    result = result[result > 0].fillna(0)
    result.to_csv('zh_xgb_v2.csv', index=False)

    end_time = datetime.strptime(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '%Y-%m-%d %H:%M:%S')
    # print('end time :', end_time)

    run_time = end_time - start_time
    # print('run time :', run_time)
