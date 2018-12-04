import gc
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression


# 特征提取
def get_fea(sku_id, goodsdaily, goodsale, goods_sku_relation):
    fea = pd.DataFrame({'sku_id': sku_id.values})
    fea.reset_index(drop=True, inplace=True)

    fea = goodsdaily_fea(fea, goodsdaily, goods_sku_relation)
    print(fea.shape[0], fea.shape[1] - 1)
    fea = sku_price_fea(fea, goodsale)
    print(fea.shape[0], fea.shape[1] - 1)
    fea = goodsale_sku_slide_fea(fea, goodsale)
    print(fea.shape[0], fea.shape[1] - 1)
    # fea = goodsale_goods_slide_sum_fea(fea, goodsale, goods_sku_relation)
    # print(fea.shape[0], fea.shape[1] - 1)
    fea = goodsale_rank_fea(fea)
    print(fea.shape[0], fea.shape[1] - 1)

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
        fea = fea.fillna(0)
    return fea


# sku价格特征
def sku_price_fea(fea, goodsale):
    data = goodsale.groupby('sku_id')['goods_price'].agg(['max', 'min', 'mean', 'var']).reset_index()
    data.columns = ['sku_id', 'sku_price_max', 'sku_price_min', 'sku_price_mean', 'sku_price_var']
    fea = pd.merge(fea, data, on='sku_id', how='left')
    fea = fea.fillna(0)
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
                        'goodsale_sku_mean_slide_' + str(i), 'goodsale_sku_median_slide_' + str(i),
                        'goodsale_sku_sum_slide_' + str(i)]
        fea = pd.merge(fea, data, on=['sku_id'], how='left')
        fea = fea.fillna(0)
        del data
        del sub_goodsale
        gc.collect()
    return fea


# goodsale表goods滑窗求和
def goodsale_goods_slide_sum_fea(fea, goodsale, goods_sku_relation):
    for i in [3, 5, 7, 14, 21, 27]:
        date_sort = sorted(list(set(goodsale['data_date'])))
        sub_goodsale = goodsale[goodsale['data_date'] >= date_sort[-i]]
        data = sub_goodsale.groupby('goods_id')['goods_num'].sum().reset_index(name='goodsale_goods_slide_sum_' + str(i))
        data = pd.merge(goods_sku_relation, data, on=['goods_id'], how='left')
        fea = pd.merge(fea, data, on=['sku_id'], how='left')
        fea = fea.fillna(0)
        del fea['goods_id']
        del data
        del sub_goodsale
        gc.collect()
    return fea


# goodsale表rank
def goodsale_rank_fea(fea):
    for i in [3, 5, 7, 14, 21, 27]:
        fea['goodsale_sku_sum_slide_%s_rank' % str(i)] = fea['goodsale_sku_sum_slide_' + str(i)].rank(ascending=True, method='min')
        fea['goodsale_sku_max_slide_%s_rank' % str(i)] = fea['goodsale_sku_max_slide_' + str(i)].rank(ascending=True, method='min')
        fea['goodsale_sku_min_slide_%s_rank' % str(i)] = fea['goodsale_sku_min_slide_' + str(i)].rank(ascending=True, method='min')
        fea['goodsale_sku_mean_slide_%s_rank' % str(i)] = fea['goodsale_sku_mean_slide_' + str(i)].rank(ascending=True, method='min')
        fea['goodsale_sku_median_slide_%s_rank' % str(i)] = fea['goodsale_sku_median_slide_' + str(i)].rank(ascending=True, method='min')
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
                                                                                'sales_uv': np.float32,
                                                                                'onsale_days': np.float32})
    goodsale = pd.read_csv('./dataset/b/goodsale.csv', index_col=False, dtype={'goods_num': np.float32})
    goods_sku_relation = pd.read_csv('./dataset/b/goods_sku_relation.csv', index_col=False)
    submit_example = pd.read_csv('./dataset/b/submit_example_2.csv', index_col=False)

    goodsale.goods_price = goodsale.goods_price.map(lambda x: float(str(x).replace(',', '')))
    goodsale.orginal_shop_price = goodsale.orginal_shop_price.map(lambda x: float(str(x).replace(',', '')))

    X_train = []
    y_train = []
    fea_region = [20170301, 20170327]
    label_region = [20170512, 20170615]
    print('train ing')
    label = get_label(goodsale[(goodsale['data_date'] >= label_region[0]) & (goodsale['data_date'] <= label_region[1])],
                      list(set(goodsale[(goodsale.data_date >= fea_region[0]) & (goodsale.data_date <= fea_region[1])]['sku_id'])))
    y_train.append(label)
    X_train.append(get_fea(label.index,
                           goodsdaily[(goodsdaily['data_date'] >= fea_region[0]) & (goodsdaily['data_date'] <= fea_region[1])],
                           goodsale[(goodsale['data_date'] >= fea_region[0]) & (goodsale['data_date'] <= fea_region[1])],
                           goods_sku_relation))

    print('test ing...')
    fea_region = [20180218, 20180316]
    X_test = get_fea(submit_example['sku_id'],
                     goodsdaily[(goodsdaily['data_date'] >= fea_region[0]) & (goodsdaily['data_date'] <= fea_region[1])],
                     goodsale[(goodsale['data_date'] >= fea_region[0]) & (goodsale['data_date'] <= fea_region[1])],
                     goods_sku_relation)

    del goodsdaily;del goodsale;del goods_sku_relation
    gc.collect()

    X_train = pd.concat(X_train, ignore_index=True)
    y_train = pd.concat(y_train, ignore_index=True)

    print('X_train', X_train.shape)
    print('X_test', X_test.shape)

    print('model predict')
    result = pd.DataFrame({'sku_id': submit_example['sku_id']})
    del submit_example
    gc.collect()
    lr = LinearRegression()
    for i in [1, 2, 3, 4, 5]:
        print('week %s' % str(i))
        lr.fit(X_train, y_train['week%s' % str(i)])
        result['week%s' % str(i)] = lr.predict(X_test)
    result = result[result > 0].fillna(0)
    result.to_csv('zh_lr_v1.csv', index=False)

    end_time = datetime.strptime(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '%Y-%m-%d %H:%M:%S')
    print('end time :', end_time)

    run_time = end_time - start_time
    print('run time :', run_time)
