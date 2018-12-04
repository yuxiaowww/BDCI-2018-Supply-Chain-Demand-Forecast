# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 09:57:53 2018

@author: yuwei
"""

import pandas as pd
import gc
import warnings
import xgboost as xgb
import lightgbm as lgb
import datetime
import numpy as np

warnings.filterwarnings("ignore")


path ='dataset//b//'

#%%
def loadData(path):
    "读取数据集"
    goods_sku_relation = pd.read_csv(path+'goods_sku_relation.csv')
    goodsale = pd.read_csv(path+'goodsale.csv', dtype={'goods_num': np.float32})
    goodsdaily = pd.read_csv(path+'goodsdaily.csv', dtype={'goods_click': np.float32,
                                                                                'cart_click': np.float32,
                                                                                'favorites_click': np.float32,
                                                                                'sales_uv': np.float32,
                                                                                'onsale_days': np.float32})  
    goodsinfo = pd.read_csv(path+'goodsinfo.csv')    

    #转化金额字符为数字
    goodsale['goods_price'] = goodsale['goods_price'].apply(lambda x: float(str(x).replace(',', '')))
    goodsale['orginal_shop_price'] = goodsale['orginal_shop_price'].apply(lambda x: float(str(x).replace(',', '')))
    
    
    return goods_sku_relation,goodsale,goodsdaily,goodsinfo

#%%
def make_label(goodsale_label, goodsale_fea):
    label = pd.DataFrame({'sku_id': list(set(goodsale_fea['sku_id']))})
    date = sorted(list(set(goodsale_label['data_date'])))
    for i in range(5):
        start_date = date[i * 7]
        end_date = date[i * 7 + 6]
        sub_goodsale_label = goodsale_label[(goodsale_label['data_date'] >= start_date) & (goodsale_label['data_date'] <= end_date)]
        group = sub_goodsale_label['goods_num'].groupby(sub_goodsale_label['sku_id']).sum()
        df = pd.DataFrame({'sku_id': group.index, 'week' + str(i + 1): group})
        label = pd.merge(label, df, on=['sku_id'], how='left')
    label.sort_values(by=['sku_id'], inplace=True)
    label.fillna(0, inplace=True)
#    label.index = label['sku_id']
#    del label['sku_id']
    return label

def splitData(goodsale,goods_sku_relation):
    "划分数据集"
    #训练集
    y_train = []
    feature_dateRange = [[20170301, 20170327], [20170920, 20171016]]
    label_dateRange = [[20170512, 20170615], [20171201, 20180104]]
    for feature_date, label_date in zip(feature_dateRange, label_dateRange):
        print('make label ing...')
        label = make_label(goodsale[(goodsale['data_date'] >= label_date[0]) & (goodsale['data_date'] <= label_date[1])],
                           goodsale[(goodsale['data_date'] >= feature_date[0]) & (goodsale['data_date'] <= feature_date[1])])
        y_train.append(label)

    "trainset"
    #合并对应的goods_id
    train1 = pd.merge(y_train[0],goods_sku_relation,on='sku_id',how='left')
    train2 = pd.merge(y_train[1],goods_sku_relation,on='sku_id',how='left')

    "testset"
    #feature_date = [20171217, 20180316]
    test = goodsale[(goodsale['data_date'] >= 20180218) & (goodsale['data_date'] <= 20180316)]
    test=test.drop_duplicates(subset='sku_id', keep='first', inplace=False)
    test = test[['sku_id','goods_id']]


    return train1,train2,test

#%% 
def genFeature(goodsdaily,goodsinfo,goodsale,data,date):
    "特征工程"
    ans = data.copy()
    
    #截取对应特征区间
    goodsdaily = goodsdaily[(goodsdaily['data_date']>=date[0])&(goodsdaily['data_date']<=date[1])]
    goodsale = goodsale[(goodsale['data_date']>=date[0])&(goodsale['data_date']<=date[1])]
   
    print('goodsinfo')
    "-----------goodsinfo-------------"
    brand_id = goodsinfo[['goods_id','brand_id']]
    ans = pd.merge(ans,brand_id,on='goods_id',how='left')
    del brand_id;gc.collect()
    del goodsinfo;gc.collect()
    
    print('goodsdaily')
    "-----------goodsdaily-------------"
    #goods点击次数最大值、最小值、均值、方差
    goodsdaily['goods_click_max'] = goodsdaily['goods_click']
    feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='goods_click_max',aggfunc='max').reset_index()    
    ans = pd.merge(ans,feat,on='goods_id',how='left')
    del feat;gc.collect();
    goodsdaily['goods_click_min'] = goodsdaily['goods_click']
    feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='goods_click_min',aggfunc='min').reset_index()    
    ans = pd.merge(ans,feat,on='goods_id',how='left')
    del feat;gc.collect();
    goodsdaily['goods_click_mean'] = goodsdaily['goods_click']
    feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='goods_click_mean',aggfunc='mean').reset_index()    
    ans = pd.merge(ans,feat,on='goods_id',how='left')
    del feat;gc.collect();
    goodsdaily['goods_click_var'] = goodsdaily['goods_click']
    feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='goods_click_var',aggfunc='var').reset_index()    
    ans = pd.merge(ans,feat,on='goods_id',how='left')    
    del feat;gc.collect();
    #计算总次数
    goodsdaily['goods_click_sum'] = goodsdaily['goods_click']
    feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='goods_click_sum',aggfunc='sum').reset_index()    
    ans = pd.merge(ans,feat,on='goods_id',how='left')    
    del feat;gc.collect();    
    
    #goods加购次数最大值、最小值、均值、方差
    goodsdaily['cart_click_max'] = goodsdaily['cart_click']
    feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='cart_click_max',aggfunc='max').reset_index()    
    ans = pd.merge(ans,feat,on='goods_id',how='left')
    del feat;gc.collect();
    goodsdaily['cart_click_min'] = goodsdaily['cart_click']
    feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='cart_click_min',aggfunc='min').reset_index()    
    ans = pd.merge(ans,feat,on='goods_id',how='left')
    del feat;gc.collect();
    goodsdaily['cart_click_mean'] = goodsdaily['cart_click']
    feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='cart_click_mean',aggfunc='mean').reset_index()    
    ans = pd.merge(ans,feat,on='goods_id',how='left')
    del feat;gc.collect();
    goodsdaily['cart_click_var'] = goodsdaily['cart_click']
    feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='cart_click_var',aggfunc='var').reset_index()    
    ans = pd.merge(ans,feat,on='goods_id',how='left')
    del feat;gc.collect();  
    #计算总次数
    goodsdaily['cart_click_sum'] = goodsdaily['cart_click']
    feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='cart_click_sum',aggfunc='sum').reset_index()    
    ans = pd.merge(ans,feat,on='goods_id',how='left')
    del feat;gc.collect();    

    #goods收藏次数最大值、最小值、均值、方差
    goodsdaily['favorites_click_max'] = goodsdaily['favorites_click']
    feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='favorites_click_max',aggfunc='max').reset_index()    
    ans = pd.merge(ans,feat,on='goods_id',how='left')
    del feat;gc.collect();
    goodsdaily['favorites_click_min'] = goodsdaily['favorites_click']
    feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='favorites_click_min',aggfunc='min').reset_index()    
    ans = pd.merge(ans,feat,on='goods_id',how='left')
    del feat;gc.collect();
    goodsdaily['favorites_click_mean'] = goodsdaily['favorites_click']
    feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='favorites_click_mean',aggfunc='mean').reset_index()    
    ans = pd.merge(ans,feat,on='goods_id',how='left')
    del feat;gc.collect();
    goodsdaily['favorites_click_var'] = goodsdaily['favorites_click']
    feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='favorites_click_var',aggfunc='var').reset_index()    
    ans = pd.merge(ans,feat,on='goods_id',how='left')
    del feat;gc.collect();
    #计算总次数
    goodsdaily['favorites_click_sum'] = goodsdaily['favorites_click']
    feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='favorites_click_sum',aggfunc='sum').reset_index()    
    ans = pd.merge(ans,feat,on='goods_id',how='left')
    del feat;gc.collect();     
    
    #goods购买次数最大值、最小值、均值、方差
    goodsdaily['sales_uv_max'] = goodsdaily['sales_uv']
    feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='sales_uv_max',aggfunc='max').reset_index()    
    ans = pd.merge(ans,feat,on='goods_id',how='left')
    del feat;gc.collect();
    goodsdaily['sales_uv_min'] = goodsdaily['sales_uv']
    feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='sales_uv_min',aggfunc='min').reset_index()    
    ans = pd.merge(ans,feat,on='goods_id',how='left')
    del feat;gc.collect();
    goodsdaily['sales_uv_mean'] = goodsdaily['sales_uv']
    feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='sales_uv_mean',aggfunc='mean').reset_index()    
    ans = pd.merge(ans,feat,on='goods_id',how='left')
    del feat;gc.collect();
    goodsdaily['sales_uv_var'] = goodsdaily['sales_uv']
    feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='sales_uv_var',aggfunc='var').reset_index()    
    ans = pd.merge(ans,feat,on='goods_id',how='left')
    del feat;gc.collect(); 
    #计算总次数
    goodsdaily['sales_uv_sum'] = goodsdaily['sales_uv']
    feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='sales_uv_sum',aggfunc='sum').reset_index()    
    ans = pd.merge(ans,feat,on='goods_id',how='left')
    del feat;gc.collect();     
    
    #goods转化率计算
    #点击、加购、收藏的转化率
    ans['sale_click_rate'] = ans['sales_uv_sum']/ans['goods_click_sum']
    ans['sale_cart_rate'] = ans['sales_uv_sum']/ans['cart_click_sum']
    ans['favorites_click'] = ans['sales_uv_sum']/ans['favorites_click_sum']
    "--待定是否删除总次数"
    
    onsale_days = goodsdaily[['goods_id','onsale_days']]
    onsale_days=onsale_days.drop_duplicates(subset='goods_id', keep='first', inplace=False)
    ans = pd.merge(ans,onsale_days,on='goods_id',how='left')
    del onsale_days;gc.collect();  
    
#%% 划分粒度
    goodsdaily['data_date'] = goodsdaily.data_date.map(lambda x :datetime.datetime.strptime(str(x),'%Y%m%d'))
    max_date = max(goodsdaily['data_date'])
    #统计不同粒度下统计值
    for i in [24,21,14,7,5,3,1]:
        print(i)
        goodsdaily = goodsdaily[(goodsdaily['data_date']<=max_date)&(goodsdaily['data_date']>=max_date-datetime.timedelta(days=i))]
        goodsdaily['goods_click_max'+str(i)] = goodsdaily['goods_click']
        feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='goods_click_max'+str(i),aggfunc='max').reset_index()    
        ans = pd.merge(ans,feat,on='goods_id',how='left')
        del feat;gc.collect();
        del goodsdaily['goods_click_max'+str(i)]
        
        goodsdaily['goods_click_min'+str(i)] = goodsdaily['goods_click']
        feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='goods_click_min'+str(i),aggfunc='min').reset_index()    
        ans = pd.merge(ans,feat,on='goods_id',how='left')
        del feat;gc.collect();
        del goodsdaily['goods_click_min'+str(i)]
        
        goodsdaily['goods_click_mean'+str(i)] = goodsdaily['goods_click']
        feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='goods_click_mean'+str(i),aggfunc='mean').reset_index()    
        ans = pd.merge(ans,feat,on='goods_id',how='left')
        del feat;gc.collect();
        del goodsdaily['goods_click_mean'+str(i)] 
        
        goodsdaily['goods_click_var'+str(i)] = goodsdaily['goods_click']
        feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='goods_click_var'+str(i),aggfunc='var').reset_index()    
        ans = pd.merge(ans,feat,on='goods_id',how='left')    
        del feat;gc.collect();
        del goodsdaily['goods_click_var'+str(i)]
        #计算总次数
        goodsdaily['goods_click_sum'+str(i)] = goodsdaily['goods_click']
        feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='goods_click_sum'+str(i),aggfunc='sum').reset_index()    
        ans = pd.merge(ans,feat,on='goods_id',how='left')    
        del feat;gc.collect();  
        del goodsdaily['goods_click_sum'+str(i)]
    
        #goods加购次数最大值、最小值、均值、方差
        goodsdaily['cart_click_max'+str(i)] = goodsdaily['cart_click']
        feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='cart_click_max'+str(i),aggfunc='max').reset_index()    
        ans = pd.merge(ans,feat,on='goods_id',how='left')
        del feat;gc.collect();
        del goodsdaily['cart_click_max'+str(i)]
        
        goodsdaily['cart_click_min'+str(i)] = goodsdaily['cart_click']
        feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='cart_click_min'+str(i),aggfunc='min').reset_index()    
        ans = pd.merge(ans,feat,on='goods_id',how='left')
        del feat;gc.collect();
        del goodsdaily['cart_click_min'+str(i)]


        goodsdaily['cart_click_mean'+str(i)] = goodsdaily['cart_click']
        feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='cart_click_mean'+str(i),aggfunc='mean').reset_index()    
        ans = pd.merge(ans,feat,on='goods_id',how='left')
        del feat;gc.collect();
        del goodsdaily['cart_click_mean'+str(i)]

        goodsdaily['cart_click_var'+str(i)] = goodsdaily['cart_click']
        feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='cart_click_var'+str(i),aggfunc='var').reset_index()    
        ans = pd.merge(ans,feat,on='goods_id',how='left')
        del feat;gc.collect();
        del goodsdaily['cart_click_var'+str(i)]

        #计算总次数
        goodsdaily['cart_click_sum'+str(i)] = goodsdaily['cart_click']
        feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='cart_click_sum'+str(i),aggfunc='sum').reset_index()    
        ans = pd.merge(ans,feat,on='goods_id',how='left')
        del feat;gc.collect();
        del goodsdaily['cart_click_sum'+str(i)]


        #goods收藏次数最大值、最小值、均值、方差
        goodsdaily['favorites_click_max'+str(i)] = goodsdaily['favorites_click']
        feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='favorites_click_max'+str(i),aggfunc='max').reset_index()    
        ans = pd.merge(ans,feat,on='goods_id',how='left')
        del feat;gc.collect();
        del goodsdaily['favorites_click_max'+str(i)]

        goodsdaily['favorites_click_min'+str(i)] = goodsdaily['favorites_click']
        feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='favorites_click_min'+str(i),aggfunc='min').reset_index()    
        ans = pd.merge(ans,feat,on='goods_id',how='left')
        del feat;gc.collect();
        del goodsdaily['favorites_click_min'+str(i)]

        goodsdaily['favorites_click_mean'+str(i)] = goodsdaily['favorites_click']
        feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='favorites_click_mean'+str(i),aggfunc='mean').reset_index()    
        ans = pd.merge(ans,feat,on='goods_id',how='left')
        del feat;gc.collect();
        del goodsdaily['favorites_click_mean'+str(i)]

        goodsdaily['favorites_click_var'+str(i)] = goodsdaily['favorites_click']
        feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='favorites_click_var'+str(i),aggfunc='var').reset_index()    
        ans = pd.merge(ans,feat,on='goods_id',how='left')
        del feat;gc.collect();
        del goodsdaily['favorites_click_var'+str(i)]

        #计算总次数
        goodsdaily['favorites_click_sum'+str(i)] = goodsdaily['favorites_click']
        feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='favorites_click_sum'+str(i),aggfunc='sum').reset_index()    
        ans = pd.merge(ans,feat,on='goods_id',how='left')
        del feat;gc.collect();
        del goodsdaily['favorites_click_sum'+str(i)]

    
        #goods购买次数最大值、最小值、均值、方差
        goodsdaily['sales_uv_max'+str(i)] = goodsdaily['sales_uv']
        feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='sales_uv_max'+str(i),aggfunc='max').reset_index()    
        ans = pd.merge(ans,feat,on='goods_id',how='left')
        del feat;gc.collect();
        del goodsdaily['sales_uv_max'+str(i)]

        goodsdaily['sales_uv_min'+str(i)] = goodsdaily['sales_uv']
        feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='sales_uv_min'+str(i),aggfunc='min').reset_index()    
        ans = pd.merge(ans,feat,on='goods_id',how='left')
        del feat;gc.collect();
        del goodsdaily['sales_uv_min'+str(i)]

        goodsdaily['sales_uv_mean'+str(i)] = goodsdaily['sales_uv']
        feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='sales_uv_mean'+str(i),aggfunc='mean').reset_index()    
        ans = pd.merge(ans,feat,on='goods_id',how='left')
        del feat;gc.collect();
        del goodsdaily['sales_uv_mean'+str(i)]

        goodsdaily['sales_uv_var'+str(i)] = goodsdaily['sales_uv']
        feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='sales_uv_var'+str(i),aggfunc='var').reset_index()    
        ans = pd.merge(ans,feat,on='goods_id',how='left')
        del feat;gc.collect(); 
        del goodsdaily['sales_uv_var'+str(i)]

        #计算总次数
        goodsdaily['sales_uv_sum'+str(i)] = goodsdaily['sales_uv']
        feat = pd.pivot_table(goodsdaily,index=['goods_id'],values='sales_uv_sum'+str(i),aggfunc='sum').reset_index()    
        ans = pd.merge(ans,feat,on='goods_id',how='left')
        del feat;gc.collect(); 
        del goodsdaily['sales_uv_sum'+str(i)]

    
        #goods转化率计算
        #点击、加购、收藏的转化率
        ans['sale_click_rate'+str(i)] = ans['sales_uv_sum'+str(i)]/ans['goods_click_sum'+str(i)]
        ans['sale_cart_rate'+str(i)] = ans['sales_uv_sum'+str(i)]/ans['cart_click_sum'+str(i)]
        ans['favorites_click'+str(i)] = ans['sales_uv_sum'+str(i)]/ans['favorites_click_sum'+str(i)]
      
    del goodsdaily;gc.collect();  
#%%  
    print('goodsale')
    "-----------goodsale-------------"
    #商品销售天数
    feat = goodsale['sku_id'].value_counts().reset_index()
    feat.columns = ['sku_id','sku_sale_day_num']
    ans = pd.merge(ans,feat,on='sku_id',how='left')
    del feat;gc.collect();     
    #商品销售天数/商品销售总天数
    ans['days_rate'] = ans['sku_sale_day_num']/ans['onsale_days']
    
    #销售最后一天距离窗口最后一天的天数
    goodsale['max_sale_day'] = goodsale['data_date']
    feat = pd.pivot_table(goodsale,index=['sku_id'],values='max_sale_day',aggfunc='max').reset_index()    
    ans = pd.merge(ans,feat,on='sku_id',how='left')
    del feat;gc.collect();    
    ans['max_sub_last'] = date[1] - ans['max_sale_day']
    del ans['max_sale_day']
    del ans['onsale_days']
    
    #sku销售量最大值、最小值、均值、方差
    goodsale['goods_num_max'] = goodsale['goods_num']
    feat = pd.pivot_table(goodsale,index=['sku_id'],values='goods_num_max',aggfunc='max').reset_index()    
    ans = pd.merge(ans,feat,on='sku_id',how='left')
    del feat;gc.collect();
    goodsale['goods_num_min'] = goodsale['goods_num']
    feat = pd.pivot_table(goodsale,index=['sku_id'],values='goods_num_min',aggfunc='min').reset_index()    
    ans = pd.merge(ans,feat,on='sku_id',how='left')
    del feat;gc.collect();
    goodsale['goods_num_mean'] = goodsale['goods_num']
    feat = pd.pivot_table(goodsale,index=['sku_id'],values='goods_num_mean',aggfunc='mean').reset_index()    
    ans = pd.merge(ans,feat,on='sku_id',how='left')
    del feat;gc.collect();
    goodsale['goods_num_var'] = goodsale['goods_num']
    feat = pd.pivot_table(goodsale,index=['sku_id'],values='goods_num_var',aggfunc='var').reset_index()    
    ans = pd.merge(ans,feat,on='sku_id',how='left')
    del feat;gc.collect();     
    
    #sku销售价格最大值、最小值、均值、方差
    goodsale['goods_price_max'] = goodsale['goods_price']
    feat = pd.pivot_table(goodsale,index=['sku_id'],values='goods_price_max',aggfunc='max').reset_index()    
    ans = pd.merge(ans,feat,on='sku_id',how='left')
    del feat;gc.collect();
    goodsale['goods_price_min'] = goodsale['goods_price']
    feat = pd.pivot_table(goodsale,index=['sku_id'],values='goods_price_min',aggfunc='min').reset_index()    
    ans = pd.merge(ans,feat,on='sku_id',how='left')
    del feat;gc.collect();
    goodsale['goods_price_mean'] = goodsale['goods_price']
    feat = pd.pivot_table(goodsale,index=['sku_id'],values='goods_price_mean',aggfunc='mean').reset_index()    
    ans = pd.merge(ans,feat,on='sku_id',how='left')
    del feat;gc.collect();
    goodsale['goods_price_var'] = goodsale['goods_price']
    feat = pd.pivot_table(goodsale,index=['sku_id'],values='goods_price_var',aggfunc='var').reset_index()    
    ans = pd.merge(ans,feat,on='sku_id',how='left')
    del feat;gc.collect();   
    
#    #商品吊牌价格和商品平均价格差值的最大值、最小值、均值、方差
#    goodsale['shop_sub_good_price'] = goodsale['orginal_shop_price'] - goodsale['goods_price']
#    goodsale['shop_sub_good_price_max'] = goodsale['shop_sub_good_price']
#    feat = pd.pivot_table(goodsale,index=['sku_id'],values='shop_sub_good_price_max',aggfunc='max').reset_index()    
#    ans = pd.merge(ans,feat,on='sku_id',how='left')
#    del feat;gc.collect();
#    goodsale['shop_sub_good_price_min'] = goodsale['shop_sub_good_price']
#    feat = pd.pivot_table(goodsale,index=['sku_id'],values='shop_sub_good_price_min',aggfunc='min').reset_index()    
#    ans = pd.merge(ans,feat,on='sku_id',how='left')
#    del feat;gc.collect();
#    goodsale['shop_sub_good_price_mean'] = goodsale['shop_sub_good_price']
#    feat = pd.pivot_table(goodsale,index=['sku_id'],values='shop_sub_good_price_mean',aggfunc='mean').reset_index()    
#    ans = pd.merge(ans,feat,on='sku_id',how='left')
#    del feat;gc.collect();
#    goodsale['shop_sub_good_price_var'] = goodsale['shop_sub_good_price']
#    feat = pd.pivot_table(goodsale,index=['sku_id'],values='shop_sub_good_price_var',aggfunc='var').reset_index()    
#    ans = pd.merge(ans,feat,on='sku_id',how='left')
#    del feat;gc.collect();    
    
    #销售天数rank排序
    feat = ans[['goods_id', 'sku_id', 'sku_sale_day_num']]
    feat['rank_sku_sale_day_num_asc'] = feat[['goods_id', 'sku_id']].groupby(['goods_id']).rank(ascending=False, method='min');feat = feat[['sku_id','rank_sku_sale_day_num_asc']]
    ans = pd.merge(ans,feat,on='sku_id',how='left')
    feat = ans[['goods_id', 'sku_id', 'sku_sale_day_num']]
    feat['rank_sku_sale_day_num_dec'] = feat[['goods_id', 'sku_id']].groupby(['goods_id']).rank(ascending=True, method='min');feat = feat[['sku_id','rank_sku_sale_day_num_dec']]
    ans = pd.merge(ans,feat,on='sku_id',how='left')    
    del feat;gc.collect();
    #销售量rank排序
#    feat = ans[['goods_id', 'sku_id', 'goods_num_mean']]
#    feat['rank_goods_num_mean_asc'] = feat[['goods_id', 'sku_id']].groupby(['goods_id']).rank(ascending=False, method='min');feat = feat[['sku_id','rank_goods_num_mean_asc']]
#    ans = pd.merge(ans,feat,on='sku_id',how='left')
#    feat = ans[['goods_id', 'sku_id', 'goods_num_mean']]
#    feat['rank_goods_num_mean_dec'] = feat[['goods_id', 'sku_id']].groupby(['goods_id']).rank(ascending=True, method='min');feat = feat[['sku_id','rank_goods_num_mean_dec']]
#    ans = pd.merge(ans,feat,on='sku_id',how='left')    
#    del feat;gc.collect();
    
    return ans

#%%
def modelXgb(train,test,i):
    "xgb模型"
    train_y = train['week'+str(i)].values
                         
    train_x = train.drop(['sku_id','goods_id','week1','week2','week3','week4','week5'],axis=1).values
    test_x = test.drop(['sku_id','goods_id'],axis=1).values        
                    
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x)
    
    # 模型参数
    params = {'booster': 'gbtree',
              'objective': 'reg:linear',
              #'objective':'count:poisson',
              'eta': 0.03,
              'max_depth': 5,  # 6
              'colsample_bytree': 0.8,#0.8
              'subsample': 0.8,
              #'lambda':300,
              #'scale_pos_weight': 1,
              'min_child_weight': 18  # 2
              }

    # 训练
    watchlist = [(dtrain,'train')]
    bst = xgb.train(params, dtrain, num_boost_round=1500,evals=watchlist)
    # 预测
    predict = bst.predict(dtest)
    test_xy = test[['sku_id']]
    test_xy['week'+str(i)] = predict
    
    return test_xy 

#%%

def modelLgb(train,test,i):
    "lgb模型"
    
    params = {'boosting_type': 'gbdt',
              'objective': 'regression',
              'learning_rate': 0.03,
              'lambda_l1': 0.1,
              'lambda_l2': 0.2,
              'max_depth': 25,
              'num_leaves': 31,
              'min_child_weight': 25}
    pre_result = pd.DataFrame({'sku_id': test['sku_id']})

    train_y = train['week'+str(i)]
                         
    train_x = train.drop(['sku_id','goods_id','week1','week2','week3','week4','week5'],axis=1)
    test_x = test.drop(['sku_id','goods_id'],axis=1) 
    
    
    lgb_train = lgb.Dataset(train_x.values, train_y)
    gbm = lgb.train(params, lgb_train, num_boost_round=1000)
    pre_result['week' + str(i)] = gbm.predict(test_x.values, num_iteration=gbm.best_iteration)
    return pre_result 

    

#%%
if __name__ == '__main__':
   "主函数入口"
   #获取原始数据
   print('获取数据中...')
   goods_sku_relation,goodsale,goodsdaily,goodsinfo = loadData(path)

   print('划分数据集中...')
   train1,train2,test = splitData(goodsale,goods_sku_relation)
   print('提取tr1...')
   tr1 = genFeature(goodsdaily,goodsinfo,goodsale,train1,[20170301, 20170327])
   print('提取tr2...')
   tr2 = genFeature(goodsdaily,goodsinfo,goodsale,train2,[20170920, 20171016])

   tr = pd.concat([tr1,tr2],axis=0)
   del tr1;gc.collect;del tr2;gc.collect
   print('提取te...')
   te = genFeature(goodsdaily,goodsinfo,goodsale,test,[20180218, 20180316])

   del goodsdaily;gc.collect()
   del goodsinfo;gc.collect()
   del goodsale;gc.collect()
   del goods_sku_relation;gc.collect
   del test;gc.collect()
   del train1;gc.collect()
   del train2;gc.collect()

   sku = pd.read_csv(path+'submit_example_2.csv');sku = sku[['sku_id']]
   for i in range(1,6):
       print('week'+str(i)+'训练中...')
       ans = modelLgb(tr,te,i)
       sku = pd.merge(sku,ans,on='sku_id',how='left')  
   
   sku['week1'] = sku.week1.map(lambda x:0 if x<0 else x)
   sku['week2'] = sku.week2.map(lambda x:0 if x<0 else x)
   sku['week3'] = sku.week3.map(lambda x:0 if x<0 else x)
   sku['week4'] = sku.week4.map(lambda x:0 if x<0 else x)
   sku['week5'] = sku.week5.map(lambda x:0 if x<0 else x)
   sku.to_csv('yw-lgb.csv',index=False)


