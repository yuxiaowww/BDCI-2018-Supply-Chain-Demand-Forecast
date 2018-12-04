# -*- coding: utf-8 -*-
#用来划分训练集和验证集
import pandas as pd
# import lightgbm as lgb
# from compiler.ast import flatten
import gc
import time
import operator
from functools import reduce
from scipy.sparse import hstack,vstack,csc_matrix
from sklearn.feature_extraction.text import CountVectorizer
# import xgboost as xgb
import numpy as np
from sklearn import preprocessing
import operator
import matplotlib.pyplot as plt
from dateutil.parser import parse
from sklearn.cross_validation import train_test_split
from pandas import Series,DataFrame
import time
import datetime
import scipy.stats as sp
from scipy import sparse
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics

def regressor():
    print('reading..')
    train1 = pd.read_csv('train1.csv')  #
    train2 = pd.read_csv('train2.csv')  #
    train3 = pd.read_csv('train3.csv')  #
    print('合并train。。')
    frames = [train1, train2,train3]  # 合并
    train = pd.concat(frames, axis=0)
    test = pd.read_csv('test.csv')  #
    test = test.fillna(-99)


    train_X = train [feature]
    train_X = train_X.fillna(-99)
    train_X = train_X.values

    test_X = test[feature].values

    train_Y = train[['week1','week2','week3','week4','week5']]
    train_Y = train_Y.fillna(0)
    train_Y = train_Y.values

    test_temp = test[['sku_id']]

    print(train_X)
    print(train_Y)
    # exit(1)


    print('训练。')
    clf = GradientBoostingRegressor(random_state=0)
    clf = MultiOutputRegressor(clf)
    clf.fit(train_X,train_Y)
    test_Y = clf.predict(test_X)
    print(test_Y)
    df = pd.DataFrame(test_Y, columns=['week1','week2','week3','week4','week5'])  # 将结果生成dataframe
    result = pd.concat([test_temp,df], axis=1)
    result['week1'] = list(map(lambda x:0 if x<0 else x ,result.week1))
    result['week2'] = list(map(lambda x: 0 if x < 0 else x, result.week2))
    result['week3'] = list(map(lambda x: 0 if x < 0 else x, result.week3))
    result['week4'] = list(map(lambda x: 0 if x < 0 else x, result.week4))
    result['week5'] = list(map(lambda x: 0 if x < 0 else x, result.week5))

    temp = result.groupby(['sku_id'])['week1'].agg({'week11': np.sum}).reset_index()
    result = pd.merge(result, temp, on=['sku_id'], how='left')  #
    temp = result.groupby(['sku_id'])['week2'].agg({'week22': np.sum}).reset_index()
    result = pd.merge(result, temp, on=['sku_id'], how='left')  #
    temp = result.groupby(['sku_id'])['week3'].agg({'week33': np.sum}).reset_index()
    result = pd.merge(result, temp, on=['sku_id'], how='left')  #
    temp = result.groupby(['sku_id'])['week4'].agg({'week44': np.sum}).reset_index()
    result = pd.merge(result, temp, on=['sku_id'], how='left')  #
    temp = result.groupby(['sku_id'])['week5'].agg({'week55': np.sum}).reset_index()
    result = pd.merge(result, temp, on=['sku_id'], how='left')  #

    del result['week1']
    del result['week2']
    del result['week3']
    del result['week4']
    del result['week5']

    result.rename(columns={'week11': 'week1'}, inplace=True)
    result.rename(columns={'week22': 'week2'}, inplace=True)
    result.rename(columns={'week33': 'week3'}, inplace=True)
    result.rename(columns={'week44': 'week4'}, inplace=True)
    result.rename(columns={'week55': 'week5'}, inplace=True)

    result.to_csv('cy_0907_1.csv', index=None)  # , header=None





# xgboost
def xgboosts():
    print('xgb---training')
    print('reading..')
    train1 = pd.read_csv('train1.csv')  #
    train2 = pd.read_csv('train2.csv')  #
    train3 = pd.read_csv('train3.csv')  #
    print('合并train。。')
    frames = [train1, train2, train3]  # 合并
    df_train = pd.concat(frames, axis=0)
    df_test = pd.read_csv('test.csv')  #
    feature = [x for x in df_train.columns if x not in ['goods_id', 'sku_id', 'data_date', 'goods_num']]

    print('F len :%s' % len(feature))


    dtrain = xgb.DMatrix(df_train[feature].values,df_train['goods_num'].values)
    del df_train
    gc.collect()
    dpre = xgb.DMatrix(df_test[feature].values)

    param = {'max_depth':5,
             'eta': 0.02,
             # 'objective': 'rank:pairwise',
             # 'objective': 'binary:logistic',
             'objective': 'reg:linear',
             # 'eval_metric': 'auc',
             'colsample_bytree': 0.8,
             'subsample': 0.8,
             'scale_pos_weight': 1,
             # 'booster':'gblinear',
             'silent': 1,
             # 'early_stopping_rounds':20
             # 'min_child_weight':18
             }
    # param['nthread'] =5
    print('xxxxxx')
    watchlist = [(dtrain, 'eval'), (dtrain, 'train')]
    num_round = 60
    bst = xgb.train(param, dtrain, num_round, watchlist)
    print('xxxxxx')
    # 进行预测
    # dtest= xgb.DMatrix(predict)
    preds2 = bst.predict(dpre)
    # 保存整体结果。
    predict = df_test[['sku_id', 'data_date']]
    predict['goods_num'] = preds2

    print('进行结果周统计。。')
    start_day = 20180501
    end_day = 20180507
    for j in range(1, 6):  # 第j周
        print('第%s周'%j)
        week = predict[(predict.data_date <= end_day) & (predict.data_date >= start_day)]
        week_name = 'week' + str(j)
        temp = week.groupby(['sku_id'])['goods_num'].agg({week_name: np.sum}).reset_index()
        predict = pd.merge(predict, temp, on=['sku_id'], how='left')  #
        start_day = int((datetime.datetime(int(str(start_day)[0:4]), int(str(start_day)[4:6]),int(str(start_day)[6:8])) + datetime.timedelta(days=7)).strftime("%Y%m%d"))
        end_day = int((datetime.datetime(int(str(end_day)[0:4]), int(str(end_day)[4:6]), int(str(end_day)[6:8])) + datetime.timedelta(days=7)).strftime("%Y%m%d"))
        print(start_day)
        print(end_day)
    del predict['data_date']
    del predict['goods_num']
    predict = predict.drop_duplicates(['sku_id'])  # 去重
    predict.to_csv('cy_xgb_0910_3.csv',index=None)#, header=None
    print('over..')

def Lr():
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
    print('1、reading...')

    print('reading..')
    # df_train,df_test = GY()

    train1 = pd.read_csv('train1.csv')  #
    feature = [x for x in train1.columns if x not in ['goods_id', 'sku_id', 'data_date', 'goods_num']]
    train1[feature] = train1[feature].astype(np.float32)

    train2 = pd.read_csv('train2.csv')  #
    train2[feature] = train2[feature].astype(np.float32)

    train3 = pd.read_csv('train3.csv')  #
    train3[feature] = train3[feature].astype(np.float32)


    print('合并train。。')
    frames = [train1, train2, train3]  # 合并
    df_train = pd.concat(frames, axis=0)
    df_train = df_train.reset_index()
    del df_train['index']

    # gl_obj = df_train.select_dtypes(include=['object'])
    # colunms = list(gl_obj.columns)
    # del gl_obj
    # gc.collect()
    # df_train[colunms] = df_train[colunms].stack().astype('category').unstack()
    notF = ['goods_id', 'sku_id', 'data_date', 'goods_num']
    for i in range(1, 14):
        name = 'fw' + str(i) + '_1'
        notF.append(name)
    for i in range(1, 25):
        name = 'sw_' + str(i)
        notF.append(name)
    for i in range(1, 9):
        for j in range(1, 5):
            name = 'L' + str(i) + '_' + str(j)
            notF.append(name)
    for i in range(1, 9):
        name = 's' + str(i) + '_1'
        notF.append(name)
    # for i in range(11, 29):
    #     name = 's' + str(i)
    #     notF.append(name)
    # for i in range(1,10):
    #     name = 'f'+str(i)
    #     notF.append(name)
    notF.append('marketing')
    notF.append('plan')
    print(notF)
    feature = [x for x in df_train.columns if x not in notF]
    del train3
    del train2
    del train1
    del frames


    gc.collect()
    print('LR---training')
    lr.fit(df_train[feature].fillna(-99).values,df_train['goods_num'].values)
    del df_train
    gc.collect()
    df_test = pd.read_csv('test.csv')  #
    pro = lr.predict(df_test[feature].fillna(-99).values)  # 计算该预测实例点属于各类的概率
    predict = df_test[['sku_id', 'data_date']]
    del df_test
    gc.collect()
    predict['goods_num'] = pro
    print(pro)
    del pro
    gc.collect()

    print('进行结果周统计。。')
    start_day = 20180501
    end_day = 20180507
    for j in range(1, 6):  # 第j周
        print('第%s周' % j)
        week = predict[(predict.data_date <= end_day) & (predict.data_date >= start_day)]
        week_name = 'week' + str(j)
        temp = week.groupby(['sku_id'])['goods_num'].agg({week_name: np.sum}).reset_index()
        predict = pd.merge(predict, temp, on=['sku_id'], how='left')  #
        start_day = int((datetime.datetime(int(str(start_day)[0:4]), int(str(start_day)[4:6]),
                                           int(str(start_day)[6:8])) + datetime.timedelta(days=7)).strftime("%Y%m%d"))
        end_day = int((datetime.datetime(int(str(end_day)[0:4]), int(str(end_day)[4:6]),
                                         int(str(end_day)[6:8])) + datetime.timedelta(days=7)).strftime("%Y%m%d"))
        print(start_day)
        print(end_day)
    del predict['data_date']
    del predict['goods_num']
    predict = predict.drop_duplicates(['sku_id'])  # 去重
    print('四舍五入')
    import math
    for i in range(1,6):
        col_name = 'week'+str(i)
        predict[col_name] = list(map(lambda x:0 if x<0 else round(x),predict[col_name]))

    predict.to_csv('cy-result.csv', index=None)  # , header=None
    print('train over..')
# LGB
def Light_Gbm():#'item_category_list1','item_category_list2','item_category_list3',,'shop_id'

    print('lgb---training')
    print('reading..')
    train1 = pd.read_csv('train1.csv')  #
    train2 = pd.read_csv('train2.csv')  #
    train3 = pd.read_csv('train3.csv')  #
    print('合并train。。')
    frames = [train1, train2, train3]  # 合并
    df_train = pd.concat(frames, axis=0)
    df_test = pd.read_csv('test.csv')  #
    feature = [x for x in df_train.columns if x not in ['goods_id', 'sku_id', 'data_date', 'goods_num']]

    print('F len :%s' % len(feature))

    df = pd.DataFrame(df_train[feature].columns.tolist(), columns=['feature'])#用于特征选择
    lgb_train = lgb.Dataset(df_train[feature].values, df_train['goods_num'].values)

    # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        # 'metric': {'l2', 'binary_logloss'},
        'num_leaves': 31,#这是控制树模型复杂性的重要参数。理论上，我们可以通过设定num_leaves = 2^(max_depth) 去转变成为depth-wise tree。但这样容易过拟合，因为当这两个参数相等时,  leaf-wise tree的深度要远超depth-wise tree。因此在调参时，往往会把 num_leaves的值设置得小于2^(max_depth)。
        'learning_rate': 0.01,
        'feature_fraction': 0.8,##通过设定 feature_fraction来对特征采样
        'bagging_fraction': 0.8,# 通过设定bagging_fraction和bagging_freq来使用 bagging算法
        'bagging_freq': 5,
        'verbose': 0,
        # 'min_data_in_leaf':700,#这是另一个避免leaf-wise tree算法过拟合的重要参数。该值受到训练集数量和num_leaves这两个值的影响。把该参数设的更大能够避免生长出过深的树，但也要避免欠拟合。在分析大型数据集时，该值区间在数百到数千之间较为合适。
        # 'min_sum_hessian_in_leaf' : 1,
        # 特征最大分割
        # 'max_bin':200
        # 'is_unbalance':'true'

    }
    params['metric'] = ['rmse', 'binary_logloss']
    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=1500,
                    valid_sets=lgb_train,
                    # early_stopping_rounds=1300,
                    )

    print('F select...')
    df['importance'] = list(gbm.feature_importance())
    df = df.sort_values(by='importance', ascending=False)
    print(df)
    df = df[df.importance>0]
    feature = df['feature'].values
    print(feature)
    print('特征选择完成。')
    print('Start predicting...')

    print('new F len :%s' % len(feature))
    lgb_train = lgb.Dataset(df_train[feature].values, df_train['goods_num'].values)
    # 训练
    gbm = lgb.train(params,lgb_train,num_boost_round=2000,valid_sets=lgb_train)

    # predict
    preds2 = gbm.predict(df_test[feature].values, num_iteration=gbm.best_iteration)
    # 保存整体结果。
    predict = df_test[['sku_id', 'data_date']]
    predict['goods_num'] = preds2

    print('进行结果周统计。。')
    start_day = 20180501
    end_day = 20180507
    for j in range(1, 6):  # 第j周
        print('第%s周' % j)
        week = predict[(predict.data_date <= end_day) & (predict.data_date >= start_day)]
        week_name = 'week' + str(j)
        temp = week.groupby(['sku_id'])['goods_num'].agg({week_name: np.sum}).reset_index()
        predict = pd.merge(predict, temp, on=['sku_id'], how='left')  #
        start_day = int((datetime.datetime(int(str(start_day)[0:4]), int(str(start_day)[4:6]),int(str(start_day)[6:8])) + datetime.timedelta(days=7)).strftime("%Y%m%d"))
        end_day = int((datetime.datetime(int(str(end_day)[0:4]), int(str(end_day)[4:6]),int(str(end_day)[6:8])) + datetime.timedelta(days=7)).strftime("%Y%m%d"))
        print(start_day)
        print(end_day)
    del predict['data_date']
    del predict['goods_num']
    predict = predict.drop_duplicates(['sku_id'])  # 去重
    predict.to_csv('cy_support_lgb_0911_2.csv', index=None)  # , header=None
    print('over..')
def deal_data():
    goods_user = pd.read_csv('dataset/b/goodsdaily.csv')#商品在用户的表现数据表
    # goods_info = pd.read_csv('goodsinfo.csv')#商品信息表
    goods_sale = pd.read_csv('dataset/b/goodsale.csv')#商品销售数据表
    # goods_sku = pd.read_csv('goods_sku_relation.csv')#商品id和sku对应表
    # goods_promote = pd.read_csv('goods_promote_price.csv')#商品促销价格表
    marketing = pd.read_csv('dataset/b/marketing.csv')#平台活动时间表

    print('区间1.。')
    L1_user = goods_user[(goods_user.data_date>=20170509)&(goods_user.data_date<=20170612)]
    L1_sale = goods_sale[(goods_sale.data_date>=20170509)&(goods_sale.data_date<=20170612)]
    # L1_promote = goods_promote[(goods_promote.data_date >= 20170509) & (goods_promote.data_date <= 20170612)]
    L1_marketing = marketing[(marketing.data_date >= 20170509) & (marketing.data_date <= 20170612)]

    F1_user = goods_user[(goods_user.data_date >= 20170301) & (goods_user.data_date <= 20170324)]
    F1_sale = goods_sale[(goods_sale.data_date >= 20170301) & (goods_sale.data_date <= 20170324)]
    # F1_promote = goods_promote[(goods_promote.data_date >= 20170301) & (goods_promote.data_date <= 20170324)]
    F1_marketing = marketing[(marketing.data_date >= 20170301) & (marketing.data_date <= 20170324)]

    L1_user.to_csv('L1_user.csv', index=None)
    L1_sale.to_csv('L1_sale.csv', index=None)
    # L1_promote.to_csv('L1_promote.csv', index=None)
    L1_marketing.to_csv('L1_marketing.csv', index=None)

    F1_user.to_csv('F1_user.csv', index=None)
    F1_sale.to_csv('F1_sale.csv', index=None)
    # F1_promote.to_csv('F1_promote.csv', index=None)
    F1_marketing.to_csv('F1_marketing.csv', index=None)

    print('区间2.。')
    L1_user = goods_user[(goods_user.data_date >= 20170821) & (goods_user.data_date <= 20170924)]
    L1_sale = goods_sale[(goods_sale.data_date >= 20170821) & (goods_sale.data_date <= 20170924)]
    # L1_promote = goods_promote[(goods_promote.data_date >= 20170821) & (goods_promote.data_date <= 20170924)]
    L1_marketing = marketing[(marketing.data_date >= 20170821) & (marketing.data_date <= 20170924)]

    F1_user = goods_user[(goods_user.data_date >= 20170613) & (goods_user.data_date <= 20170706)]
    F1_sale = goods_sale[(goods_sale.data_date >= 20170613) & (goods_sale.data_date <= 20170706)]
    # F1_promote = goods_promote[(goods_promote.data_date >= 20170613) & (goods_promote.data_date <= 20170706)]
    F1_marketing = marketing[(marketing.data_date >= 20170613) & (marketing.data_date <= 20170706)]

    L1_user.to_csv('L2_user.csv', index=None)
    L1_sale.to_csv('L2_sale.csv', index=None)
    # L1_promote.to_csv('L2_promote.csv', index=None)
    L1_marketing.to_csv('L2_marketing.csv', index=None)

    F1_user.to_csv('F2_user.csv', index=None)
    F1_sale.to_csv('F2_sale.csv', index=None)
    # F1_promote.to_csv('F2_promote.csv', index=None)
    F1_marketing.to_csv('F2_marketing.csv', index=None)

    print('区间3.。')
    L1_user = goods_user[(goods_user.data_date >= 20171123) & (goods_user.data_date <= 20171227)]
    L1_sale = goods_sale[(goods_sale.data_date >= 20171123) & (goods_sale.data_date <= 20171227)]
    # L1_promote = goods_promote[(goods_promote.data_date >= 20171123) & (goods_promote.data_date <= 20171227)]
    L1_marketing = marketing[(marketing.data_date >= 20171123) & (marketing.data_date <= 20171227)]

    F1_user = goods_user[(goods_user.data_date >= 20170925) & (goods_user.data_date <= 20171018)]
    F1_sale = goods_sale[(goods_sale.data_date >= 20170925) & (goods_sale.data_date <= 20171018)]
    # F1_promote = goods_promote[(goods_promote.data_date >= 20170925) & (goods_promote.data_date <= 20171018)]
    F1_marketing = marketing[(marketing.data_date >= 20170925) & (marketing.data_date <= 20171018)]

    L1_user.to_csv('L3_user.csv', index=None)
    L1_sale.to_csv('L3_sale.csv', index=None)
    # L1_promote.to_csv('L3_promote.csv', index=None)
    L1_marketing.to_csv('L3_marketing.csv', index=None)

    F1_user.to_csv('F3_user.csv', index=None)
    F1_sale.to_csv('F3_sale.csv', index=None)
    # F1_promote.to_csv('F3_promote.csv', index=None)
    F1_marketing.to_csv('F3_marketing.csv', index=None)

    print('区间4.。')

    F1_user = goods_user[(goods_user.data_date >= 20180221) & (goods_user.data_date <= 20180316)]
    F1_sale = goods_sale[(goods_sale.data_date >= 20180221) & (goods_sale.data_date <= 20180316)]
    # F1_promote = goods_promote[(goods_promote.data_date >= 20180221) & (goods_promote.data_date <= 20180316)]
    F1_marketing = marketing[(marketing.data_date >= 20180221) & (marketing.data_date <= 20180316)]

    F1_user.to_csv('F4_user.csv', index=None)
    F1_sale.to_csv('F4_sale.csv', index=None)
    # F1_promote.to_csv('F4_promote.csv', index=None)
    F1_marketing.to_csv('F4_marketing.csv', index=None)


    print('OVER..')

def makeData():#构造测试集框架
    relation = pd.read_csv('dataset/b/goods_sku_relation.csv')  # 所有关系映射表
    submit_example = pd.read_csv('dataset/b/submit_example_2.csv')  # 样例表
    submit_example = pd.merge(submit_example, relation, on=['sku_id'], how='left')  #
    del submit_example['week1']
    del submit_example['week2']
    del submit_example['week3']
    del submit_example['week4']
    del submit_example['week5']
    for i in range(1,4):#第i个测试集
        print('第%s个测试集'%i)
        name = 'L'+str(i)+'_sale.csv'
        test = pd.read_csv(name)  # 商品在用户的表现数据表
        days = []
        if i ==1:
            startday = 20170509
        elif i ==2:
            startday = 20170821
        else:
            startday = 20171123
        for k in range(1,36):
            days.append(startday)
            startday = int((datetime.datetime(int(str(startday)[0:4]), int(str(startday)[4:6]),int(str(startday)[6:8])) + datetime.timedelta(days=1)).strftime("%Y%m%d"))
        print(days)
        df_days = pd.DataFrame(days, columns=['data_date'])
        submit_example['temp']=1
        df_days['temp']=1

        result = pd.merge(submit_example, df_days, on=['temp'], how='left')  #
        del result['temp']
        del test['goods_id']
        del test['goods_price']
        del test['orginal_shop_price']
        result = pd.merge(result, test, on=['sku_id','data_date'], how='left')  #

        result.goods_num = result.goods_num.fillna(0)
        name2 = 'test'+str(i)+'.csv'
        result.to_csv(name2, index=None)
    #最后的测试集单独弄
    df_days = pd.DataFrame([20180501,20180502,20180503,20180504,20180505,20180506,20180507,20180508,20180509,20180510,20180511,
                            20180512,20180513,20180514,20180515,20180516,20180517,20180518,20180519,20180520,20180521,20180522,
                            20180523,20180524,20180525,20180526,20180527,20180528,20180529,20180530,20180531,20180601,20180602,
                            20180603,20180604], columns=['data_date'])


    df_days['temp']=1
    submit_example['temp']=1
    test = pd.merge(submit_example, df_days, on=['temp'], how='left')  #
    del test['temp']
    test.to_csv('test4.csv', index=None)
    print(test)


def getF(F1_user,F1_sale,F1_marketing,L,memo):

    goods_info = pd.read_csv('dataset/b/goodsinfo.csv')  # 商品信息
    L = pd.merge(L, goods_info, on=['goods_id'], how='left')  #

    F1_user = pd.merge(F1_user, goods_info, on=['goods_id'], how='left')  #
    F1_sale = pd.merge(F1_sale, goods_info, on=['goods_id'], how='left')  #

    F1_user = pd.merge(F1_user, F1_marketing, on=['data_date'], how='left')  #
    F1_sale = pd.merge(F1_sale, F1_marketing, on=['data_date'], how='left')  #
    # F1_promote = pd.merge(F1_promote, F1_marketing, on=['data_date'], how='left')  #

    print('提取%s特征'%memo)
    print('生成周。。')
    F1_user['week'] = list(map(lambda x: 'week'+str((datetime.datetime(int(str(x)[0:4]), int(str(x)[4:6]), int(str(x)[6:8])).weekday())), F1_user.data_date))
    F1_sale['week'] = list(map(lambda x:  'week'+str((datetime.datetime(int(str(x)[0:4]), int(str(x)[4:6]), int(str(x)[6:8])).weekday())), F1_sale.data_date))
    # F1_promote['week'] = list(map(lambda x: 'week'+str((datetime.datetime(int(str(x)[0:4]), int(str(x)[4:6]), int(str(x)[6:8])).weekday())), F1_promote.data_date))
    F1_marketing['week'] = list(map(lambda x:  'week'+str((datetime.datetime(int(str(x)[0:4]), int(str(x)[4:6]), int(str(x)[6:8])).weekday())), F1_marketing.data_date))
    L['week'] = list(map(lambda x:datetime.datetime(int(str(x)[0:4]), int(str(x)[4:6]), int(str(x)[6:8])).weekday(),L.data_date))
    # L['day_rank'] = L.data_date.rank(ascending=True, method="min")
    print('day_ran...')
    days = list(L.data_date.values)
    days = list(set(days))
    print(days)
    days.sort()
    print(days)
    ins = []
    for i, val in enumerate(days):
        ins.append(i)
    df_index = pd.DataFrame({'data_date':days,'day_rank':ins})
    L = pd.merge(L, df_index, on=['data_date'], how='left')  #
    print('holiday..')
    holiday = pd.read_csv('dataset/other-data/holiday.csv')  #
    # del holiday['holiday']
    L = pd.merge(L, holiday, on=['data_date'], how='left')  #
    print('weather..')
    weather = pd.read_csv('dataset/other-data/weather_ariba.csv')  #
    L = pd.merge(L, weather, on=['data_date'], how='left')  #
    del holiday
    # del weather
    gc.collect()
    print(L)
    # 商品和用户的表现特征
    print('商品和用户的表现特征')
    # 商品被用户总的点击数
    temp = F1_user.groupby(['goods_id'])['goods_click'].agg({'f1': np.sum}).reset_index()
    L = pd.merge(L,temp, on=['goods_id'], how='left')  #
    #
    # 商品被用户总的加购数
    temp = F1_user.groupby(['goods_id'])['cart_click'].agg({'f2': np.sum}).reset_index()
    L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # 商品被用户总的收藏数
    temp = F1_user.groupby(['goods_id'])['favorites_click'].agg({'f3': np.sum}).reset_index()
    L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # 商品总多少用户购买过
    temp = F1_user.groupby(['goods_id'])['sales_uv'].agg({'f4': np.sum}).reset_index()
    L = pd.merge(L, temp, on=['goods_id'], how='left')  #

    # 商品历史上的平均在售天数
    temp = F1_user.groupby(['goods_id'])['onsale_days'].agg({'f5': np.mean}).reset_index()
    L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # 商品被用户平均点击数
    temp = F1_user.groupby(['goods_id'])['goods_click'].agg({'f6': np.mean}).reset_index()
    L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # 商品被用户平均加购数
    temp = F1_user.groupby(['goods_id'])['cart_click'].agg({'f7': np.mean}).reset_index()
    L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # 商品被用户平均收藏数
    temp = F1_user.groupby(['goods_id'])['favorites_click'].agg({'f8': np.mean}).reset_index()
    L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # 商品平均多少用户购买过
    temp = F1_user.groupby(['goods_id'])['sales_uv'].agg({'f9': np.mean}).reset_index()
    L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # print('标准差。。')
    # # 商品被用户点击数方差
    # temp = F1_user.groupby(['goods_id'])['goods_click'].agg({'f10': np.std}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 商品被用户加购数方差
    # temp = F1_user.groupby(['goods_id'])['cart_click'].agg({'f11': np.std}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 商品被用户收藏数方差
    # temp = F1_user.groupby(['goods_id'])['favorites_click'].agg({'f12': np.std}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 商品多少用户购买过方差
    # temp = F1_user.groupby(['goods_id'])['sales_uv'].agg({'f13': np.std}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # print('中位数。。')
    # # 商品被用户点击数中位数
    # temp = F1_user.groupby(['goods_id'])['goods_click'].agg({'f10_1': np.median}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 商品被用户加购数中位数
    # temp = F1_user.groupby(['goods_id'])['cart_click'].agg({'f11_1': np.median}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 商品被用户收藏数中位数
    # temp = F1_user.groupby(['goods_id'])['favorites_click'].agg({'f12_1': np.median}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 商品多少用户购买过中位数
    # temp = F1_user.groupby(['goods_id'])['sales_uv'].agg({'f13_1': np.median}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # print('最大值。。')
    # # 商品被用户点击数max
    # temp = F1_user.groupby(['goods_id'])['goods_click'].agg({'f10_2': np.max}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 商品被用户加购数max
    # temp = F1_user.groupby(['goods_id'])['cart_click'].agg({'f11_2': np.max}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 商品被用户收藏数max
    # temp = F1_user.groupby(['goods_id'])['favorites_click'].agg({'f12_2': np.max}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 商品多少用户购买过max
    # temp = F1_user.groupby(['goods_id'])['sales_uv'].agg({'f13_2': np.max}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # print('最小值。。')
    # # 商品被用户点击数min
    # temp = F1_user.groupby(['goods_id'])['goods_click'].agg({'f10_3': np.min}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 商品被用户加购数min
    # temp = F1_user.groupby(['goods_id'])['cart_click'].agg({'f11_3': np.min}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 商品被用户收藏数min
    # temp = F1_user.groupby(['goods_id'])['favorites_click'].agg({'f12_3': np.min}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 商品多少用户购买过min
    # temp = F1_user.groupby(['goods_id'])['sales_uv'].agg({'f13_3': np.min}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # print('偏度。。')
    # # 商品被用户点击数偏度
    # temp = F1_user.groupby(['goods_id'])['goods_click'].agg({'f10_4': sp.stats.skew}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 商品被用户加购数偏度
    # temp = F1_user.groupby(['goods_id'])['cart_click'].agg({'f11_4':sp.stats.skew}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 商品被用户收藏数偏度
    # temp = F1_user.groupby(['goods_id'])['favorites_click'].agg({'f12_4':sp.stats.skew}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 商品多少用户购买过偏度
    # temp = F1_user.groupby(['goods_id'])['sales_uv'].agg({'f13_4': sp.stats.skew}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # print('峰度。。')
    # # 商品被用户点击数峰度sp.stats.kurtosis
    # temp = F1_user.groupby(['goods_id'])['goods_click'].agg({'f10_5': sp.stats.kurtosis}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 商品被用户加购数峰度
    # temp = F1_user.groupby(['goods_id'])['cart_click'].agg({'f11_5': sp.stats.kurtosis}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 商品被用户收藏数峰度
    # temp = F1_user.groupby(['goods_id'])['favorites_click'].agg({'f12_5': sp.stats.kurtosis}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 商品多少用户购买过峰度
    # temp = F1_user.groupby(['goods_id'])['sales_uv'].agg({'f13_5': sp.stats.kurtosis}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    #
    # print('星期0的各种统计。。')
    # week0 = F1_user[F1_user.week=='week0']
    # # 星期0商品被用户总的点击数
    # temp = week0.groupby(['goods_id'])['goods_click'].agg({'f14': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # week0商品被用户总的加购数
    # temp = week0.groupby(['goods_id'])['cart_click'].agg({'f15': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # week0商品被用户总的收藏数
    # temp = week0.groupby(['goods_id'])['favorites_click'].agg({'f16': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # week0商品总多少用户购买过
    # temp = week0.groupby(['goods_id'])['sales_uv'].agg({'f17': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # print('星期1的各种统计。。')
    # week0 = F1_user[F1_user.week == 'week1']
    # # 星期0商品被用户总的点击数
    # temp = week0.groupby(['goods_id'])['goods_click'].agg({'f18': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # week0商品被用户总的加购数
    # temp = week0.groupby(['goods_id'])['cart_click'].agg({'f19': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # week0商品被用户总的收藏数
    # temp = week0.groupby(['goods_id'])['favorites_click'].agg({'f20': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # week0商品总多少用户购买过
    # temp = week0.groupby(['goods_id'])['sales_uv'].agg({'f21': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # print('星期2的各种统计。。')
    # week0 = F1_user[F1_user.week == 'week2']
    # # 星期0商品被用户总的点击数
    # temp = week0.groupby(['goods_id'])['goods_click'].agg({'f22': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # week0商品被用户总的加购数
    # temp = week0.groupby(['goods_id'])['cart_click'].agg({'f23': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # week0商品被用户总的收藏数
    # temp = week0.groupby(['goods_id'])['favorites_click'].agg({'f24': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # week0商品总多少用户购买过
    # temp = week0.groupby(['goods_id'])['sales_uv'].agg({'f25': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # print('星期3的各种统计。。')
    # week0 = F1_user[F1_user.week == 'week3']
    # # 星期0商品被用户总的点击数
    # temp = week0.groupby(['goods_id'])['goods_click'].agg({'f26': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # week0商品被用户总的加购数
    # temp = week0.groupby(['goods_id'])['cart_click'].agg({'f27': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # week0商品被用户总的收藏数
    # temp = week0.groupby(['goods_id'])['favorites_click'].agg({'f28': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # week0商品总多少用户购买过
    # temp = week0.groupby(['goods_id'])['sales_uv'].agg({'f29': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # print('星期4的各种统计。。')
    # week0 = F1_user[F1_user.week == 'week4']
    # # 星期0商品被用户总的点击数
    # temp = week0.groupby(['goods_id'])['goods_click'].agg({'f30': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # week0商品被用户总的加购数
    # temp = week0.groupby(['goods_id'])['cart_click'].agg({'f31': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # week0商品被用户总的收藏数
    # temp = week0.groupby(['goods_id'])['favorites_click'].agg({'f32': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # week0商品总多少用户购买过
    # temp = week0.groupby(['goods_id'])['sales_uv'].agg({'f33': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # print('星期5的各种统计。。')
    # week0 = F1_user[F1_user.week == 'week5']
    # # 星期0商品被用户总的点击数
    # temp = week0.groupby(['goods_id'])['goods_click'].agg({'f34': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # week0商品被用户总的加购数
    # temp = week0.groupby(['goods_id'])['cart_click'].agg({'f35': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # week0商品被用户总的收藏数
    # temp = week0.groupby(['goods_id'])['favorites_click'].agg({'f36': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # week0商品总多少用户购买过
    # temp = week0.groupby(['goods_id'])['sales_uv'].agg({'f37': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # print('星期6的各种统计。。')
    # week0 = F1_user[F1_user.week == 'week6']
    # # 星期0商品被用户总的点击数
    # temp = week0.groupby(['goods_id'])['goods_click'].agg({'f38': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # week0商品被用户总的加购数
    # temp = week0.groupby(['goods_id'])['cart_click'].agg({'f39': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # week0商品被用户总的收藏数
    # temp = week0.groupby(['goods_id'])['favorites_click'].agg({'f40': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # week0商品总多少用户购买过
    # temp = week0.groupby(['goods_id'])['sales_uv'].agg({'f41': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    #
    # print('历史区间倒数第一周的各种统计')
    # maxday = max(list(F1_user.data_date.values))
    # print('maxday:%s' % maxday)
    # F1_user_oneweek = F1_user[(F1_user.data_date >= int((datetime.datetime(int(str(maxday)[0:4]), int(str(maxday)[4:6]),int(str(maxday)[6:8])) - datetime.timedelta(days=6)).strftime("%Y%m%d"))) & (F1_user.data_date <= maxday)]
    # # 最后一周商品被用户总的点击数
    # temp = F1_user_oneweek.groupby(['goods_id'])['goods_click'].agg({'fw1_1': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # #
    # # 最后一周商品被用户总的加购数
    # temp = F1_user_oneweek.groupby(['goods_id'])['cart_click'].agg({'fw2_1': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 最后一周商品被用户总的收藏数
    # temp = F1_user_oneweek.groupby(['goods_id'])['favorites_click'].agg({'fw3_1': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 最后一周商品总多少用户购买过
    # temp = F1_user_oneweek.groupby(['goods_id'])['sales_uv'].agg({'fw4_1': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    #
    # # # 最后一周商品历史上的平均在售天数
    # # temp = F1_user_oneweek.groupby(['goods_id'])['onsale_days'].agg({'fw5_1': np.mean}).reset_index()
    # # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 最后一周商品被用户平均点击数
    # temp = F1_user_oneweek.groupby(['goods_id'])['goods_click'].agg({'fw6_1': np.mean}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 最后一周商品被用户平均加购数
    # temp = F1_user_oneweek.groupby(['goods_id'])['cart_click'].agg({'fw7_1': np.mean}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 最后一周商品被用户平均收藏数
    # temp = F1_user_oneweek.groupby(['goods_id'])['favorites_click'].agg({'fw8_1': np.mean}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 最后一周商品平均多少用户购买过
    # temp = F1_user_oneweek.groupby(['goods_id'])['sales_uv'].agg({'fw9_1': np.mean}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    #
    # # 最后一周商品被用户点击数方差
    # temp = F1_user_oneweek.groupby(['goods_id'])['goods_click'].agg({'fw10_1': np.std}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 最后一周 商品被用户加购数方差
    # temp = F1_user_oneweek.groupby(['goods_id'])['cart_click'].agg({'fw11_1': np.std}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 最后一周商品被用户收藏数方差
    # temp = F1_user_oneweek.groupby(['goods_id'])['favorites_click'].agg({'fw12_1': np.std}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 最后一周商品多少用户购买过方差
    # temp = F1_user_oneweek.groupby(['goods_id'])['sales_uv'].agg({'fw13_1': np.std}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    #
    # F1_user_oneweek = F1_sale[(F1_sale.data_date >= int((datetime.datetime(int(str(maxday)[0:4]), int(str(maxday)[4:6]),int(str(maxday)[6:8])) - datetime.timedelta(days=6)).strftime("%Y%m%d"))) & (F1_sale.data_date <= maxday)]
    # # 各商品在最后一周总的销售数量
    # temp = F1_user_oneweek.groupby(['goods_id'])['goods_num'].agg({'sw_1': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 各sku_id在最后一周总的销售数量
    # temp = F1_user_oneweek.groupby(['sku_id'])['goods_num'].agg({'sw_2': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['sku_id'], how='left')  #
    # # 各商品在对应sku_id下最后一周总的销售数量
    # temp = F1_user_oneweek.groupby(['goods_id', 'sku_id'])['goods_num'].agg({'sw_3': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id', 'sku_id'], how='left')  #
    # # 各商品在最后一周平均销售数量
    # temp = F1_user_oneweek.groupby(['goods_id'])['goods_num'].agg({'sw_4': np.mean}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 各sku_id在最后一周平均销售数量
    # temp = F1_user_oneweek.groupby(['sku_id'])['goods_num'].agg({'sw_5': np.mean}).reset_index()
    # L = pd.merge(L, temp, on=['sku_id'], how='left')  #
    # # 各商品在对应sku_id下最后一周平均销售数量
    # temp = F1_user_oneweek.groupby(['goods_id', 'sku_id'])['goods_num'].agg({'sw_6': np.mean}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id', 'sku_id'], how='left')  #
    #
    # # 各商品在最后一周平均销售数量sp.stats.skew
    # temp = F1_user_oneweek.groupby(['goods_id'])['goods_num'].agg({'sw_7': sp.stats.skew}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 各sku_id在最后一周平均销售数量
    # temp = F1_user_oneweek.groupby(['sku_id'])['goods_num'].agg({'sw_8': sp.stats.skew}).reset_index()
    # L = pd.merge(L, temp, on=['sku_id'], how='left')  #
    # # 各商品在对应sku_id下最后一周平均销售数量
    # temp = F1_user_oneweek.groupby(['goods_id', 'sku_id'])['goods_num'].agg({'sw_9': sp.stats.skew}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id', 'sku_id'], how='left')  #
    #
    # # 各商品在最后一周平均销售数量sp.stats.kurtosis
    # temp = F1_user_oneweek.groupby(['goods_id'])['goods_num'].agg({'sw_10': sp.stats.kurtosis}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 各sku_id在最后一周平均销售数量
    # temp = F1_user_oneweek.groupby(['sku_id'])['goods_num'].agg({'sw_11': sp.stats.kurtosis}).reset_index()
    # L = pd.merge(L, temp, on=['sku_id'], how='left')  #
    # # 各商品在对应sku_id下最后一周平均销售数量
    # temp = F1_user_oneweek.groupby(['goods_id', 'sku_id'])['goods_num'].agg({'sw_12': sp.stats.kurtosis}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id', 'sku_id'], how='left')  #
    #
    # # 各商品在最后一周std销售数量
    # temp = F1_user_oneweek.groupby(['goods_id'])['goods_num'].agg({'sw_13': np.std}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 各sku_id在最后一周std销售数量
    # temp = F1_user_oneweek.groupby(['sku_id'])['goods_num'].agg({'sw_14': np.std}).reset_index()
    # L = pd.merge(L, temp, on=['sku_id'], how='left')  #
    # # 各商品在对应sku_id下最后一周std销售数量
    # temp = F1_user_oneweek.groupby(['goods_id', 'sku_id'])['goods_num'].agg({'sw_15': np.std}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id', 'sku_id'], how='left')  #
    #
    # # 各商品在最后一周max销售数量
    # temp = F1_user_oneweek.groupby(['goods_id'])['goods_num'].agg({'sw_16': np.max}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 各sku_id在最后一周max销售数量
    # temp = F1_user_oneweek.groupby(['sku_id'])['goods_num'].agg({'sw_17': np.max}).reset_index()
    # L = pd.merge(L, temp, on=['sku_id'], how='left')  #
    # # 各商品在对应sku_id下最后一周max销售数量
    # temp = F1_user_oneweek.groupby(['goods_id', 'sku_id'])['goods_num'].agg({'sw_18': np.max}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id', 'sku_id'], how='left')  #
    #
    # # 各商品在最后一周min销售数量
    # temp = F1_user_oneweek.groupby(['goods_id'])['goods_num'].agg({'sw_19': np.min}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 各sku_id在最后一周min销售数量
    # temp = F1_user_oneweek.groupby(['sku_id'])['goods_num'].agg({'sw_20': np.min}).reset_index()
    # L = pd.merge(L, temp, on=['sku_id'], how='left')  #
    # # 各商品在对应sku_id下最后一周min销售数量
    # temp = F1_user_oneweek.groupby(['goods_id', 'sku_id'])['goods_num'].agg({'sw_21': np.min}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id', 'sku_id'], how='left')  #
    #
    # # 各商品在最后一周median销售数量
    # temp = F1_user_oneweek.groupby(['goods_id'])['goods_num'].agg({'sw_22': np.median}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # # 各sku_id在最后一周median销售数量
    # temp = F1_user_oneweek.groupby(['sku_id'])['goods_num'].agg({'sw_23': np.median}).reset_index()
    # L = pd.merge(L, temp, on=['sku_id'], how='left')  #
    # # 各商品在对应sku_id下最后一周median销售数量
    # temp = F1_user_oneweek.groupby(['goods_id', 'sku_id'])['goods_num'].agg({'sw_24': np.median}).reset_index()
    # L = pd.merge(L, temp, on=['goods_id', 'sku_id'], how='left')  #
    #
    # print('历史区间各天（24天）销量 crosstab')
    #
    # print('历史区间各级类目用户相关数据统计')
    # print('一级类目')
    # # 历史总的被点击次数
    # temp = F1_user.groupby(['cat_level1_id'])['goods_click'].agg({'L1_1': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level1_id'], how='left')  #
    # # 历史总的被加购次数
    # temp = F1_user.groupby(['cat_level1_id'])['cart_click'].agg({'L1_2': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level1_id'], how='left')  #
    # # 历史总的收藏次数
    # temp = F1_user.groupby(['cat_level1_id'])['favorites_click'].agg({'L1_3': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level1_id'], how='left')  #
    # # 历史总的购买人数
    # temp = F1_user.groupby(['cat_level1_id'])['sales_uv'].agg({'L1_4': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level1_id'], how='left')  #
    # print('2级类目')
    # # 历史总的被点击次数
    # temp = F1_user.groupby(['cat_level2_id'])['goods_click'].agg({'L2_1': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level2_id'], how='left')  #
    # # 历史总的被加购次数
    # temp = F1_user.groupby(['cat_level2_id'])['cart_click'].agg({'L2_2': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level2_id'], how='left')  #
    # # 历史总的收藏次数
    # temp = F1_user.groupby(['cat_level2_id'])['favorites_click'].agg({'L2_3': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level2_id'], how='left')  #
    # # 历史总的购买人数
    # temp = F1_user.groupby(['cat_level2_id'])['sales_uv'].agg({'L2_4': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level2_id'], how='left')  #
    #
    # print('3级类目')
    # # 历史总的被点击次数
    # temp = F1_user.groupby(['cat_level3_id'])['goods_click'].agg({'L3_1': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level3_id'], how='left')  #
    # # 历史总的被加购次数
    # temp = F1_user.groupby(['cat_level3_id'])['cart_click'].agg({'L3_2': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level3_id'], how='left')  #
    # # 历史总的收藏次数
    # temp = F1_user.groupby(['cat_level3_id'])['favorites_click'].agg({'L3_3': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level3_id'], how='left')  #
    # # 历史总的购买人数
    # temp = F1_user.groupby(['cat_level3_id'])['sales_uv'].agg({'L3_4': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level3_id'], how='left')  #
    #
    # print('4级类目')
    # # 历史总的被点击次数
    # temp = F1_user.groupby(['cat_level4_id'])['goods_click'].agg({'L4_1': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level4_id'], how='left')  #
    # # 历史总的被加购次数
    # temp = F1_user.groupby(['cat_level4_id'])['cart_click'].agg({'L4_2': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level4_id'], how='left')  #
    # # 历史总的收藏次数
    # temp = F1_user.groupby(['cat_level4_id'])['favorites_click'].agg({'L4_3': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level4_id'], how='left')  #
    # # 历史总的购买人数
    # temp = F1_user.groupby(['cat_level4_id'])['sales_uv'].agg({'L4_4': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level4_id'], how='left')  #
    #
    # print('5级类目')
    # # 历史总的被点击次数
    # temp = F1_user.groupby(['cat_level5_id'])['goods_click'].agg({'L5_1': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level5_id'], how='left')  #
    # # 历史总的被加购次数
    # temp = F1_user.groupby(['cat_level5_id'])['cart_click'].agg({'L5_2': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level5_id'], how='left')  #
    # # 历史总的收藏次数
    # temp = F1_user.groupby(['cat_level5_id'])['favorites_click'].agg({'L5_3': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level5_id'], how='left')  #
    # # 历史总的购买人数
    # temp = F1_user.groupby(['cat_level5_id'])['sales_uv'].agg({'L5_4': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level5_id'], how='left')  #
    #
    # print('6级类目')
    # # 历史总的被点击次数
    # temp = F1_user.groupby(['cat_level6_id'])['goods_click'].agg({'L6_1': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level6_id'], how='left')  #
    # # 历史总的被加购次数
    # temp = F1_user.groupby(['cat_level6_id'])['cart_click'].agg({'L6_2': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level6_id'], how='left')  #
    # # 历史总的收藏次数
    # temp = F1_user.groupby(['cat_level6_id'])['favorites_click'].agg({'L6_3': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level6_id'], how='left')  #
    # # 历史总的购买人数
    # temp = F1_user.groupby(['cat_level6_id'])['sales_uv'].agg({'L6_4': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level6_id'], how='left')  #
    #
    # print('7级类目')
    # # 历史总的被点击次数
    # temp = F1_user.groupby(['cat_level7_id'])['goods_click'].agg({'L7_1': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level7_id'], how='left')  #
    # # 历史总的被加购次数
    # temp = F1_user.groupby(['cat_level7_id'])['cart_click'].agg({'L7_2': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level7_id'], how='left')  #
    # # 历史总的收藏次数
    # temp = F1_user.groupby(['cat_level7_id'])['favorites_click'].agg({'L7_3': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level7_id'], how='left')  #
    # # 历史总的购买人数
    # temp = F1_user.groupby(['cat_level7_id'])['sales_uv'].agg({'L7_4': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level7_id'], how='left')  #
    #
    # print('品牌')
    # # 历史总的被点击次数
    # temp = F1_user.groupby(['brand_id'])['goods_click'].agg({'L8_1': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['brand_id'], how='left')  #
    # # 历史总的被加购次数
    # temp = F1_user.groupby(['brand_id'])['cart_click'].agg({'L8_2': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['brand_id'], how='left')  #
    # # 历史总的收藏次数
    # temp = F1_user.groupby(['brand_id'])['favorites_click'].agg({'L8_3': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['brand_id'], how='left')  #
    # # 历史总的购买人数
    # temp = F1_user.groupby(['brand_id'])['sales_uv'].agg({'L8_4': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['brand_id'], how='left')  #
    #
    # print('历史区间各级类目销量统计')
    # print('一级类目')
    # # 历史总销量
    # temp = F1_sale.groupby(['cat_level1_id'])['goods_num'].agg({'S1_1': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level1_id'], how='left')  #
    #
    # print('2级类目')
    # # 历史总的被点击次数
    # temp = F1_sale.groupby(['cat_level2_id'])['goods_num'].agg({'S2_1': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level2_id'], how='left')  #
    #
    # print('3级类目')
    # # 历史总的被点击次数
    # temp = F1_sale.groupby(['cat_level3_id'])['goods_num'].agg({'S3_1': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level3_id'], how='left')  #
    #
    # print('4级类目')
    # # 历史总的被点击次数
    # temp = F1_sale.groupby(['cat_level4_id'])['goods_num'].agg({'S4_1': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level4_id'], how='left')  #
    #
    # print('5级类目')
    # # 历史总的被点击次数
    # temp = F1_sale.groupby(['cat_level5_id'])['goods_num'].agg({'S5_1': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level5_id'], how='left')  #
    #
    # print('6级类目')
    # # 历史总的被点击次数
    # temp = F1_sale.groupby(['cat_level6_id'])['goods_num'].agg({'S6_1': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level6_id'], how='left')  #
    #
    # print('7级类目')
    # # 历史总的被点击次数
    # temp = F1_sale.groupby(['cat_level7_id'])['goods_num'].agg({'S7_1': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['cat_level7_id'], how='left')  #
    #
    # print('品牌')
    # # 历史总的被点击次数
    # temp = F1_sale.groupby(['brand_id'])['goods_num'].agg({'S8_1': np.sum}).reset_index()
    # L = pd.merge(L, temp, on=['brand_id'], how='left')  #

    print('平台活动类型，活动节奏')
    marketing = pd.read_csv('dataset/b/marketing.csv')
    L = pd.merge(L, marketing, on=['data_date'], how='left')  #

    print('商品销售相关特征')
    # 各商品在历史区间总的销售数量
    temp = F1_sale.groupby(['goods_id'])['goods_num'].agg({'s1': np.sum}).reset_index()
    L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # 各sku_id在历史区间总的销售数量
    temp = F1_sale.groupby(['sku_id'])['goods_num'].agg({'s2': np.sum}).reset_index()
    L = pd.merge(L, temp, on=['sku_id'], how='left')  #
    # 各商品在对应sku_id下历史区间总的销售数量
    temp = F1_sale.groupby(['goods_id', 'sku_id'])['goods_num'].agg({'s3': np.sum}).reset_index()
    L = pd.merge(L, temp, on=['goods_id', 'sku_id'], how='left')  #
    # 各商品在历史区间平均销售数量
    temp = F1_sale.groupby(['goods_id'])['goods_num'].agg({'s4': np.mean}).reset_index()
    L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # 各sku_id在历史区间平均销售数量
    temp = F1_sale.groupby(['sku_id'])['goods_num'].agg({'s5': np.mean}).reset_index()
    L = pd.merge(L, temp, on=['sku_id'], how='left')  #
    # 各商品在对应sku_id下历史区间平均销售数量
    temp = F1_sale.groupby(['goods_id', 'sku_id'])['goods_num'].agg({'s6': np.mean}).reset_index()
    L = pd.merge(L, temp, on=['goods_id', 'sku_id'], how='left')  #
    # 各商品在历史区间平均价格"1,332.00"

    F1_sale['goods_price'] = list(map(lambda x: float(x) if str(x)[1] != ',' else 11, F1_sale.goods_price))

    temp = F1_sale.groupby(['goods_id'])['goods_price'].agg({'s7': np.mean}).reset_index()
    L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # 各商品在各sku下历史区间平均价格
    temp = F1_sale.groupby(['goods_id', 'sku_id'])['goods_price'].agg({'s8': np.mean}).reset_index()
    L = pd.merge(L, temp, on=['goods_id', 'sku_id'], how='left')  #
    # 各商品在历史区间平均吊牌价格
    F1_sale['orginal_shop_price'] = list(
        map(lambda x: float(x) if str(x)[1] != ',' else 11, F1_sale.orginal_shop_price))

    temp = F1_sale.groupby(['goods_id'])['orginal_shop_price'].agg({'s9': np.mean}).reset_index()
    L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # 各商品在各sku下历史区间平均吊牌价格
    temp = F1_sale.groupby(['goods_id', 'sku_id'])['orginal_shop_price'].agg({'s10': np.mean}).reset_index()
    L = pd.merge(L, temp, on=['goods_id', 'sku_id'], how='left')  #

    print('均值方差那些。。')
    # 各商品销售数量sp.stats.skew
    temp = F1_sale.groupby(['goods_id'])['goods_num'].agg({'s11': sp.stats.skew}).reset_index()
    L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # 各sku_id销售数量
    temp = F1_sale.groupby(['sku_id'])['goods_num'].agg({'s12': sp.stats.skew}).reset_index()
    L = pd.merge(L, temp, on=['sku_id'], how='left')  #
    # 各商品在对应sku_id下销售数量
    temp = F1_sale.groupby(['goods_id', 'sku_id'])['goods_num'].agg({'s13': sp.stats.skew}).reset_index()
    L = pd.merge(L, temp, on=['goods_id', 'sku_id'], how='left')  #

    # 各商品销售数量sp.stats.kurtosis
    temp = F1_sale.groupby(['goods_id'])['goods_num'].agg({'s14': sp.stats.kurtosis}).reset_index()
    L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # 各sku_id在最后一周平均销售数量
    temp = F1_sale.groupby(['sku_id'])['goods_num'].agg({'s15': sp.stats.kurtosis}).reset_index()
    L = pd.merge(L, temp, on=['sku_id'], how='left')  #
    # 各商品在对应sku_id下销售数量
    temp = F1_sale.groupby(['goods_id', 'sku_id'])['goods_num'].agg({'s16': sp.stats.kurtosis}).reset_index()
    L = pd.merge(L, temp, on=['goods_id', 'sku_id'], how='left')  #

    # 各商品std销售数量
    temp = F1_sale.groupby(['goods_id'])['goods_num'].agg({'s17': np.std}).reset_index()
    L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # 各sku_idstd销售数量
    temp = F1_sale.groupby(['sku_id'])['goods_num'].agg({'s18': np.std}).reset_index()
    L = pd.merge(L, temp, on=['sku_id'], how='left')  #
    # 各商品在对应sku_id下std销售数量
    temp = F1_sale.groupby(['goods_id', 'sku_id'])['goods_num'].agg({'s19': np.std}).reset_index()
    L = pd.merge(L, temp, on=['goods_id', 'sku_id'], how='left')  #

    # 各商品max销售数量
    temp = F1_sale.groupby(['goods_id'])['goods_num'].agg({'s20': np.max}).reset_index()
    L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # 各sku_idmax销售数量
    temp = F1_sale.groupby(['sku_id'])['goods_num'].agg({'s21': np.max}).reset_index()
    L = pd.merge(L, temp, on=['sku_id'], how='left')  #
    # 各商品在对应sku_id max销售数量
    temp = F1_sale.groupby(['goods_id', 'sku_id'])['goods_num'].agg({'s22': np.max}).reset_index()
    L = pd.merge(L, temp, on=['goods_id', 'sku_id'], how='left')  #

    # 各商品min销售数量
    temp = F1_sale.groupby(['goods_id'])['goods_num'].agg({'s23': np.min}).reset_index()
    L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # 各sku_id min销售数量
    temp = F1_sale.groupby(['sku_id'])['goods_num'].agg({'s24': np.min}).reset_index()
    L = pd.merge(L, temp, on=['sku_id'], how='left')  #
    # 各商品在对应sku_id min销售数量
    temp = F1_sale.groupby(['goods_id', 'sku_id'])['goods_num'].agg({'s25': np.min}).reset_index()
    L = pd.merge(L, temp, on=['goods_id', 'sku_id'], how='left')  #

    # 各商品median销售数量
    temp = F1_sale.groupby(['goods_id'])['goods_num'].agg({'s26': np.median}).reset_index()
    L = pd.merge(L, temp, on=['goods_id'], how='left')  #
    # 各sku_id median销售数量
    temp = F1_sale.groupby(['sku_id'])['goods_num'].agg({'s27': np.median}).reset_index()
    L = pd.merge(L, temp, on=['sku_id'], how='left')  #
    # 各商品在对应sku_id median销售数量
    temp = F1_sale.groupby(['goods_id', 'sku_id'])['goods_num'].agg({'s28': np.median}).reset_index()
    L = pd.merge(L, temp, on=['goods_id', 'sku_id'], how='left')  #


    print(L)
    name = memo+'.csv'
    L.to_csv(name,index=None)


def RH():#模型融合
    data1 = pd.read_csv('cy_lgb_0902_2.csv')
    data2 = pd.read_csv('cy_xgb_0902_2.csv')
    data1.rename(columns={'Probability': 'Probability1'}, inplace=True)
    data2.rename(columns={'Probability': 'Probability2'}, inplace=True)
    temp = data2['Probability2']
    data1 = pd.concat([data1, temp], axis=1)
    data1['rank1'] = data1.groupby(data1['Coupon_id'])['Probability1'].rank(ascending=True, method="average")
    data1['rank2'] = data1.groupby(data1['Coupon_id'])['Probability2'].rank(ascending=True, method="average")

    p1 = data1["rank1"]
    p2 = data1["rank2"]


    data1["Probability"] = list(map(lambda x, y: (0.5 / x + 0.5 / y), p1, p2))
    del data1['rank1']
    del data1['rank2']
    del data1['Probability1']
    del data1['Probability2']

    data1['Probability'] = data1.groupby(data1['Coupon_id'])['Probability'].rank(ascending=False, method="first")
    data1.to_csv('rh_0902_2.csv', index=None)

def GY():
    train1 = pd.read_csv('train1.csv')  #
    train2 = pd.read_csv('train2.csv')  #
    train3 = pd.read_csv('train3.csv')  #
    test = pd.read_csv('test.csv')  #
    print('合并train。。')
    frames = [train1, train2, train3]  # 合并
    train = pd.concat(frames, axis=0)
    train['dis_label'] = 1
    test['dis_label'] = 0
    frames = [train, test]  # 合并
    data = pd.concat(frames, axis=0)
    feature = [x for x in train.columns if x not in ['goods_id', 'sku_id', 'data_date', 'goods_num','dis_label']]
    print('gui yi hua ..')
    k=0
    for i in feature:
        k=k+1
        print(k)
        print(i)
        data[i] = (data[i] - data[i].min()) / (data[i].max() - data[i].min())  # 最大最小归一化
    train = data[data.dis_label==1]
    test = data[data.dis_label==0]
    del train['dis_label']
    del test['dis_label']
    return train,test

def main():
    deal_data()
    makeData()


    print('读1。')
    L1 = pd.read_csv('test1.csv')

    F1_user = pd.read_csv('F1_user.csv')
    F1_sale = pd.read_csv('F1_sale.csv')
    # F1_promote = pd.read_csv('F1_promote.csv')
    F1_marketing = pd.read_csv('F1_marketing.csv')
    getF(F1_user,F1_sale,F1_marketing,L1,'train1')

    print('读2。')
    L1 = pd.read_csv('test2.csv')

    F1_user = pd.read_csv('F2_user.csv')
    F1_sale = pd.read_csv('F2_sale.csv')
    # F1_promote = pd.read_csv('F2_promote.csv')
    F1_marketing = pd.read_csv('F2_marketing.csv')
    getF(F1_user, F1_sale, F1_marketing, L1, 'train2')

    print('读3。')
    L1 = pd.read_csv('test3.csv')

    F1_user = pd.read_csv('F3_user.csv')
    F1_sale = pd.read_csv('F3_sale.csv')
    # F1_promote = pd.read_csv('F3_promote.csv')
    F1_marketing = pd.read_csv('F3_marketing.csv')
    getF(F1_user, F1_sale, F1_marketing, L1, 'train3')

    print('读4。')
    L1 = pd.read_csv('test4.csv')

    F1_user = pd.read_csv('F4_user.csv')
    F1_sale = pd.read_csv('F4_sale.csv')
    # F1_promote = pd.read_csv('F4_promote.csv')
    F1_marketing = pd.read_csv('F4_marketing.csv')
    getF(F1_user, F1_sale, F1_marketing, L1, 'test')

main()
gc.collect()
# GY()
# regressor()#多目标回归
# xgboosts()
# Light_Gbm()
Lr()
print('transform...')
data = pd.read_csv('cy-result.csv')

data['week1_bak'] = data.week1
data['week1'] = data.week5
data['week2_bak'] = data.week2
data['week2'] = data.week4
data['week4'] = data.week2_bak
data['week5'] = data.week1_bak
del data['week1_bak']
del data['week2_bak']
print(data)
data.to_csv('cy-result.csv', index=None)  # , header=None
print('all over!')


