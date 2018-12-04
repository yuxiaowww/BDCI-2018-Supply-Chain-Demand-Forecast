# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 09:24:45 2018

@author: yuwei
"""

import pandas as pd
import warnings

warnings.filterwarnings("ignore")


path ='dataset//b//'



'目前规则最优'
goodsale = pd.read_csv(path+'goodsale.csv')
submit_example = pd.read_csv(path+'submit_example_2.csv', index_col=None)
goodsale = goodsale[(goodsale['data_date'] >= 20180214) & (goodsale['data_date'] <= 20180316)]
sale_num = goodsale.groupby('sku_id')['goods_num'].sum().reset_index(name='sale_num')
sale_num['sale_num'] /= 4.428
result = pd.DataFrame({'sku_id': submit_example['sku_id']})
result = pd.merge(result, sale_num, on='sku_id', how='left')
w1=290000/sum(result['sale_num'])
w2=400000/sum(result['sale_num'])
w3=550000/sum(result['sale_num'])
w4=600000/sum(result['sale_num'])
w5=240000/sum(result['sale_num'])
result['week1'] = result['sale_num'].apply(lambda x: w1 * x)
result['week2'] = result['sale_num'].apply(lambda x: w2 * 1.02 * x)
result['week3'] = result['sale_num'].apply(lambda x: w3 * 1.02 * x)
result['week4'] = result['sale_num'].apply(lambda x: w4 * 1.12 * x)
result['week5'] = result['sale_num'].apply(lambda x: w5 * 1.04 * x)
del result['sale_num']
result.to_csv('yw-rule.csv', index=False)






