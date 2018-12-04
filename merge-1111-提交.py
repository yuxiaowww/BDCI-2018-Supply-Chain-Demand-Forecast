# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:26:28 2018

@author: yuwei
"""

import pandas as pd



'---------------------------------------------------1-----------------------------------------------------'
'第一步融合：规则+张浩lr更新+张浩lgb'
data1 = pd.read_csv('yw-rule.csv')
data2 = pd.read_csv('zh_lr_v1.csv')
data3 = pd.read_csv('zh_lgb_v1.csv')
data4 = pd.read_csv('zh_lgb_v3.csv')
#对data2归总和
w1 = 296264;w2 = 399559;w3 = 554991;w4 = 641003;w5 = 244596
w1 = w1/sum(data2['week1']);w2 = w2/sum(data2['week2']);w3 = w3/sum(data2['week3']);w4 = w4/sum(data2['week4']);w5 = w5/sum(data2['week5'])
data2['week1'] = data2['week1'].map(lambda x: w1*x)
data2['week2'] = data2['week2'].map(lambda x: w2*x)
data2['week3'] = data2['week3'].map(lambda x: w3*x)
data2['week4'] = data2['week4'].map(lambda x: w4*x)
data2['week5'] = data2['week5'].map(lambda x: w5*x)
#对data3归总和
w1 = 296264;w2 = 399559;w3 = 554991;w4 = 641003;w5 = 244596
w1 = w1/sum(data3['week1']);w2 = w2/sum(data3['week2']);w3 = w3/sum(data3['week3']);w4 = w4/sum(data3['week4']);w5 = w5/sum(data3['week5'])
data3['week1'] = data3['week1'].map(lambda x: w1*x)
data3['week2'] = data3['week2'].map(lambda x: w2*x)
data3['week3'] = data3['week3'].map(lambda x: w3*x)
data3['week4'] = data3['week4'].map(lambda x: w4*x)
data3['week5'] = data3['week5'].map(lambda x: w5*x)
#对data4归总和
w1 = 296264;w2 = 399559;w3 = 554991;w4 = 641003;w5 = 244596
w1 = w1/sum(data4['week1']);w2 = w2/sum(data4['week2']);w3 = w3/sum(data4['week3']);w4 = w4/sum(data4['week4']);w5 = w5/sum(data4['week5'])
data4['week1'] = data4['week1'].map(lambda x: w1*x)
data4['week2'] = data4['week2'].map(lambda x: w2*x)
data4['week3'] = data4['week3'].map(lambda x: w3*x)
data4['week4'] = data4['week4'].map(lambda x: w4*x)
data4['week5'] = data4['week5'].map(lambda x: w5*x)
data1.columns = ['sku_id','week1_1','week2_1','week3_1','week4_1','week5_1']
data2.columns = ['sku_id','week1_2','week2_2','week3_2','week4_2','week5_2']
data3.columns = ['sku_id','week1_3','week2_3','week3_3','week4_3','week5_3']
data4.columns = ['sku_id','week1_4','week2_4','week3_4','week4_4','week5_4']
data = pd.merge(data1,data2,on='sku_id',how='left')
data = pd.merge(data,data3,on='sku_id',how='left')
data = pd.merge(data,data4,on='sku_id',how='left')
data['week1'] = list(map(lambda x,y,z,t: x*0.48+y*0.32+z*0.1+t*0.1 ,data.week1_1,data.week1_2,data.week1_3,data.week1_4))
data['week2'] = list(map(lambda x,y,z,t: x*0.48+y*0.32+z*0.1+t*0.1 ,data.week2_1,data.week2_2,data.week2_3,data.week2_4))
data['week3'] = list(map(lambda x,y,z,t: x*0.48+y*0.32+z*0.1+t*0.1 ,data.week3_1,data.week3_2,data.week3_3,data.week3_4))
data['week4'] = list(map(lambda x,y,z,t: x*0.48+y*0.32+z*0.1+t*0.1 ,data.week4_1,data.week4_2,data.week4_3,data.week4_4))
data['week5'] = list(map(lambda x,y,z,t: x*0.48+y*0.32+z*0.1+t*0.1 ,data.week5_1,data.week5_2,data.week5_3,data.week5_4))
data = data[['sku_id','week1','week2','week3','week4','week5']]
data['week1'] = data['week1'].map(lambda x: 1*x)
data['week2'] = data['week2'].map(lambda x: 1*x)
data['week3'] = data['week3'].map(lambda x: 1*x)
data['week4'] = data['week4'].map(lambda x: 1*x)
data['week5'] = data['week5'].map(lambda x: 1*x)
data.to_csv('merge1.csv',index=False)


'融合：张浩lgb+陈禹更新'
data1 = pd.read_csv('zh_xgb_v2.csv')
data2 = pd.read_csv('cy-result.csv')
data3 = pd.read_csv('zh_xgb_v1.csv')
data4 = pd.read_csv('zh_lgb_v4.csv')
#对data1归总和
w1 = 296264;w2 = 399559;w3 = 554991;w4 = 641003;w5 = 244596
w1 = w1/sum(data1['week1']);w2 = w2/sum(data1['week2']);w3 = w3/sum(data1['week3']);w4 = w4/sum(data1['week4']);w5 = w5/sum(data1['week5'])
data1['week1'] = data1['week1'].map(lambda x: w1*x)
data1['week2'] = data1['week2'].map(lambda x: w2*x)
data1['week3'] = data1['week3'].map(lambda x: w3*x)
data1['week4'] = data1['week4'].map(lambda x: w4*x)
data1['week5'] = data1['week5'].map(lambda x: w5*x)
#对data2归总和
w1 = 296264;w2 = 399559;w3 = 554991;w4 = 641003;w5 = 244596
w1 = w1/sum(data2['week1']);w2 = w2/sum(data2['week2']);w3 = w3/sum(data2['week3']);w4 = w4/sum(data2['week4']);w5 = w5/sum(data2['week5'])
data2['week1'] = data2['week1'].map(lambda x: w1*x)
data2['week2'] = data2['week2'].map(lambda x: w2*x)
data2['week3'] = data2['week3'].map(lambda x: w3*x)
data2['week4'] = data2['week4'].map(lambda x: w4*x)
data2['week5'] = data2['week5'].map(lambda x: w5*x)
#对data3归总和
w1 = 296264;w2 = 399559;w3 = 554991;w4 = 641003;w5 = 244596
w1 = w1/sum(data3['week1']);w2 = w2/sum(data3['week2']);w3 = w3/sum(data3['week3']);w4 = w4/sum(data3['week4']);w5 = w5/sum(data3['week5'])
data3['week1'] = data3['week1'].map(lambda x: w1*x)
data3['week2'] = data3['week2'].map(lambda x: w2*x)
data3['week3'] = data3['week3'].map(lambda x: w3*x)
data3['week4'] = data3['week4'].map(lambda x: w4*x)
data3['week5'] = data3['week5'].map(lambda x: w5*x)
#对data4归总和
w1 = 296264;w2 = 399559;w3 = 554991;w4 = 641003;w5 = 244596
w1 = w1/sum(data4['week1']);w2 = w2/sum(data4['week2']);w3 = w3/sum(data4['week3']);w4 = w4/sum(data4['week4']);w5 = w5/sum(data4['week5'])
data4['week1'] = data4['week1'].map(lambda x: w1*x)
data4['week2'] = data4['week2'].map(lambda x: w2*x)
data4['week3'] = data4['week3'].map(lambda x: w3*x)
data4['week4'] = data4['week4'].map(lambda x: w4*x)
data4['week5'] = data4['week5'].map(lambda x: w5*x)
data1.columns = ['sku_id','week1_1','week2_1','week3_1','week4_1','week5_1']
data2.columns = ['sku_id','week1_2','week2_2','week3_2','week4_2','week5_2']
data3.columns = ['sku_id','week1_3','week2_3','week3_3','week4_3','week5_3']
data4.columns = ['sku_id','week1_4','week2_4','week3_4','week4_4','week5_4']
data = pd.merge(data1,data2,on='sku_id',how='left')
data = pd.merge(data,data3,on='sku_id',how='left')
data = pd.merge(data,data4,on='sku_id',how='left')
data['week1'] = list(map(lambda x,y,z,t: x*0.65+y*0.19+z*0.09+t*0.07 ,data.week1_1,data.week1_2,data.week1_3,data.week1_4))
data['week2'] = list(map(lambda x,y,z,t: x*0.65+y*0.19+z*0.09+t*0.07 ,data.week2_1,data.week2_2,data.week2_3,data.week2_4))
data['week3'] = list(map(lambda x,y,z,t: x*0.65+y*0.19+z*0.09+t*0.07 ,data.week3_1,data.week3_2,data.week3_3,data.week3_4))
data['week4'] = list(map(lambda x,y,z,t: x*0.65+y*0.19+z*0.09+t*0.07 ,data.week4_1,data.week4_2,data.week4_3,data.week4_4))
data['week5'] = list(map(lambda x,y,z,t: x*0.65+y*0.19+z*0.09+t*0.07 ,data.week5_1,data.week5_2,data.week5_3,data.week5_4))
data = data[['sku_id','week1','week2','week3','week4','week5']]
data['week1'] = data['week1'].map(lambda x: 1*x)
data['week2'] = data['week2'].map(lambda x: 1*x)
data['week3'] = data['week3'].map(lambda x: 1*x)
data['week4'] = data['week4'].map(lambda x: 1*x)
data['week5'] = data['week5'].map(lambda x: 1*x)
data.to_csv('merge2.csv',index=False)
'融合：两部分融合结果再融合'
data1 = pd.read_csv('merge1.csv')
data2 = pd.read_csv('merge2.csv')
data1.columns = ['sku_id','week1_1','week2_1','week3_1','week4_1','week5_1']
data2.columns = ['sku_id','week1_2','week2_2','week3_2','week4_2','week5_2']
data = pd.merge(data1,data2,on='sku_id',how='left')
data['week1'] = list(map(lambda x,y: x*0.67+y*0.33 ,data.week1_1,data.week1_2))
data['week2'] = list(map(lambda x,y: x*0.67+y*0.33 ,data.week2_1,data.week2_2))
data['week3'] = list(map(lambda x,y: x*0.67+y*0.33 ,data.week3_1,data.week3_2))
data['week4'] = list(map(lambda x,y: x*0.67+y*0.33 ,data.week4_1,data.week4_2))
data['week5'] = list(map(lambda x,y: x*0.67+y*0.33 ,data.week5_1,data.week5_2))
data = data[['sku_id','week1','week2','week3','week4','week5']]
data['week1'] = data['week1'].map(lambda x: 1*x)
data['week2'] = data['week2'].map(lambda x: 1*x)
data['week3'] = data['week3'].map(lambda x: 1*x)
data['week4'] = data['week4'].map(lambda x: 1*x)
data['week5'] = data['week5'].map(lambda x: 1*x)
data.to_csv('merge3.csv',index=False)

------------------------------------------------------------------------------1
'第二步融合：三个lgb模型的融合+其中lgb更新+xwlr'
data1 = pd.read_csv('zh_lgb_v1.csv')
data2 = pd.read_csv('zh_lgb_v2.csv')
data3 = pd.read_csv('yw-lgb.csv')
data4 = pd.read_csv('zh_lr_v2.csv')
#对data1归总和
data1['week1'] = data1['week1'].map(lambda x: 1.15*x)
data1['week2'] = data1['week2'].map(lambda x: 1.30*x)
data1['week3'] = data1['week3'].map(lambda x: 1.75*x)
data1['week4'] = data1['week4'].map(lambda x: 1.30*x)
data1['week5'] = data1['week5'].map(lambda x: 0.85*x)
#对data2归总和
w1 = 298763;w2 = 396929;w3 = 553596;w4 = 630889;w5 = 242841
w1 = w1/sum(data2['week1']);w2 = w2/sum(data2['week2']);w3 = w3/sum(data2['week3']);w4 = w4/sum(data2['week4']);w5 = w5/sum(data2['week5'])
data2['week1'] = data2['week1'].map(lambda x: w1*x)
data2['week2'] = data2['week2'].map(lambda x: w2*x)
data2['week3'] = data2['week3'].map(lambda x: w3*x)
data2['week4'] = data2['week4'].map(lambda x: w4*x)
data2['week5'] = data2['week5'].map(lambda x: w5*x)
#对data3归总和
w1 = 298763;w2 = 396929;w3 = 553596;w4 = 630889;w5 = 242841
w1 = w1/sum(data3['week1']);w2 = w2/sum(data3['week2']);w3 = w3/sum(data3['week3']);w4 = w4/sum(data3['week4']);w5 = w5/sum(data3['week5'])
data3['week1'] = data3['week1'].map(lambda x: w1*x)
data3['week2'] = data3['week2'].map(lambda x: w2*x)
data3['week3'] = data3['week3'].map(lambda x: w3*x)
data3['week4'] = data3['week4'].map(lambda x: w4*x)
data3['week5'] = data3['week5'].map(lambda x: w5*x)
#对data4归总和
w1 = 298763;w2 = 396929;w3 = 553596;w4 = 630889;w5 = 242841
w1 = w1/sum(data4['week1']);w2 = w2/sum(data4['week2']);w3 = w3/sum(data4['week3']);w4 = w4/sum(data4['week4']);w5 = w5/sum(data4['week5'])
data4['week1'] = data4['week1'].map(lambda x: w1*x)
data4['week2'] = data4['week2'].map(lambda x: w2*x)
data4['week3'] = data4['week3'].map(lambda x: w3*x)
data4['week4'] = data4['week4'].map(lambda x: w4*x)
data4['week5'] = data4['week5'].map(lambda x: w5*x)
data1.columns = ['sku_id','week1_1','week2_1','week3_1','week4_1','week5_1']
data2.columns = ['sku_id','week1_2','week2_2','week3_2','week4_2','week5_2']
data3.columns = ['sku_id','week1_3','week2_3','week3_3','week4_3','week5_3']
data4.columns = ['sku_id','week1_4','week2_4','week3_4','week4_4','week5_4']
data = pd.merge(data1,data2,on='sku_id',how='left')
data = pd.merge(data,data3,on='sku_id',how='left')
data = pd.merge(data,data4,on='sku_id',how='left')
data['week1'] = list(map(lambda x,y,z,t: x*0.11+y*0.01+z*0.01+t*0.87 ,data.week1_1,data.week1_2,data.week1_3,data.week1_4))
data['week2'] = list(map(lambda x,y,z,t: x*0.11+y*0.01+z*0.01+t*0.87 ,data.week2_1,data.week2_2,data.week2_3,data.week2_4))
data['week3'] = list(map(lambda x,y,z,t: x*0.11+y*0.01+z*0.01+t*0.87 ,data.week3_1,data.week3_2,data.week3_3,data.week3_4))
data['week4'] = list(map(lambda x,y,z,t: x*0.11+y*0.01+z*0.01+t*0.87 ,data.week4_1,data.week4_2,data.week4_3,data.week4_4))
data['week5'] = list(map(lambda x,y,z,t: x*0.11+y*0.01+z*0.01+t*0.87 ,data.week5_1,data.week5_2,data.week5_3,data.week5_4))
data = data[['sku_id','week1','week2','week3','week4','week5']]
w1 = 297491;w2 = 398007;w3 = 553967;w4 = 635277;w5 = 243644
w1 = w1/sum(data['week1']);w2 = w2/sum(data['week2']);w3 = w3/sum(data['week3']);w4 = w4/sum(data['week4']);w5 = w5/sum(data['week5'])
data['week1'] = data['week1'].map(lambda x: w1*x)
data['week2'] = data['week2'].map(lambda x: w2*x)
data['week3'] = data['week3'].map(lambda x: w3*x)
data['week4'] = data['week4'].map(lambda x: w4*x)
data['week5'] = data['week5'].map(lambda x: w5*x)
data.to_csv('merge4.csv',index=False)

------------------------------------------------------------------------------2
'最后一步融合：两部分融合结果再融合'
data1 = pd.read_csv('merge3.csv')
data2 = pd.read_csv('merge4.csv')
data1.columns = ['sku_id','week1_1','week2_1','week3_1','week4_1','week5_1']
data2.columns = ['sku_id','week1_2','week2_2','week3_2','week4_2','week5_2']
data = pd.merge(data1,data2,on='sku_id',how='left')
data['week1'] = list(map(lambda x,y: x*0.18+y*0.82 ,data.week1_1,data.week1_2))
data['week2'] = list(map(lambda x,y: x*0.18+y*0.82 ,data.week2_1,data.week2_2))
data['week3'] = list(map(lambda x,y: x*0.18+y*0.82 ,data.week3_1,data.week3_2))
data['week4'] = list(map(lambda x,y: x*0.18+y*0.82 ,data.week4_1,data.week4_2))
data['week5'] = list(map(lambda x,y: x*0.18+y*0.82 ,data.week5_1,data.week5_2))
data = data[['sku_id','week1','week2','week3','week4','week5']]
data['week1'] = data['week1'].map(lambda x: 1.01*x)
data['week2'] = data['week2'].map(lambda x: 1.03*x)
data['week3'] = data['week3'].map(lambda x: 1.05*x)
data['week4'] = data['week4'].map(lambda x: 1.12*x)
data['week5'] = data['week5'].map(lambda x: 1.04*x)
data.to_csv('ans_merge_1111_B_lucky_复现.csv',index=False)














