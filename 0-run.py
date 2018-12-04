# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 12:23:47 2018

@author: yuwei
"""

"说明-代码执行文件"

#----简单介绍----
'大力出奇迹
小巧拨千斤'
'1、运行此python文件即可生成最终结果.'
'2、每个模型效率挺高,但因为模型较多,因此运行总时间偏长.'
'3、文件名：ans_merge_1111_B_lucky_复现.csv 即为最终线上提交结果.'
'4、lgb模型和lr模型执行较快,xgb运行时间较长.'
'5、模型多,首先是因为我们人多,其次是队友给力:有差异的滑窗/有差异的特征/有差异的模型,相当于做了加强版的模型融合.'
'6、会产生一些中间结果文件,不过没关系，找到生成的这份结果文件就好：ans_merge_1111_B_lucky_复现.csv'
#----简单介绍----

import os

print('正在执行zh的模型1: lr_v1 ')
os.system("python ./zh_lr_v1.py")
print('执行结束!')

print('正在执行zh的模型2: lr_v2 ')
os.system("python ./zh_lr_v2.py")
print('执行结束!')

print('正在执行yw的模型3...')
os.system("python ./yw-lgb.py")
print('执行结束!')

print('正在执行yw的模型4...')
os.system("python ./yw-rule.py")
print('执行结束!\n')

print('正在执行cy的模型5...')
os.system("python ./cy-lr.py")
print('执行结束!')

print('正在执行zh的模型6: lgb_v1 ')
os.system("python ./zh_lgb_v1.py")
print('执行结束!')

print('正在执行zh的模型7: lgb_v2 ')
os.system("python ./zh_lgb_v2.py")
print('执行结束!')

print('正在执行zh的模型8: lgb_v3 ')
os.system("python ./zh_lgb_v3.py")
print('执行结束!')

print('正在执行zh的模型9: lgb_v4 ')
os.system("python ./zh_lgb_v4.py")
print('执行结束!')

print('正在执行zh的模型10: xgb_v1 ')
os.system("python ./zh_xgb_v1.py")
print('执行结束!')

print('正在执行zh的模型11: xgb_v2 ')
os.system("python ./zh_xgb_v2.py")
print('执行结束!')

print('正在执行最终的模型融合... ')
os.system("python ./merge-1111-提交.py")
print('恭喜你,执行全部结束!\n')
