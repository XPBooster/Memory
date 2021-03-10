import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size':20 }) 
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Log等级总开关
formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logfile='Preprocess.log'
outlog=logging.FileHandler(logfile, mode='a')
outlog.setLevel(logging.DEBUG) # 输出到log的等级开关
outlog.setFormatter(formatter)
logger.addHandler(outlog)

screenlog=logging.StreamHandler()
screenlog.setLevel(logging.INFO)
screenlog.setFormatter(formatter)
logger.addHandler(screenlog)

def drop_dups(data, keys):
        
        logging.info('删除数据中的重复条目')
        logging.info('主键：{0}'.format(keys))
        logging.info('去重前数据大小：{0}'.format(data.shape))
        if 'collect_time' in data.columns:
            data_out = data.drop_duplicates(subset=keys, keep='first',inplace=False)
        else:
            data_out = data.drop_duplicates(subset=keys, keep='first',inplace=False)
        logging.info('去重后数据大小：{0}'.format(data_out.shape))

        return data_out

MODE = 'test'

raw_mce = pd.read_csv('memory_sample_mce_log_round1_a_{0}.csv'.format(MODE))        
raw_kernel = pd.read_csv('memory_sample_kernel_log_round1_a_{0}.csv'.format(MODE))
raw_add = pd.read_csv('memory_sample_address_log_round1_a_{0}.csv'.format(MODE))
raw_failure = pd.read_csv('memory_sample_failure_tag_round1_a_train.csv')

mid_mce = drop_dups(raw_mce, ['serial_number','collect_time', 'mca_id'])
mid_add = drop_dups(raw_add, ['collect_time', 'serial_number', 'memory','bankid'])
mid_kernel = drop_dups(raw_kernel, ['collect_time', 'serial_number'])
mid_failure = drop_dups(raw_failure, ['serial_number']) # 没有重复条目，说明每个server只有一次异常

def merge(data_mce, data_add, data_kernel, data_failure):

    logging.info('合并mce和add数据...')
    data = pd.merge(data_mce, data_add, how='left', on=['serial_number','collect_time'], suffixes=['_mce', '_add'], sort=False)

    logging.info('合并mce和kernel数据...')
    data = pd.merge(data, data_kernel, how='left', on=['serial_number','collect_time'], suffixes=['_mce', '_kernel'], sort=False)

    logging.info('创建报错映射...')
    failure_time = dict()
    failure_tag = dict()

    for i in range(data_failure.shape[0]):
        failure_time[raw_failure.loc[i].serial_number] = raw_failure.loc[i].failure_time
        failure_tag[raw_failure.loc[i].serial_number] = raw_failure.loc[i].tag
    
    logging.info('进行报错映射...')

    # for i in range(data.shape[0]):
    #     serial_number = data.loc[i].serial_number
    #     if serial_number in failure_time:
    #         data.loc[i, :].failure_time = failure_time[serial_number]
    #         data.loc[i, :].tag = failure_tag[serial_number]

    def errortime_mapping(item, failure_time):

        serial_number = item.serial_number
        if serial_number in failure_time:
            return failure_time[serial_number]
            

    def errortag_mapping(item, failure_tag):

        serial_number = item.serial_number
        if serial_number in failure_tag:
            return failure_tag[serial_number]
           
    data['failure_time'] = data.apply(errortime_mapping,axis=1,args=(failure_time,))
    data['failure_tag'] = data.apply(errortag_mapping,axis=1,args=(failure_tag,))

    return data

df = merge(mid_mce, mid_add, mid_kernel, mid_failure)

NUM_ROW = 100
NUM_COL = 20
NUM_FORMAT = 20

def process_kernel(df=raw_kernel, save_num=NUM_FORMAT):
    
    def drop_cols(df, save_num): # 保留NULL值最多的日志（时间戳、vendor等没有缺失值的特征不在这个10里边）
    
        non_null = (df.shape[0] - df.isna().sum()).sort_values()
        save_columns = non_null.index[-4-save_num:]
        logging.info('we select the feature set in kernel:{0}'.format(save_columns))
        
        return df[save_columns]

    df = drop_cols(df, save_num)
    
    return df

def process_address(df=raw_kernel, save_num_col=NUM_COL, save_num_row=NUM_ROW):
    
    def drop_cols(data, save_num_row): # 保留NULL值最多的日志（时间戳、vendor等没有缺失值的特征不在这个10里边）
        
        data_out = data
        # 对col做聚合，并按照日志上报的次数从大到小排序 
        df_col = data.groupby(['col']).agg(dict(collect_time='count')).collect_time.sort_values(ascending=False)
        logging.debug('数据按列聚合并将次数倒序排列：{0}'.format(df_col))

        # 选择日志上报最多的前save_num_row个row
        save_cols = df_col.index[:save_num_col]
        logging.info('选择保留的列及其出现频率:{0}'.format(df_col[save_cols]))
        
        # 对不在保留col集合里的，将其统一置为-2
        def row_map(x, save_cols=save_cols):

            if x in save_cols:     
                return x
            else:   
                return -2
            return x

        data_out.loc[:,'col'] = data['col'].map(row_map)
        logging.info('有效列占总数的{0}'.format(df_col[save_cols].values.sum()/data.shape[0]))

        return data_out
        
    
    def drop_rows(data, save_num_col): # 保留NULL值最多的日志（时间戳、vendor等没有缺失值的特征不在这个10里边）

        data_out = data
        # 对row做聚合，并按照日志上报的次数从大到小排序 
        df_row = data.groupby(['row']).agg(dict(collect_time='count')).collect_time.sort_values(ascending=False)
        logging.debug('数据按行聚合并将次数倒序排列：{0}'.format(df_row))

        # 选择日志上报最多的前save_num_row个row
        save_rows = df_row.index[:save_num_row]
        logging.info('选择保留的行及其出现频率:{0}'.format(df_row[save_rows]))
        
        # 对不在保留row集合里的，将其统一置为-1
        def row_map(x, save_rows=save_rows):

            if x in save_rows:     
                return x
            else:   
                return -2
            return x

        data_out.loc[:,'row'] = data['row'].map(row_map)
        logging.info('有效行占总数的{0}'.format(df_row[save_rows].values.sum()/data.shape[0]))
        
        return data_out

    df = drop_cols(df, save_num_col)
    df = drop_rows(df, save_num_row)
    return df

df = process_address(df)
df = process_kernel(df)

df.to_csv('data.csv')

## TODO
# 同类众数补全（ipynb）+多表格数据修正
