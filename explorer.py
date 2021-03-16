import argparse
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os
from pandarallel import pandarallel



def drop_dups(data, keys):

    logging.info('删除数据中的重复条目')
    logging.info('主键：{0}'.format(keys))
    logging.info('去重前数据大小：{0}'.format(data.shape))
    data_out = data.drop_duplicates(
        subset=keys, keep='first', inplace=False)
    logging.info('去重后数据大小：{0}'.format(data_out.shape))

    return data_out


def merge(data_mce, data_add, data_kernel, data_failure):

    '''
    首先按outer join方法对mce,add,kernel数据进行归并
    如果args.mode == 'train'则进行故障映射,即将failure数据
    中的故障标记和故障时间合并到现在的表格里.
    如果args.mode == 'test'则不进行故障映射
    最后输出中间结果_merge.txt
    '''
    def error_mapping(item, failure_time, failure_tag):

        serial_number = item.serial_number
        logging.debug('对服务器{0}进行故障检查'.format(serial_number))
        if serial_number in failure_time:
            logging.info('服务器：{0}；报错时间：{1}；报错类型{2}；'.format(serial_number, failure_time[serial_number], failure_tag[serial_number]))
            return failure_time[serial_number], failure_tag[serial_number]
        return None

    logging.info('合并mce和add数据...')
    data = pd.merge(data_add, data_kernel, how='outer', on=[
                    'serial_number', 'collect_time'], suffixes=['_add', '_kernel'], sort=False)

    logging.info('合并mce和kernel数据...')
    data = pd.merge(data, data_mce, how='outer', on=[
                    'serial_number', 'collect_time'], sort=False)

    if args.mode == 'test':                 # 测试数据没有failure标签，不需要故障映射
        return data

    logging.info('创建故障映射...')
    failuretime_map = dict()
    failuretag_map = dict()

    for i in range(data_failure.shape[0]):  # server到故障的映射关系
        failuretime_map[data_failure.loc[i, 'serial_number']
                        ] = data_failure.loc[i, 'failure_time']
        failuretag_map[data_failure.loc[i, 'serial_number']
                       ] = data_failure.loc[i, 'tag']

    logging.info('进行故障映射...')
    # data[['failure_time', 'failure_tag']] = data.parallel_apply(lambda item: error_mapping(item['serial_number'], failuretime_map, failuretag_map))
    data[['failure_time', 'failure_tag']] = data.parallel_apply(
        error_mapping, axis=1, args=(failuretime_map,failuretag_map,))

    return data


def process_kernel(df, save_num):

    '''
    保留NULL值最少的日志（时间戳、vendor等没有缺失值的特征不在这个save_num里边）
    '''

    def drop_cols(df, save_num):  
        
        non_null = (df.shape[0] - df.isna().sum()).sort_values()
        save_columns = non_null.index[-4-save_num:]
        logging.info(
            '选择kernel数据中的以下特征:{0}'.format(save_columns))

        return df[save_columns]

    df = drop_cols(df, save_num)

    return df


def process_address(df, save_num_col, save_num_row):

    '''
    保留出现次数最多的行值和列值
    其他的置为-2(原数据中-1已经有明确含义)
    '''

    def drop_cols(data, save_num_col):  
        
        data_out = data
        # 对col做聚合，并按照日志上报的次数从大到小排序
        df_col = data.groupby(['col']).agg(
            dict(collect_time='count')).collect_time.sort_values(ascending=False)
        logging.debug('数据按列聚合并将次数倒序排列：{0}'.format(df_col))

        # 选择日志上报最多的前save_num_col个col
        save_cols = df_col.index[:save_num_col]
        logging.info('选择保留的列及其出现频率:{0}'.format(df_col[save_cols]))

        # 对不在保留col集合里的，将其统一置为-2
        def row_map(x, save_cols=save_cols):

            if x in save_cols:
                return x
            else:
                return -2
            return x

        data_out.loc[:, 'col'] = data['col'].map(row_map)
        logging.info('有效列占总数的{0}'.format(
            df_col[save_cols].values.sum()/data.shape[0]))

        return data_out

    def drop_rows(data, save_num_row):  
        
        data_out = data
        # 对row做聚合，并按照日志上报的次数从大到小排序
        df_row = data.groupby(['row']).agg(
            dict(collect_time='count')).collect_time.sort_values(ascending=False)
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

        data_out.loc[:, 'row'] = data['row'].map(row_map)
        logging.info('有效行占总数的{0}'.format(
            df_row[save_rows].values.sum()/data.shape[0]))

        return data_out

    df = drop_cols(df, save_num_col)
    df = drop_rows(df, save_num_row)
    return df


def complete(data):
    '''
    同类众数补全（ipynb）+多表格数据修正
    '''

    def complete_map(data):
        '''
        输入join后的无重数据，根据不同服务器的各指标众数创建服务器:指标映射
        '''

        data_mode = data.groupby('serial_number').agg(
            dict(vendor=pd.Series.mode, manufacturer=pd.Series.mode, mca_id=pd.Series.mode))
        vendor_map, manufacturer_map, mca_id_map = {}, {}, {}
        delete_set = {'manufacturer': [], 'vendor': [], 'mca_id': []}

        for item in data_mode.index:

            # 不同服务器的vendor的众数
            vendor_mode = data_mode.vendor[item]
            if type(vendor_mode) != np.ndarray:  # 不存在众数冲突
                vendor_map[item] = vendor_mode
            else:
                if len(vendor_mode):  # 排除空列表情况
                    vendor_map[item] = vendor_mode[0]
                    delete_set['vendor'].append(item)

            manufacturer_mode = data_mode.manufacturer[item]
            if type(manufacturer_mode) != np.ndarray:
                manufacturer_map[item] = manufacturer_mode
            else:
                if len(manufacturer_mode):
                    manufacturer_map[item] = manufacturer_mode[0]
                    delete_set['manufacturer'].append(item)

            mca_id_mode = data_mode.mca_id[item]
            if type(mca_id_mode) != np.ndarray:             # 22402没有任何mca_id，但是有故障，所以需要单独考虑
                mca_id_map[item] = mca_id_mode
            else:
                if len(mca_id_mode):
                    mca_id_map[item] = mca_id_mode[0]
                    delete_set['mca_id'].append(item)

        logging.info('建立vendor补全映射时，{0}个服务器存在众数冲突，已删去'.format(
            len(delete_set['vendor'])))
        logging.info('建立manufacturer补全映射时，{0}个服务器存在众数冲突，已删去'.format(
            len(delete_set['manufacturer'])))
        logging.info('建立mca_id补全映射时，{0}个服务器存在众数冲突，已删去'.format(
            len(delete_set['mca_id'])))
        # 总共2万个服务器，冲突500个，因此mca_id约有1%的噪声
        # 其中出现服务器故障的有5个：

        return vendor_map, manufacturer_map, mca_id_map, delete_set

    def complete_mca_id(item, mca_id_map):

        '''
        按照mce-add-kernel的优先级依次补全。即：如果mce表中没有缺失，则不处理；如有缺失，则
        用add表中相同服务器-相同上报时间的对应属性值填充；如果add表中仍然没有，则查看kernel表；
        如果都没有，则取相同服务器对应属性的众数
        '''

        if pd.isnull(item.mca_id) == False:
            return item.mca_id
        # if pd.isnull(item.mca_id_add) == False:
        #     logging.info('用add补偿mca_id，值为{0}'.format(item.mca_id_add))
        #     return item.mca_id_add
        # if pd.isnull(item.mca_id_kernel) == False:
        #     logging.info('用kernel补偿mca_id，值为{0}'.format(item.mca_id_kernel))
        #     return item.mca_id_kernel
        if item.serial_number in mca_id_map:
            logging.info('用norm补偿mca_id，值为{0}, 服务器为{1}'.format(
                mca_id_map[item.serial_number], item.serial_number))
            return mca_id_map[item.serial_number]
        else:
            return None

    def complete_vendor(item, vendor_map):

        if pd.isnull(item.vendor) == False:
            return item.vendor
        if pd.isnull(item.vendor_add) == False:
            logging.info('用add补偿vendor，值为{0}, 服务器为{1}'.format(
                item.vendor_add, item.serial_number))
            return item.vendor_add
        if pd.isnull(item.vendor_kernel) == False:
            logging.info('用kernel补偿vendor，值为{0}，服务器为{1}'.format(
                item.vendor_kernel, item.serial_number))
            return item.vendor_kernel
        if item.serial_number in vendor_map:
            logging.info('用norm补偿vendor，值为{0}，服务器为{1}'.format(
                vendor_map[item.serial_number], item.serial_number))
            return vendor_map[item.serial_number]
        else:
            return None

    def complete_manufacturer(item, manufacturer_map):

        if pd.isnull(item.manufacturer) == False:
            return item.manufacturer
        if pd.isnull(item.manufacturer_add) == False:
            logging.info('用add补偿manufacturer，值为{0}，服务器为{1}'.format(
                item.manufacturer_add, item.serial_number))
            return item.manufacturer_add
        if pd.isnull(item.manufacturer_kernel) == False:
            logging.info('用kernel补偿manufacturer，值为{0}，服务器为{1}'.format(
                item.manufacturer_kernel, item.serial_number))
            return item.manufacturer_kernel
        if item.serial_number in manufacturer_map:
            logging.info('用norm补偿vendor，值为{0}，服务器为{1}'.format(
                manufacturer_map[item.serial_number], item.serial_number))
            return manufacturer_map[item.serial_number]
        else:
            return None

    logging.info('创建补全映射')
    vendor_map, manufacturer_map, mca_id_map, delete_set = complete_map(data)
    logging.info('进行补全映射')
    data['mca_id'] = data.parallel_apply(
        complete_mca_id, axis=1, args=(mca_id_map,))
    data['vendor'] = data.parallel_apply(
        complete_vendor, axis=1, args=(vendor_map,))
    data['manufacturer'] = data.parallel_apply(
        complete_manufacturer, axis=1, args=(manufacturer_map,))

    return data


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Data Preprocess')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--num_row', type=int, default=200)
    parser.add_argument('--num_col', type=int, default=20)
    parser.add_argument('--num_format', type=int, default=20)
    args = parser.parse_args()

    pandarallel.initialize()  # pandas并行加速数据处理
    matplotlib.rcParams.update({'font.size': 20})
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    formatter = logging.Formatter(
        '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    logfile = 'Preprocess_{0}.log'.format(args.mode)
    outlog = logging.FileHandler(logfile, mode='a')
    outlog.setLevel(logging.DEBUG)  # 输出到log的等级开关
    outlog.setFormatter(formatter)
    logger.addHandler(outlog)

    screenlog = logging.StreamHandler()
    screenlog.setLevel(logging.INFO)
    screenlog.setFormatter(formatter)
    logger.addHandler(screenlog)

    if not 'data_{0}_merge.csv'.format(args.mode) in os.listdir():

        data_mce = pd.read_csv(
            'memory_sample_mce_log_round1_a_{0}.csv'.format(args.mode))
        data_kernel = pd.read_csv(
            'memory_sample_kernel_log_round1_a_{0}.csv'.format(args.mode))
        data_add = pd.read_csv(
            'memory_sample_address_log_round1_a_{0}.csv'.format(args.mode))
        data_failure = pd.read_csv(
            'memory_sample_failure_tag_round1_a_train.csv')

        data_mce = drop_dups(
            data_mce, ['serial_number', 'collect_time', 'mca_id'])
        data_add = drop_dups(
            data_add, ['collect_time', 'serial_number', 'memory', 'bankid'])
        data_kernel = drop_dups(data_kernel, ['collect_time', 'serial_number'])
        # 没有重复条目，说明每个server只有一次异常
        data_failure = drop_dups(data_failure, ['serial_number'])

        data = merge(data_mce, data_add, data_kernel, data_failure)
        data.to_csv('data_{0}_merge.csv'.format(args.mode))

    data = pd.read_csv('data_{0}_merge.csv'.format(args.mode))
    data = complete(data)
    data.drop(labels=['manufacturer_add', 'vendor_add',
                      'manufacturer_kernel', 'vendor_kernel'], axis=1, inplace=True)  # 删去冗余补全行
    data = process_address(data, save_num_col=args.num_col,
                           save_num_row=args.num_row)  # 处理address数据，主要是内存地址的稀疏化
    data = process_kernel(data, save_num=args.num_format)
    data = drop_dups(data, ['serial_number', 'collect_time','mca_id','row', 'col', 'rankid', 'memory']) # 最后去重
    data.to_csv('data_{0}_final.csv'.format(args.mode))
