# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 20:18:59 2021

@author: summer@University of Cincinnati
"""
import treetaggerwrapper
import pandas as pd
import numpy as np
import os

from matplotlib import rcParams
params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'normal',
        'font.weight':'normal',
        'font.size':16
        }
rcParams.update(params)

# studied bigram patterns in this study
bigram_patterns = [['J', 'I'], ['V', 'I'], ['N', 'I'], ['V', 'R'], ['J', 'N'],
                   ['R', 'J'], ['R', 'V'], ['N', 'V'], ['V', 'N'], ['N', 'N'],
                   ['I', 'N']]

levels = pd.read_excel('YELC_2011.xlsx')
levels.set_index('Student ID', drop=True, inplace=True)
path = 'D:\\BB\\New folder\\TreeTagger'
tagger = treetaggerwrapper.TreeTagger(TAGLANG='en', TAGDIR=path)

summary_list = []
path = 'D:\\BB\\New folder\\demo\A2\\'
excel_files = os.listdir(path + 'result\\')

def normalize_entropy(x):
    # base is e
    x = x / np.sum(x)
    n = len(x)
    h = -np.sum([p * np.log(p) / np.log(n) for p in x])
    
    return h


def str_to_float(x):
    if ',' in x:
        return float(x.replace(',', '.'))
    else:
        return float(x)
    

def tag_identificaiton(tags):
    num = len(tags)
    bigram_raw = []
    bigram_root = []
    bigram_relation = []
    for ii in range(num-1):
        tag_split = tags[ii].split('\t')
        if len(tag_split) == 3:
            start_word_raw, start_tag, start_word_root = tag_split
        else: # take outlier as the ending of a sentence
            start_word_raw, start_tag, start_word_root = ['.', 'SENT', '.']
        tag_split = tags[ii+1].split('\t')
        if len(tag_split) == 3:
            end_word_raw, end_tag, end_word_root = tag_split
        else:
            end_word_raw, end_tag, end_word_root = ['.', 'SENT', '.']
        if [start_tag[0], end_tag[0]] in bigram_patterns:
            bigram_raw.append(start_word_raw + ' ' + end_word_raw)
            bigram_root.append(start_word_root + ' ' + end_word_root)
            bigram_relation.append(start_tag[0] + '_' + end_tag[0])
    df = pd.DataFrame(index=['root', 'relation'], columns=bigram_raw, data=[bigram_root, bigram_relation]).T
    return df
            

for ii, excel_file in enumerate(excel_files):
    print(ii)
    ID = excel_file.split('.')[0].split('_')[0]
    excel_rst = pd.read_excel(path + 'result\\' + excel_file, sheet_name='calculations')
    excel_rst.set_index('bigram types', drop=True, inplace=True)
    excel_rst.dropna(axis=0, how='any', inplace=True)
    excel_rst = excel_rst[['freq in text', 'freq in COCA', 'MI', 't']]
    excel_rst['MI'] = excel_rst['MI'].apply(str_to_float)
    excel_rst['t'] = excel_rst['t'].apply(str_to_float)
    txt_file = path + 'text\\' + excel_file.split('.')[0] + '.txt'
    with open(txt_file, 'r') as file:
        txt_str = file.read().replace('\n', ' ')
    tags = tagger.tag_text(txt_str)
    relation_df = tag_identificaiton(tags)
    tmp_merge_df = excel_rst.merge(relation_df, left_index=True, right_index=True)
    tmp_merge_df['Student ID'] = ID
    tmp_merge_df['level'] = levels.loc[int(ID)]['Grade']
    tmp_merge_df.drop_duplicates(inplace=True)
    if ii == 0:
        merge_df = tmp_merge_df
    else:
        merge_df = merge_df.append(tmp_merge_df)
        
merge_df.to_csv('data_summary.csv')


#%% summarize information for each  student
data = pd.read_csv('D:\\BB\\New folder\\demo\\data_summary.csv', index_col='bigrams')
unique_relation = data['relation'].unique()
unique_student = data['Student ID'].unique()
multi_index = pd.MultiIndex.from_product([unique_student, unique_relation])
summary_per_student = pd.DataFrame(index=multi_index, columns=['level', 't', 'MI', 'w_t', 'w_MI', 'text_f_sum', 
                                                               'coca_f_mean', 'coca_f_mean_w', 'TTR', 'entropy'])
for ii, student in enumerate(unique_student):
    print("{} of {} has been processed!".format(ii, len(unique_student)))
    student_data = data.loc[data['Student ID'] == student]
    for relation in unique_relation:
        summary_per_student.loc[(student, relation)]['level'] = student_data.iloc[0]['level']
        if relation in student_data['relation'].to_list():
            tmp_data = student_data[student_data['relation'] == relation]
            summary_per_student.loc[(student, relation)]['t'] = tmp_data['t'].mean()
            summary_per_student.loc[(student, relation)]['MI'] = tmp_data['MI'].mean()
            summary_per_student.loc[(student, relation)]['w_t'] = np.dot(tmp_data['freq in text'], tmp_data['t']).mean() / tmp_data['freq in text'].sum()
            summary_per_student.loc[(student, relation)]['w_MI'] = np.dot(tmp_data['freq in text'], tmp_data['MI']).mean() / tmp_data['freq in text'].sum()
            summary_per_student.loc[(student, relation)]['text_f_sum'] = tmp_data['freq in text'].sum()
            summary_per_student.loc[(student, relation)]['coca_f_mean'] = tmp_data['freq in COCA'].mean()
            summary_per_student.loc[(student, relation)]['coca_f_mean_w'] = \
                np.sum(tmp_data['freq in text'].values * tmp_data['freq in COCA'].values) / tmp_data['freq in text'].sum()
            summary_per_student.loc[(student, relation)]['TTR'] = \
                tmp_data['freq in text'].shape[0] / tmp_data['freq in text'].sum()
            summary_per_student.loc[(student, relation)]['entropy'] = normalize_entropy(tmp_data['freq in text'])
        else:
            summary_per_student.loc[(student, relation)]['t'] = np.nan
            summary_per_student.loc[(student, relation)]['MI'] = np.nan
            summary_per_student.loc[(student, relation)]['w_t'] = np.nan
            summary_per_student.loc[(student, relation)]['w_MI'] = np.nan
            summary_per_student.loc[(student, relation)]['text_f_sum'] = np.nan
            summary_per_student.loc[(student, relation)]['COCA_f_mean'] = np.nan
            summary_per_student.loc[(student, relation)]['coca_f_mean_w'] = np.nan
            summary_per_student.loc[(student, relation)]['entropy'] = np.nan
            summary_per_student.loc[(student, relation)]['TTR'] = np.nan
summary_per_student.to_csv('D:\\BB\\New folder\\demo\\summary_per_student.csv')
summary_per_student['entropy'].hist(bins=100)

#%% summarize data for each level
data_summary = pd.read_csv('data_summary.csv', index_col='Student ID')
data = pd.read_csv('summary_per_student.csv', index_col='student ID')
levels = ['A2', 'B1', 'B2']
for level in levels:
    level_data = data.loc[data['level'] == level]
    ids = list(level_data.index.unique())
    idx = np.arange(len(ids))
    np.random.shuffle(idx)
    ids = [ids[ii] for ii in idx][:450]
    for ii, id in enumerate(ids):
        if ii == 0:
            level_data = data.loc[id][:]
            new_data_summary = data_summary.loc[id][:]
        else:
            level_data = level_data.append(data.loc[id][:])
            new_data_summary = new_data_summary.append(data_summary.loc[id][:])
    print(len(level_data.index.unique()))
    level_data.to_csv(level + 'per_student.csv')
    new_data_summary.to_csv(level + 'data_summary.csv')

