#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Basic Library
import os
import glob
import math
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import random as rd
import time
from tqdm import tqdm
import json
import warnings
import pickle
warnings.simplefilter(action='ignore', category=FutureWarning)

# ML Library
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

# Custormized Package
from tools.cleansing import *
from tools.calculate import calculate_position_change

# For API
import requests
from urllib import parse

def searching(api_key, nickname, tag, num=3):
    if (type(num)!=int) & (num <= 5):
        raise('게임 수는 5개 안의 정수로만 입력하세요')
        
    print('데이터를 불러오는 중입니다. 잠시 기다려주세요')
        
    REQUEST_HEADERS={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Whale/3.26.244.21 Safari/537.36",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": "https://developer.riotgames.com",
        "X-Riot-Token": api_key
    }
    
    # puuid 로드
    encodedName = parse.quote(nickname)
    puuid_url = f"https://asia.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{encodedName}/{tag}"
    puuid_response = requests.get(puuid_url, headers=REQUEST_HEADERS)
    if puuid_response.status_code == 200:
        puuid_data = puuid_response.json()
        puuid = puuid_data['puuid']
    elif puuid_response.status_code == 400:
        raise Exception('올바른 닉네임과 태그가 아닙니다')
    elif puuid_response.status_code == 403:
        raise Exception('API 키가 올바르지 않습니다')
    elif puuid_response.status_code == 404:
        raise Exception('일치하는 데이터가 없습니다')
    elif puuid_response.status_code == 503:
        raise Exception('Riot 서버 문제 발생')
    else:
        raise Exception('에러 발생')
    
    # match id 로드
    start = 0
    count = num
    
    matchid_url = f"https://asia.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?type=ranked&start={start}&count={count}"
    matchid_response = requests.get(matchid_url, headers=REQUEST_HEADERS)
    if matchid_response.status_code == 200:
        matchid_list = matchid_response.json()
    elif matchid_response.status_code == 400:
        raise Exception('올바른 닉네임과 태그가 아닙니다')
    elif matchid_response.status_code == 403:
        raise Exception('API 키가 올바르지 않습니다')
    elif matchid_response.status_code == 404:
        raise Exception('일치하는 데이터가 없습니다')
    elif matchid_response.status_code == 503:
        raise Exception('Riot 서버 문제 발생')
    else:
        raise Exception('에러 발생')
    
    # 이벤트, 분당지표, 위치 정보 데이터 추출
    event = pd.DataFrame()
    participant = pd.DataFrame()
    position = pd.DataFrame()
    for match_id in matchid_list:
        time_url = f'https://asia.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline'
        time_response = requests.get(time_url, headers=REQUEST_HEADERS)
        
        if time_response.status_code == 200:
            time_data = time_response.json()
            match_timeline=pd.DataFrame(time_data['info']['frames'])
            match_timeline.insert(0,'match_id',match_id)
            match_timeline=match_timeline.drop(match_timeline.index[-1])
            result = match_timeline.apply(for_timeline, axis=1)
            
            event_list, participant_list, position_list = zip(*result)
            event_small = pd.concat(event_list).reset_index(drop=True)
            participant_small = pd.concat(participant_list).reset_index(drop=True)
            position_small = pd.concat(position_list).reset_index(drop=True)
            
            if event.empty:
                event=event_small.copy()
            else:
                event=pd.concat([event,event_small]).reset_index(drop=True)

            if participant.empty:
                participant=participant_small.copy()
            else:
                participant=pd.concat([participant,participant_small]).reset_index(drop=True)
                
            if position.empty:
                position=position_small.copy()
            else:
                position=pd.concat([position,position_small]).reset_index(drop=True)
                
        elif time_response.status_code == 400:
            raise Exception('올바른 닉네임과 태그가 아닙니다')
        elif time_response.status_code == 403:
            raise Exception('API 키가 올바르지 않습니다')
        elif time_response.status_code == 404:
            raise Exception('일치하는 데이터가 없습니다')
        elif time_response.status_code == 503:
            raise Exception('Riot 서버 문제 발생')
        else:
            raise Exception('에러 발생')

    event.to_csv(f'./st_data/event/event1.csv',index=False)
    participant.to_csv(f'./st_data/participant/participant1.csv',index=False)
    position.to_csv(f'./st_data/position/position1.csv',index=False)
    
    # 경기 결과, 오브젝트 데이터 추출
    match_info = pd.DataFrame()
    objectives = pd.DataFrame()
    for match_id in matchid_list:
        match_url = f'https://asia.api.riotgames.com/lol/match/v5/matches/{match_id}'
        match_response = requests.get(match_url, headers=REQUEST_HEADERS)
        
        if match_response.status_code == 200:
            match_data = match_response.json()
            result = for_match_info(match_data['info'])
            match_info_small, objectives_small = result
            
            match_info_small.insert(0, 'match_id', match_id)
            objectives_small.insert(0, 'match_id', match_id)
            
            summmonerId_list = list(match_info_small['summonerId'].unique())
            match_info_small.insert(5,'tier',np.NaN)
            match_info_small.insert(6,'rank',np.NaN)
            for i in range(0,len(summmonerId_list),2):
                tier_url = f'https://kr.api.riotgames.com/lol/league/v4/entries/by-summoner/{summmonerId_list[i]}'
                tier_response = requests.get(tier_url, headers=REQUEST_HEADERS)
                if tier_response.status_code == 200:
                    league_entries = tier_response.json()
                    for entry in league_entries:
                        if entry['queueType'] == 'RANKED_SOLO_5x5':
                            tier = entry['tier']
                            rank = entry['rank']
                    match_info_small.loc[i,'tier'] = tier
                    match_info_small.loc[i,'rank'] = rank
                elif tier_response.status_code == 400:
                    raise('올바른 닉네임과 태그가 아닙니다')
                elif tier_response.status_code == 403:
                    raise('API 키가 올바르지 않습니다')
                elif tier_response.status_code == 404:
                    raise('일치하는 데이터가 없습니다')
                elif tier_response.status_code == 503:
                    raise('Riot 서버 문제 발생')
                else:
                    raise('에러 발생')
            
            tier_mapping = {
                'IRON': 1,
                'BRONZE': 2,
                'SILVER': 3,
                'GOLD': 4,
                'PLATINUM': 5,
                'EMERALD': 6,
                'DIAMOND': 7,
                'MASTER': 8,
                'GRANDMASTER': 9,
                'CHALLENGER': 10
            }
            match_info_small['tier'] = match_info_small['tier'].map(tier_mapping)

            rank_mapping = {
                'I': 0,
                'II': 0.25,
                'III': 0.5,
                'IV': 0.75
            }
            match_info_small['tier'] = match_info_small['tier'] + match_info_small['rank'].map(rank_mapping)
            match_info_small.drop(columns=['rank'], inplace=True)
            tier_mean = round(match_info_small['tier'].mean(), 2)
            match_info_small['tier'] = tier_mean
            
            if match_info.empty:
                match_info = match_info_small.copy()
            else:
                match_info = pd.concat([match_info, match_info_small]).reset_index(drop=True)

            if objectives.empty:
                objectives = objectives_small.copy()
            else:
                objectives = pd.concat([objectives, objectives_small]).reset_index(drop=True)
            
        elif match_response.status_code == 400:
            raise Exception('올바른 닉네임과 태그가 아닙니다')
        elif match_response.status_code == 403:
            raise Exception('API 키가 올바르지 않습니다')
        elif match_response.status_code == 404:
            raise Exception('일치하는 데이터가 없습니다')
        elif match_response.status_code == 503:
            raise Exception('Riot 서버 문제 발생')
        else:
            raise Exception('에러 발생')
    
    match_info.to_csv(f'./st_data/match_info/match_info1.csv', index=False)
    objectives.to_csv(f'./st_data/objectives/objectives1.csv', index=False)
    
    # 챔피언 데이터 로드
    champion_url = 'https://ddragon.leagueoflegends.com/cdn/14.24.1/data/en_US/champion.json'
    champion_response = requests.get(champion_url, headers=REQUEST_HEADERS)
    
    if matchid_response.status_code == 200:
        champion_data = champion_response.json()
        champion = pd.DataFrame(champion_data['data']).T.reset_index()
        champion = champion.rename(columns={'index':'championName'})
        
        champ_role = pd.DataFrame(champion['tags'].tolist(), columns=['mainRole','subRole'])
        champion = pd.concat([champion['championName'],champ_role], axis=1)
    elif matchid_response.status_code == 400:
        raise Exception('올바른 닉네임과 태그가 아닙니다')
    elif matchid_response.status_code == 403:
        raise Exception('API 키가 올바르지 않습니다')
    elif matchid_response.status_code == 404:
        raise Exception('일치하는 데이터가 없습니다')
    elif matchid_response.status_code == 503:
        raise Exception('Riot 서버 문제 발생')
    else:
        raise Exception('에러 발생')
    
    champion.to_csv(f'./st_data/champion.csv',index=False)
    
def anomaly_detection(df):
    
    def first_preprocess(df):
        columns_to_calculate = ['cs', 'goldSpent', 'champExperience', 'totalDamageTaken','deaths',
            'totalDamageDealtToChampions', 'damageDealtToObjectives', 'visionScore', 'jungle_kill']

        for col in columns_to_calculate:
            df[col] = df[col] / (df['timePlayed'])
        df = df.drop('timePlayed', axis=1)

        def item_dup(x):
            if x==0:
                return 7
            else:
                return x

        df['item_duplication'] = df['item_duplication'].apply(item_dup)

        def item_anomal(x):
            if x>=4:
                return True
            else:
                return False

        df['item_anomal'] = df['item_duplication'].apply(item_anomal)
        df = df.drop('item_duplication', axis=1)

        return df

    def second_preprocess(df):
        df = df[(df['afk']==False) & (df['item_anomal']==False)].reset_index(drop=True)
        df = df.drop(['afk','item_anomal'], axis=1)

        df = df.drop(['match_id','teamId','puuid','summonerName'],axis=1)
        df = df.drop(['goldEarned','death_ratio','championName',
                      'item0','item1','item2','item3','item4','item5', 
                      'summoner1Id', 'summoner2Id'],axis=1)

        scaler = RobustScaler()

        # 스케일링할 피처 선택 (여기서는 예시로 모든 수치형 피처를 스케일링)
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        columns_to_scale = [col for col in numeric_columns if col != 'teamId']
        # RobustScaler를 사용하여 피처 스케일링
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

        # 서브롤이 '-'인 경우 'Only'로 대체
        df['subRole'] = df['subRole'].replace('-', 'Only')

        # 모든 가능한 조합을 생성 (자기 자신을 제외 또는 'Only'로 표시된 경우)
        main_roles = ['Fighter', 'Assassin','Marksman', 'Tank', 'Mage', 'Support']
        sub_roles = ['Only', 'Fighter','Mage', 'Support', 'Assassin', 'Tank', 'Marksman']

        # 자기 자신을 제외한 모든 조합의 리스트를 생성
        all_combinations = [f"{mr}_{sr}" for mr in main_roles for sr in sub_roles if mr != sr or sr == 'Only']

        # 기존 조합 리스트 생성
        combine_list = [f"{a}_{b}" for a, b in zip(df['mainRole'], df['subRole'])]

        # 모든 조합을 False로 초기화한 DataFrame 생성
        sr = pd.DataFrame(False, index=np.arange(len(df)), columns=all_combinations)

        # 실제 존재하는 조합에 대해 True로 설정
        for comb in combine_list:
            if comb in sr.columns:
                sr.loc[:, comb] = True

        # 기존 df와 결합
        df = pd.concat([df, sr], axis=1).drop(['mainRole', 'subRole'], axis=1)

        df = pd.get_dummies(df, columns=['teamPosition'])

        return df

    def isf_threshold(df):
        high_threshold = -0.01
        low_threshold = -999

        df['isf_score_anomaly'] = (df['isf_score']<high_threshold)&(df['isf_score']>=low_threshold)

        return df

    def ocsvm_threshold(df):
        high_threshold = -3
        low_threshold = -9999999

        df['ocsvm_score_anomaly'] = (df['ocsvm_score']<high_threshold)&(df['ocsvm_score']>=low_threshold)

        return df

    def voting(df):
        cluster_df = df.copy()

        cluster_df.insert(1,'anomal',1)

        cluster_df['anomal'] = cluster_df.apply(
            lambda row: True if row['ocsvm_score_anomaly'] == True or row['isf_score_anomaly'] == True else False,
            axis=1
        )

        cluster_df = cluster_df.drop(['ocsvm_score_anomaly','isf_score_anomaly'],axis=1)
        return cluster_df

    def anomal_to_normal(df):
        columns_to_check = ['kill_involve_ratio', 'tower_involve_ratio', 'object_involve_ratio','kda']
        thresholds = {}

        Q1 = df[columns_to_check].quantile(0.25)
        Q3 = df[columns_to_check].quantile(0.75)
        IQR = Q3 - Q1

        upper_bound = Q3 + 1.5 * IQR

        thresholds = upper_bound

        # 그룹 내에서 이상치 수정
        idx_to_correct = df[(df[columns_to_check] > thresholds).any(axis=1) & (df['anomal'] == True)].index
        df.loc[idx_to_correct, 'anomal'] = False

        return df
    
    original_df = df.copy()
    df['jungle_kill'] = df['alliedJungleMonsterKills'] + df['enemyJungleMonsterKills']
    df = df.drop(['alliedJungleMonsterKills','enemyJungleMonsterKills'],axis=1)
    
    with open('./model/pca_model_liner.pkl', 'rb') as f:
        pca_liner = pickle.load(f)

    with open('./model/ocsvm_model_liner.pkl', 'rb') as f:
        ocsvm_liner = pickle.load(f)

    with open('./model/isf_model_liner.pkl', 'rb') as f:
        isf_liner = pickle.load(f)

    with open('./model/pca_model_jungle.pkl', 'rb') as f:
        pca_jungle = pickle.load(f)

    with open('./model/ocsvm_model_jungle.pkl', 'rb') as f:
        ocsvm_jungle = pickle.load(f)

    with open('./model/isf_model_jungle.pkl', 'rb') as f:
        isf_jungle = pickle.load(f)

    with open('./model/pca_model_supporter.pkl', 'rb') as f:
        pca_supporter = pickle.load(f)

    with open('./model/ocsvm_model_supporter.pkl', 'rb') as f:
        ocsvm_supporter = pickle.load(f)

    with open('./model/isf_model_supporter.pkl', 'rb') as f:
        isf_supporter = pickle.load(f)
        
    display_list = []    
    for match in df['match_id'].unique():
        ori_df = original_df[original_df['match_id']==match]
        o_df = first_preprocess(df[df['match_id']==match])

        liner_df = o_df.copy()
        jungle_df = o_df.copy()
        supporter_df = o_df.copy()

        liner_df = liner_df[~liner_df['teamPosition'].isin(['JUNGLE','UTILITY'])].reset_index(drop=True)
        jungle_df = jungle_df[jungle_df['teamPosition']=='JUNGLE'].reset_index(drop=True)
        supporter_df = supporter_df[supporter_df['teamPosition']=='UTILITY'].reset_index(drop=True)

        liner_smiteUser = liner_df['smiteUser']
        liner_afk = liner_df['afk']
        liner_item = liner_df['item_anomal']
        liner_df = liner_df[liner_df['smiteUser']==False].reset_index(drop=True)
        liner_df = liner_df.drop(['smiteUser'], axis=1)

        jungle_smiteUser = jungle_df['smiteUser']
        jungle_afk = jungle_df['afk']
        jungle_item = jungle_df['item_anomal']
        jungle_df = jungle_df[jungle_df['smiteUser']==True].reset_index(drop=True)
        jungle_df = jungle_df.drop(['smiteUser'], axis=1)

        supporter_smiteUser = supporter_df['smiteUser']
        supporter_afk = supporter_df['afk']
        supporter_item = supporter_df['item_anomal']
        supporter_df = supporter_df[supporter_df['smiteUser']==False].reset_index(drop=True)
        supporter_df = supporter_df.drop(['smiteUser'], axis=1)

        liner_df = second_preprocess(liner_df)
        jungle_df = second_preprocess(jungle_df)
        supporter_df = second_preprocess(supporter_df)

        liner_data = liner_df.drop('win',axis=1).values
        jungle_data = jungle_df.drop('win',axis=1).values
        supporter_data = supporter_df.drop('win',axis=1).values

        isf_liner_pred = isf_liner.predict(liner_data)
        isf_liner_scores = isf_liner.decision_function(liner_data)

        isf_jungle_pred = isf_jungle.predict(jungle_data)
        isf_jungle_scores = isf_jungle.decision_function(jungle_data)

        isf_supporter_pred = isf_supporter.predict(supporter_data)
        isf_supporter_scores = isf_supporter.decision_function(supporter_data)

        ori_liner_final_df = ori_df[~ori_df['teamPosition'].isin(['JUNGLE','UTILITY'])]
        liner_final_df = ori_liner_final_df[(ori_liner_final_df['item_duplication'].isin([1,2,3])) & (ori_liner_final_df['afk']==False) & (ori_liner_final_df['smiteUser']==False)]
        liner_final_df.insert(1,'isf_score',0)
        liner_final_df.insert(1,'isf_anomaly',0)
        liner_final_df['isf_anomaly'] = isf_liner_pred
        liner_final_df['isf_score'] = isf_liner_scores
        liner_final_df = isf_threshold(liner_final_df)

        ori_jungle_final_df = ori_df[ori_df['teamPosition']=='JUNGLE']
        jungle_final_df = ori_jungle_final_df[(ori_jungle_final_df['item_duplication'].isin([1,2,3])) & (ori_jungle_final_df['afk']==False) & (ori_jungle_final_df['smiteUser']==True)]
        jungle_final_df.insert(1,'isf_score',0)
        jungle_final_df.insert(1,'isf_anomaly',0)
        jungle_final_df['isf_anomaly'] = isf_jungle_pred
        jungle_final_df['isf_score'] = isf_jungle_scores
        jungle_final_df = isf_threshold(jungle_final_df)

        ori_support_final_df = ori_df[ori_df['teamPosition']=='UTILITY']
        support_final_df = ori_support_final_df[(ori_support_final_df['item_duplication'].isin([1,2,3])) & (ori_support_final_df['afk']==False) & (ori_support_final_df['smiteUser']==False)]
        support_final_df.insert(1,'isf_score',0)
        support_final_df.insert(1,'isf_anomaly',0)
        support_final_df['isf_anomaly'] = isf_supporter_pred
        support_final_df['isf_score'] = isf_supporter_scores
        support_final_df = isf_threshold(support_final_df)
        
        liner_pca_result = pca_liner.transform(liner_df)
        liner_data = pd.DataFrame(data=liner_pca_result, columns=[f'PC{i+1}' for i in range(10)]).values
        jungle_pca_result = pca_jungle.transform(jungle_df)
        jungle_data = pd.DataFrame(data=jungle_pca_result, columns=[f'PC{i+1}' for i in range(10)]).values
        supporter_pca_result = pca_supporter.transform(supporter_df)
        supporter_data = pd.DataFrame(data=supporter_pca_result, columns=[f'PC{i+1}' for i in range(10)]).values

        ocsvm_liner_pred = ocsvm_liner.predict(liner_data)
        ocsvm_liner_scores = ocsvm_liner.decision_function(liner_data)

        ocsvm_jungle_pred = ocsvm_jungle.predict(jungle_data)
        ocsvm_jungle_scores = ocsvm_jungle.decision_function(jungle_data)

        ocsvm_supporter_pred = ocsvm_supporter.predict(supporter_data)
        ocsvm_supporter_scores = ocsvm_supporter.decision_function(supporter_data)

        liner_final_df.insert(1,'ocsvm_score',0)
        liner_final_df.insert(1,'ocsvm_anomaly',0)
        liner_final_df['ocsvm_anomaly'] = ocsvm_liner_pred
        liner_final_df['ocsvm_score'] = ocsvm_liner_scores
        liner_final_df = ocsvm_threshold(liner_final_df)

        jungle_final_df.insert(1,'ocsvm_score',0)
        jungle_final_df.insert(1,'ocsvm_anomaly',0)
        jungle_final_df['ocsvm_anomaly'] = ocsvm_jungle_pred
        jungle_final_df['ocsvm_score'] = ocsvm_jungle_scores
        jungle_final_df = ocsvm_threshold(jungle_final_df)

        support_final_df.insert(1,'ocsvm_score',0)
        support_final_df.insert(1,'ocsvm_anomaly',0)
        support_final_df['ocsvm_anomaly'] = ocsvm_supporter_pred
        support_final_df['ocsvm_score'] = ocsvm_supporter_scores
        support_final_df = ocsvm_threshold(support_final_df)

        liner_cluster_df = voting(liner_final_df)
        jungle_cluster_df = voting(jungle_final_df)
        support_cluster_df = voting(support_final_df)

        liner_cluster_df = anomal_to_normal(liner_cluster_df)
        jungle_cluster_df = anomal_to_normal(jungle_cluster_df)
        support_cluster_df = anomal_to_normal(support_cluster_df)

        liner_cluster_df = pd.merge(ori_liner_final_df, 
                      liner_cluster_df[['match_id','teamId','teamPosition','anomal']],
                      on = ['match_id','teamId','teamPosition'], 
                      how = 'left')
        liner_cluster_df = liner_cluster_df.fillna(False)

        jungle_cluster_df = pd.merge(ori_jungle_final_df, 
                      jungle_cluster_df[['match_id','teamId','teamPosition','anomal']],
                      on = ['match_id','teamId','teamPosition'], 
                      how = 'left')
        jungle_cluster_df = jungle_cluster_df.fillna(False)

        support_cluster_df = pd.merge(ori_support_final_df, 
                      support_cluster_df[['match_id','teamId','teamPosition','anomal']],
                      on = ['match_id','teamId','teamPosition'], 
                      how = 'left')
        support_cluster_df = support_cluster_df.fillna(False)

        liner_cluster_df['afk']=liner_afk
        liner_cluster_df['item_anomal']=liner_item
        liner_cluster_df['smiteUser']=liner_smiteUser

        jungle_cluster_df['afk']=jungle_afk
        jungle_cluster_df['item_anomal']=jungle_item
        jungle_cluster_df['smiteUser']=jungle_smiteUser

        support_cluster_df['afk']=supporter_afk
        support_cluster_df['item_anomal']=supporter_item
        support_cluster_df['smiteUser']=supporter_smiteUser

        for raw in liner_cluster_df.index:
            if (liner_cluster_df.loc[raw,'afk']==True) or (liner_cluster_df.loc[raw,'item_anomal']==True) or (liner_cluster_df.loc[raw,'smiteUser']==True):
                liner_cluster_df.loc[raw,'anomal']=True
        liner_cluster_df = liner_cluster_df.drop(['item_anomal','smiteUser'],axis=1)

        for raw in jungle_cluster_df.index:
            if (jungle_cluster_df.loc[raw,'afk']==True) or (jungle_cluster_df.loc[raw,'item_anomal']==True) or (jungle_cluster_df.loc[raw,'smiteUser']==False):
                jungle_cluster_df.loc[raw,'anomal']=True
        jungle_cluster_df = jungle_cluster_df.drop(['item_anomal','smiteUser'],axis=1)

        for raw in support_cluster_df.index:
            if (support_cluster_df.loc[raw,'afk']==True) or (support_cluster_df.loc[raw,'item_anomal']==True) or (support_cluster_df.loc[raw,'smiteUser']==True):
                support_cluster_df.loc[raw,'anomal']=True
        support_cluster_df = support_cluster_df.drop(['item_anomal','smiteUser'],axis=1)

        final_df = pd.concat([liner_cluster_df, jungle_cluster_df, support_cluster_df]).reset_index(drop=True)
        final_df = final_df.sort_values(['match_id','teamId','teamPosition']).reset_index(drop=True)
        
        display_list.append(final_df)

    return display_list
    
def for_display(api_key, nickname, tag, num=3):
    searching(api_key, nickname, tag, num=num)
    participant, position, match_info, objectives = concat_data(directory='./st_data', make_file=True)
    position = calculate_position_change(position)
    position.to_csv('./st_data/concat_data/position.csv',index=False)
    match_tier, team_result, participant_preset, participant_minute, participant_result, champion = sql_normalization(directory='./st_data/concat_data',make_file=True)
    df = load_final_data(directory='./st_data/after_norm',for_model=True)
    df = anomaly_detection(df)

    folder_path = './st_data'
    csv_files = glob.glob(os.path.join(folder_path, '**', '*.csv'), recursive=True)
    for file in csv_files:
        os.remove(file)
        
    return df

