#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
파일명: cleansing.py
작성자: 서민혁, 서영석
작성일자: 2024-07-22
수정 이력:
    - 2024-07-23 17:59 서영석
        * load_data 함수에 KDA 카운트 코드 추가
    - 2024-07-24 21:00 서영석
        * 병렬 처리를 위해 Dask를 사용하는 코드로 변환
    - 2024-07-25 13:00 서영석
        * 코드 간결화를 위해 지표 계산 함수 분리
    - 2024-07-26 13:00 서영석
        * 코드 간결화를 위해 calculate 함수들 calculate.py로 분리
    - 2024-08-06 14:00 서영석
        * SQL 정규화 된 데이터셋 최종데이터로 통합하는 함수 생성
    - 2024-08-12 10:45 서영석
        * load_data > concat_data 후, csv로 저장하는 코드 추가
    - 2024-08-12 13:45 서영석
        * SQL 정규화 함수 생성
    - 2024-08-12 15:00 서영석
        * load_final_data에 모델링을 위한 데이터 추출 인자 및 코드 추가
설명: 데이터 정제를 위한 함수 모음
    - for_timeline : 매치 타임라인 데이터 추출 시,
    Output인 event와 participant 분할을 위해 사용
    - for_match_info : 경기 결과 데이터 추출 시,
    Output인 match_info 생성을 위해 사용
    - concat_data : 정제 작업을 위해 추출된 Raw 데이터 Concat 시 사용
    - sql_normalization : concat 데이터 sql 정규화
    - load_final_data : SQL 정규화 된 데이터셋 최종데이터로 통합 또는 모델링용 데이터 추출
'''
# Basic Library
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', message='`meta` is not specified, inferred from partial data. Please provide `meta` if the result is unexpected.')
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Dask Library
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# Custormized Module
from .calculate import *

def for_timeline(df):
    # event용
    event_df = pd.DataFrame(df['events']) # 리스트를 DataFrame으로 변환
    event_df['match_id'] = df['match_id']  # match_id 열 추가

    # 분석에 필요한 행만 선택
    if 'type' in event_df.columns:
        event_df = event_df[event_df['type'].isin(['ITEM_PURCHASED', 'ITEM_UNDO',
                                                'CHAMPION_KILL', 'ITEM_SOLD',
                                                'BUILDING_KILL', 'ELITE_MONSTER_KILL'])].reset_index(drop=True)
    event_df = event_df.dropna(axis=1, how='all')

    # killerId와 creatorId는 행위자 Id로써
    # participantId와 동일한 성격을 띄므로 해당 컬럼 모두 participantId로 통일
    if 'participantId' not in event_df.columns:
        event_df['participantId'] = np.nan
    if 'creatorId' in event_df.columns:
        event_df.loc[event_df['creatorId'].notnull(), 'participantId'] = event_df['creatorId']
        event_df.drop('creatorId',axis=1,inplace=True)
    if 'killerId' in event_df.columns:    
        event_df.loc[event_df['killerId'].notnull(), 'participantId'] = event_df['killerId']
        event_df.drop('killerId',axis=1,inplace=True)
    
    # 분당 지표 생성을 위해 timestamp 분 단위 컬럼 생성
    if 'timestamp' in event_df.columns:    
        event_df.insert(1,'minute',0)
        event_df['minute'] = event_df['timestamp'].apply(lambda x: x // 60000)

    # assistingParticipantIds를 풀어 assist_1~4 컬럼으로 생성
    if 'assistingParticipantIds' in event_df.columns:
        assist = event_df[event_df['assistingParticipantIds'].notnull()][['type', 'match_id',
                                                                'minute', 'assistingParticipantIds', 
                                                                'participantId']]
        assist = assist[assist['participantId'] > 0]
        assist['assistingParticipantIds'] = assist['assistingParticipantIds'].apply(lambda x: list(set(x)))

        try:
            # # 람다 함수를 사용하여 assistingParticipantIds 필터링
            assist['assistingParticipantIds'] = assist.apply(
                lambda row: [aid for aid in row['assistingParticipantIds'] 
                            if (1 <= aid <= 5 and 1 <= row['participantId'] <= 5) 
                            or (6 <= aid <= 10 and 6 <= row['participantId'] <= 10) 
                            or (1 <= aid <= 10 and not (1 <= row['participantId'] <= 10))], axis=1)

            # assistingParticipantIds가 빈 리스트인 행 제거
            assist = assist[assist['assistingParticipantIds'].map(len) > 0]
            assist_list = assist['assistingParticipantIds'].apply(
                lambda x: pd.Series(x + [np.nan] * (4 - len(x))))
            assist[['a_1', 'a_2', 'a_3', 'a_4']] = assist_list

            event_df[['assist_1', 'assist_2', 'assist_3', 'assist_4']] = assist[['a_1', 'a_2', 'a_3', 'a_4']]

        except:
            pass
            
        event_df = event_df.drop('assistingParticipantIds',axis=1)

    # pf용
    p_df = pd.DataFrame(df['participantFrames']).T.reset_index(drop=True)
    p_df['timestamp'] = df['timestamp'] # timestamp열 추가
    p_df['timestamp'] = p_df['timestamp'].apply(lambda x: x // 60000)
    p_df['match_id'] = df['match_id']  # match_id 열 추가

    # damageStats' 열을 각각의 열로 분해하여 DataFrame으로 변환
    ds = pd.DataFrame(p_df['damageStats'].tolist())[['totalDamageDone','totalDamageDoneToChampions','totalDamageTaken']]
    # championStats 열을 각각의 열로 분해하여 DataFrame으로 변환
    cs = pd.DataFrame(p_df['championStats'].tolist())
    # position 열을 각각의 열로 분해하여 DataFrame으로 변환
    po = pd.DataFrame(p_df['position'].tolist())

    # position DataFrame 생성
    position_df = p_df[['match_id', 'participantId', 'timestamp']].copy()
    position_df[['position_x', 'position_y']] = po

    # 원래의 participant DataFrame과 분해된 DataFrame들을 열기준으로 결합
    participant_df = pd.concat([p_df, ds, cs], axis=1)
    # 불필요한 열 제거
    participant_df = participant_df.drop(['championStats', 'damageStats', 'position'], axis=1)

    participant_df = pd.concat([participant_df.iloc[:, 9],
                                participant_df.iloc[:, :9],
                                participant_df.iloc[:, 10:]], axis=1)
    participant_df = pd.concat([participant_df.iloc[:, 10],
                                participant_df.iloc[:, :10],
                                participant_df.iloc[:, 11:]], axis=1)
    
    position_df['lane'] = position_df.apply(lambda row: calculate_lane(row['position_x'], row['position_y']), axis=1)

    return event_df, participant_df, position_df

# -------------------------------------------------------------------------------

def for_match_info(json_file):
    import numpy as np
    import pandas as pd

    # participants 데이터를 DataFrame으로 파싱
    match_df = pd.DataFrame(json_file['participants'])
    
    # required_columns에 summonerName과 riotIdGameName 모두 포함
    required_columns = ['teamId', 'puuid', 'riotIdGameName', 'summonerName', 'summonerId', 'participantId',
                        'teamPosition', 'challenges', 'championName', 'lane',
                        'kills', 'deaths', 'assists', 'summoner1Id', 'summoner2Id',
                        'totalMinionsKilled', 'neutralMinionsKilled', 'goldEarned', 'goldSpent',
                        'champExperience', 'item0', 'item1', 'item2', 'item3', 'item4', 'item5',
                        'item6', 'totalDamageDealt', 'totalDamageDealtToChampions', 'totalDamageTaken',
                        'damageDealtToBuildings', 'damageDealtToObjectives', 'damageDealtToTurrets',
                        'totalTimeSpentDead', 'visionScore', 'win', 'timePlayed']
    
    # 누락된 컬럼은 np.nan으로 채움
    for col in required_columns:
        if col not in match_df.columns:
            match_df[col] = np.nan

    # 반드시 copy()를 사용하여 독립적인 DataFrame 생성
    sample = match_df[required_columns].copy()
    
    # 만약 summonerName이 NaN 또는 빈 문자열이면, riotIdGameName 값을 사용하여 summonerName에 채워 넣기
    sample['summonerName'] = sample['summonerName'].mask(
        (sample['summonerName'].isna()) | (sample['summonerName'] == ""),
        sample['riotIdGameName']
    )
    # 더 이상 필요 없는 riotIdGameName 컬럼은 제거
    sample.drop(columns=['riotIdGameName'], inplace=True)
    
    # challenges 처리
    if 'challenges' in sample.columns:
        challenge = pd.DataFrame(sample['challenges'].tolist())
        challenge_columns = ['abilityUses', 'skillshotsDodged', 'skillshotsHit',
                             'enemyChampionImmobilizations', 'laneMinionsFirst10Minutes',
                             'controlWardsPlaced', 'wardTakedowns', 'effectiveHealAndShielding',
                             'dragonTakedowns', 'baronTakedowns']
        jungle_col = challenge.filter(regex='^jungle|Jungle|kda')
        for col in challenge_columns:
            if col not in challenge.columns:
                challenge[col] = np.nan
        for col in jungle_col.columns:
            if col not in challenge.columns:
                challenge[col] = np.nan
        match_info = pd.concat([sample, challenge[challenge_columns], jungle_col], axis=1)
        match_info = match_info.drop(['challenges'], axis=1)
        if 'moreEnemyJungleThanOpponent' in match_info.columns:
            match_info = match_info.drop(['moreEnemyJungleThanOpponent'], axis=1)
    else:
        match_info = sample

    # 데이터가 비어있거나 10줄이 아니면 None 반환
    if match_info.empty or match_info.isna().all().all():
        return None
    if len(match_info) != 10:
        return None

    # 팀(오브젝트) 데이터 처리
    object_df = pd.DataFrame(json_file['teams'])
    objectives = pd.json_normalize(object_df['objectives'])
    possible_objectives = ['baron.kills', 'champion.kills', 'dragon.kills', 'horde.kills',
                           'inhibitor.kills', 'riftHerald.kills', 'tower.kills']
    existing_objectives = [col for col in possible_objectives if col in objectives.columns]
    objectives = objectives[existing_objectives]
    objectives.columns = [col.split('.')[0] + '_kills' for col in objectives.columns]
    objectives = pd.concat([object_df[['teamId']], objectives], axis=1)

    return match_info, objectives
    
# -------------------------------------------------------------------------------    

def concat_data(directory='./data', to_pandas=True, make_file=False):
    data_list = ['event', 'participant', 'position', 'match_info','objectives']
    df_list = []
    # print('데이터 불러오는 중..')
    for dt in data_list:
        # print(dt)
        dt_pocket = []
        for i in range(1,len(os.listdir(f'{directory}/{dt}'))+1):
            file = dd.read_csv(
                f'{directory}/{dt}/{dt}{i}.csv',
                dtype={'buildingType': 'object', 'killType': 'object', 
                       'laneType': 'object', 'monsterSubType': 'object',
                       'monsterType': 'object', 'position': 'object',
                       'towerType': 'object', 'transformType': 'object',
                       'victimDamageDealt': 'object', 'victimDamageReceived': 'object',
                       'wardType': 'object', 'levelUpType' : 'object'}
            )
            dt_pocket.append(file)
        df = dd.concat(dt_pocket, axis=0)
        df_list.append(df)

    # print('데이터 정제 중..')
    event = df_list[0]
    participant = df_list[1]
    position = df_list[2]
    match_info = df_list[3]
    objectives = df_list[4]

    participant = dd.merge(participant, 
                           match_info[['match_id', 'puuid', 'summonerName',
                                       'participantId', 'teamPosition', 'championName']], 
                           on=['match_id', 'participantId'], 
                           how='inner')
    
    cols = participant.columns.tolist()
    cols.insert(0, cols.pop(cols.index('puuid')))
    cols.insert(1, cols.pop(cols.index('summonerName')))
    # cols.insert(2, cols.pop(cols.index('tier')))
    cols.insert(4, cols.pop(cols.index('teamPosition')))
    cols.insert(5, cols.pop(cols.index('championName')))
    participant = participant[cols]

    # print('지표 추가 중..')
    # with ProgressBar():
    # print('KDA')
    participant = participant.persist()
    participant = calculate_KDA(participant, event)
    # with ProgressBar():
        # print('Involve')
    participant = participant.persist()
    participant = calculate_involve(participant, event)
    # with ProgressBar():
        # print('Item')
    participant = participant.persist()
    participant = calculate_item(participant, event)
    # with ProgressBar():
        # print('캐시 저장 및 중복 제거 중..')
    participant = participant.persist()

    participant = participant.drop_duplicates(subset=['puuid', 'match_id', 'timestamp', 'participantId'])
    position = position.drop_duplicates(subset=['match_id', 'participantId', 'timestamp'])
    match_info = match_info.drop_duplicates(subset=['puuid', 'match_id', 'participantId'])
    objectives = objectives.drop_duplicates(subset=['match_id', 'teamId'])

    if to_pandas:
        # print('DASK -> Pandas로 변환 중')
        # with ProgressBar():
            # print('event')
            # event = event.compute().sort_values(['match_id','participantId','timestamp']).reset_index(drop=True)
        # print('participant')
        participant = participant.compute().sort_values(['puuid','match_id','timestamp']).reset_index(drop=True)
        # print('position')
        position = position.compute().sort_values(['match_id','participantId','timestamp']).reset_index(drop=True)
        # print('match_info')
        match_info = match_info.compute().sort_values(['match_id','participantId']).reset_index(drop=True)
        # print('objectives')
        objectives = objectives.compute().sort_values(['match_id','teamId']).reset_index(drop=True)

    if make_file:
        # print('최종 데이터 저장 중')
        participant.to_csv(f'{directory}/concat_data/participant.csv', index=False)
        position.to_csv(f'{directory}/concat_data/position.csv', index=False)
        match_info.to_csv(f'{directory}/concat_data/match_info.csv', index=False)
        objectives.to_csv(f'{directory}/concat_data/objectives.csv', index=False)

    # print('완료')

    return participant, position, match_info, objectives

# -------------------------------------------------------------------------------    
def sql_normalization(directory = './data/concat_data', make_file = False):
    data_list = ['participant', 'position', 'match_info', 'objectives']
    df_list = []
    # print('데이터 불러오는 중..')
    for dt in data_list:
        # print(dt)
        file = pd.read_csv(f'{directory}/{dt}.csv')
        df_list.append(file)

    # print('변수 처리 중..')
    participant = df_list[0]
    position = df_list[1]
    match_info = df_list[2]
    objectives = df_list[3]

    # 타임스탬프 동기화
    match_info['timePlayed'] = match_info['timePlayed'].apply(lambda x: round(x/60,1))

    # 팀처치 지표 그루핑
    objectives['monster_kills'] = objectives[['baron_kills', 
                                              'dragon_kills', 
                                              'riftHerald_kills',
                                              'horde_kills']].sum(axis=1)
    objectives['building_kills'] = objectives[['inhibitor_kills', 
                                               'tower_kills']].sum(axis=1)
    objectives['objective_kills'] = objectives[['monster_kills', 
                                                'building_kills']].sum(axis=1)
    
    # 킬관여율
    match_info = pd.merge(match_info, objectives[['match_id','teamId','champion_kills']],
                          on = ['match_id','teamId'],
                          how = 'inner')

    match_info['kill_involve_ratio'] = round((match_info['kills'] + match_info['assists']) / match_info['champion_kills'],3)
    match_info['kill_involve_ratio'] = match_info['kill_involve_ratio'].fillna(0)
    match_info = match_info.drop(['champion_kills'],axis=1)

    # 팀내 데스 비율
    team_deaths = match_info.groupby(['match_id', 'teamId'])['deaths'].sum().reset_index()
    team_deaths.rename(columns={'deaths': 'total_team_deaths'}, inplace=True)
    match_info = match_info.merge(team_deaths, on=['match_id', 'teamId'])
    match_info['death_ratio']  = round(match_info['deaths'] / match_info['total_team_deaths'],3)
    match_info['death_ratio'] = match_info['death_ratio'].fillna(0)

    # CS
    match_info['cs'] = match_info['totalMinionsKilled'] + match_info['neutralMinionsKilled']

    # item_duplication 컬럼 추가
    def calculate_max_duplication(row):
        item_counts = row[row != 0].value_counts()
        if item_counts.empty:
            return 0
        max_duplication = item_counts.max()
        return max_duplication

    # item0부터 item5까지의 열만 선택하여 중복 횟수 계산
    match_info['item_duplication'] = match_info[['item0', 'item1', 
                                                 'item2', 'item3', 
                                                 'item4', 'item5']].apply(calculate_max_duplication, axis=1)

    # 관여
    involve = participant[['match_id','participantId','puuid',
                           'timestamp','involve_tower','involve_object']]

    # 순거래량
    participant['transaction_margin'] = participant['item_transaction'] - participant['item_undo']
    transaction = participant[['match_id', 'participantId','puuid', 
                               'timestamp', 'transaction_margin']]
    transaction.loc[(transaction['timestamp'] == 0) & (transaction['transaction_margin'] < 0), 'transaction_margin'] = 1

    # 스탯
    participant['attack_stat'] = participant['attackDamage'] + participant['abilityPower']
    participant['defense_stat'] = participant['armor'] + participant['magicResist']
    participant['max_health'] = participant['healthMax']

    stat = participant[['match_id', 'participantId','puuid', 'timestamp', 
                        'attack_stat', 'attackSpeed', 'defense_stat', 'max_health']]

    # 합치기
    result = pd.merge(involve,transaction,on=['match_id','participantId','puuid','timestamp'])
    result = pd.merge(result,stat,on=['match_id','participantId','puuid','timestamp'])
    result = pd.merge(result,
                      position[['match_id','participantId','timestamp','position_change','lane']],
                      on=['match_id','participantId','timestamp'])
    result = pd.merge(result, 
                      participant[['match_id','participantId','timestamp','death','totalGold','level','xp']],
                      on=['match_id','participantId','timestamp'])
    
    # 정규화된 테이블 생성
    match_tier = match_info[['match_id','tier', 'timePlayed']].drop_duplicates(subset=['match_id']).reset_index(drop=True)
    team_result = pd.merge(match_info[['match_id', 'teamId', 'win']],
                        objectives[['match_id', 'teamId', 'champion_kills', 'monster_kills', 'building_kills']],
                        on=['match_id', 'teamId']).drop_duplicates(subset=['match_id', 'teamId']).reset_index(drop=True)
    participant_preset = match_info[['match_id','teamId','summonerName', 'puuid','teamPosition','championName',
                            'summoner1Id', 'summoner2Id']]
    participant_minute = result.drop(['participantId'],axis=1)
    participant_result = match_info[['match_id', 'puuid', 'kill_involve_ratio', 'deaths','cs', 'goldEarned',
                                     'goldSpent','champExperience','item0','item1','item2','item3','item4','item5',
                                     'item_duplication', 'totalDamageTaken','totalDamageDealtToChampions',
                                     'damageDealtToObjectives','totalTimeSpentDead','visionScore',
                                     'abilityUses','skillshotsDodged','laneMinionsFirst10Minutes',
                                     'alliedJungleMonsterKills','enemyJungleMonsterKills', 'kda',
                                     'jungleCsBefore10Minutes','takedownsBeforeJungleMinionSpawn',
                                     'death_ratio']]
    champion = pd.read_csv('./st_data/champion.csv')

    if make_file:
        # print('최종 데이터 저장 중')
        direct = directory.replace('/concat_data','')
        match_tier.to_csv(f'{direct}/after_norm/match_tier.csv', index=False)
        team_result.to_csv(f'{direct}/after_norm/team_result.csv', index=False)
        participant_preset.to_csv(f'{direct}/after_norm/participant_preset.csv', index=False)
        participant_minute.to_csv(f'{direct}/after_norm/participant_minute.csv', index=False)
        participant_result.to_csv(f'{direct}/after_norm/participant_result.csv', index=False)
        champion.to_csv(f'{direct}/after_norm/champion.csv', index=False)

    # print('완료')

    return match_tier, team_result, participant_preset, participant_minute, participant_result, champion

# -------------------------------------------------------------------------------    

def load_final_data(directory='./data/after_norm', make_file = False, for_model=False):
    csv_list = os.listdir(directory)
    
    # print('데이터 불러오는 중..')
    champion = pd.read_csv(os.path.join(directory,csv_list[0]))
    match_tier = pd.read_csv(os.path.join(directory,csv_list[1]))
    participant_minute = pd.read_csv(os.path.join(directory,csv_list[2]))
    participant_preset = pd.read_csv(os.path.join(directory,csv_list[3]))
    participant_result = pd.read_csv(os.path.join(directory,csv_list[4]))
    team_result = pd.read_csv(os.path.join(directory,csv_list[5]))
    
    df = pd.merge(match_tier,team_result,
                  on=['match_id'])
    df = pd.merge(df,participant_preset,
                  on=['match_id','teamId'])
    df = pd.merge(df, participant_result, 
                  on=['match_id','puuid'])
    df = pd.merge(df, champion,
                  on=['championName'])
    
    df['subRole'] = df['subRole'].fillna('-')

    column_list = [col for col in participant_minute.drop(['match_id','puuid','timestamp','lane'],axis=1)]
    # print('데이터 병합 중..')
    for col in column_list:
        if col==column_list[0]:
            # 타워 관여율
            involve = participant_minute.groupby(['match_id',
                                                  'puuid'],as_index=False)[['involve_tower']].sum()

            result = pd.merge(df, involve[['puuid','match_id','involve_tower']],
                              on=['puuid','match_id'],
                              how='inner')

            result['tower_involve_ratio'] = round(result['involve_tower']/result['building_kills'],3)
            result['tower_involve_ratio'] = result['tower_involve_ratio'].fillna(0)
            result = result.drop(['involve_tower'],axis=1)

        elif col==column_list[1]:
            # 오브젝트 관여율
            involve = participant_minute.groupby(['match_id',
                                                  'puuid'],as_index=False)[['involve_object']].sum()

            result = pd.merge(result, involve[['puuid','match_id','involve_object']],
                              on=['puuid','match_id'],
                              how='inner')

            result['object_involve_ratio'] = round(result['involve_object']/result['monster_kills'],3)
            result['object_involve_ratio'] = result['object_involve_ratio'].fillna(0)
            result = result.drop(['involve_object'],axis=1)

        elif col==column_list[2]:
            # 10분당 순거래율
            summary = participant_minute.groupby(['match_id', 'puuid']).agg({
                'transaction_margin': 'sum',
                'timestamp': 'max'
            }).reset_index()

            summary['average_transaction_margin_per_10min'] = round(summary['transaction_margin'] / (summary['timestamp'] / 10),2)

            result = pd.merge(result,
                              summary[['match_id','puuid','average_transaction_margin_per_10min']],
                              on = ['match_id','puuid'],
                              how = 'inner')

        elif col in column_list[3:7]:
            # 스탯
            stat = participant_minute[['match_id','puuid','timestamp', col]]
            stat[col] = stat.groupby(['match_id', 'puuid'])[col].diff()
            mean_changes = stat.groupby(['match_id', 'puuid'])[col].mean().reset_index()
            mean_changes[col] = mean_changes[col].round(1)

            result = pd.merge(result,mean_changes,
                              on = ['match_id','puuid'],
                              how = 'inner')
            
            result = result.rename(columns={col:f'delta_{col}'})

        elif col==column_list[7]:
            # 안움직임
            po_anorm = participant_minute[['match_id','timestamp','puuid','lane','position_change']]
            po_anorm = po_anorm[po_anorm['timestamp']>0]
            po_anorm = po_anorm[po_anorm['position_change']==0]
            po_anorm = po_anorm[po_anorm['lane'].isin(['red_zone','blue_zone'])]

            po_anorm = po_anorm.groupby(['match_id', 'puuid']).size().reset_index(name='no_moving_minute')

            result = pd.merge(result,po_anorm,
                              on = ['match_id', 'puuid'],
                              how = 'left')

            result['no_moving_minute'] = result['no_moving_minute'].fillna(0)

        elif col==column_list[8]:
            # 10분전 데스
            filtered_pa = participant_minute[participant_minute['timestamp'] <= 10]
            summary = filtered_pa.groupby(['match_id', 'puuid'],as_index=False)[['death']].sum()
            summary = summary.rename(columns={'death':'deathBefore10Minutes'})

            result = pd.merge(result,summary,
                              on = ['match_id', 'puuid'],
                              how = 'left')
            
        elif col==column_list[9]:
            # 10분전 누적 골드
            filtered_pa = participant_minute[['match_id', 'puuid', col]]
            filtered_pa = filtered_pa[participant_minute['timestamp'] == 10]
            filtered_pa = filtered_pa.rename(columns={'totalGold':'goldBefore10Minutes'})
            
            result = pd.merge(result,filtered_pa,
                              on = ['match_id', 'puuid'],
                              how = 'left')
            
        elif col==column_list[10]:
            # 10분 레벨
            filtered_pa = participant_minute[['match_id', 'puuid', col]]
            filtered_pa = filtered_pa[participant_minute['timestamp'] == 10]
            filtered_pa = filtered_pa.rename(columns={'level':'level10Minutes'})

            result = pd.merge(result,filtered_pa,
                              on = ['match_id', 'puuid'],
                              how = 'left')
            
        elif col==column_list[11]:
            # 10분 레벨
            filtered_pa = participant_minute[['match_id', 'puuid', col]]
            filtered_pa = filtered_pa[participant_minute['timestamp'] == 10]
            filtered_pa = filtered_pa.rename(columns={'xp':'xp10Minutes'})

            result = pd.merge(result,filtered_pa,
                              on = ['match_id', 'puuid'],
                              how = 'left')
            
    result = pd.concat([result.iloc[:,:18],
                        result.iloc[:,42:44],
                        result.iloc[:,18:42],
                        result.iloc[:,44:]], axis=1)
            
    # result = pd.concat([result.iloc[:,:11],
    #                     result.iloc[:,35:37],
    #                     result.iloc[:,11:35],
    #                     result.iloc[:,37:]], axis=1)

    # 소환사 주문 딕셔너리
    summoner_spells = {
        1: "Cleanse",
        3: "Exhaust",
        4: "Flash",
        6: "Ghost",
        7: "Heal",
        11: "Smite",
        12: "Teleport",
        14: "Ignite",
        21: "Barrier",
        30: "To the King!",
        31: "Poro Toss",
        32: "Mark"
    }

    # summoner1Id 및 summoner2Id 열을 소환사 주문 이름으로 변환
    result['summoner1Id'] = result['summoner1Id'].map(summoner_spells)
    result['summoner2Id'] = result['summoner2Id'].map(summoner_spells)
    
    if make_file:
        # print('최종 데이터 저장 중')
        result.to_csv(f'./data/final_df.csv', index=False)

    if for_model:
        # print('모델링을 위한 전처리 중')

        # 플레이 시간이 15분 이하 매치 제거
        result = result[result['timePlayed']>=15].reset_index(drop=True)

        # teamPosition 결측치 겹치는 매치 없으므로 없는 라인으로 채우기
        positions = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']

        # 결측치가 있는 매치 및 팀 추출
        missing_positions = result[result['teamPosition'].isna()]
        unique_matches = missing_positions[['match_id', 'teamId']].drop_duplicates()

        # 결측치 채우기 함수 정의
        def fill_missing_positions(group):
            existing_positions = group['teamPosition'].dropna().unique().tolist()
            missing_positions = [pos for pos in positions if pos not in existing_positions]
            
            # 결측치가 있는 경우 순서대로 채우기
            for idx in group.index:
                if pd.isna(group.at[idx, 'teamPosition']) and missing_positions:
                    group.at[idx, 'teamPosition'] = missing_positions.pop(0)
            return group

        # 결측치가 있는 각 매치 및 팀에 대해 결측치 채우기
        for _, row in unique_matches.iterrows():
            match_id, team_id = row['match_id'], row['teamId']
            condition = (result['match_id'] == match_id) & (result['teamId'] == team_id)
            result.loc[condition] = fill_missing_positions(result.loc[condition])

        # 10명 모두의 기록이 없는 매치 제거
        match_counts = result['match_id'].value_counts()
        valid_matches = match_counts[match_counts == 10].index
        result = result[result['match_id'].isin(valid_matches)].reset_index(drop=True)

        # 해석이 불가능한 컬럼 드롭
        result = result.drop(['takedownsBeforeJungleMinionSpawn'],axis=1)

        team_gold = result.groupby(['match_id', 'teamId', 'teamPosition'])['goldBefore10Minutes'].sum().reset_index()
        
        # 팀별 10분 이내 골드 합계 pivot 테이블 생성
        pivot_table = team_gold.pivot_table(index=['match_id', 
                                                   'teamPosition'], columns='teamId', 
                                                   values='goldBefore10Minutes', fill_value=0).reset_index()
        
        # 골드 차이 계산
        pivot_table['gold_diff_100'] = pivot_table[100] - pivot_table[200]
        pivot_table['gold_diff_200'] = pivot_table[200] - pivot_table[100]

        # pivot_table을 원본 result에 병합하여 early_gold_diff 컬럼 추가
        result = pd.merge(result, pivot_table[['match_id', 'teamPosition', 
                                               'gold_diff_100', 'gold_diff_200']], on=['match_id', 'teamPosition'], how='left')

        # teamId에 따른 early_gold_diff 컬럼 값 설정
        result['early_gold_diff'] = result.apply(lambda x: x['gold_diff_100'] if x['teamId'] == 100 else x['gold_diff_200'], axis=1)

        # 불필요한 컬럼 드롭
        result = result.drop(columns=['gold_diff_100', 'gold_diff_200'])

        team_levels = result.groupby(['match_id', 'teamId', 'teamPosition'])['level10Minutes'].sum().reset_index()

        # 팀별 10분 이내 레벨 합계 pivot 테이블 생성
        pivot_table = team_levels.pivot_table(index=['match_id', 'teamPosition'], columns='teamId', values='level10Minutes', fill_value=0).reset_index()

        # 레벨 차이 계산
        pivot_table['level_diff_100'] = pivot_table[100] - pivot_table[200]
        pivot_table['level_diff_200'] = pivot_table[200] - pivot_table[100]

        # pivot_table을 원본 result 병합하여 diff_early_level 컬럼 추가
        result = pd.merge(result, pivot_table[['match_id', 'teamPosition', 'level_diff_100', 'level_diff_200']], on=['match_id', 'teamPosition'], how='left')

        # teamId에 따른 diff_early_level 컬럼 값 설정
        result['diff_early_level'] = result.apply(lambda x: x['level_diff_100'] if x['teamId'] == 100 else x['level_diff_200'], axis=1)

        # 불필요한 컬럼 드롭
        result = result.drop(columns=['level_diff_100', 'level_diff_200'])

        # 라인전 터진 매치 드롭
        conditions = [
            (result['early_gold_diff'] >= 1000) | (result['diff_early_level'] >= 2),
            (result['early_gold_diff'] <= -1000) | (result['diff_early_level'] <= -2)
        ]
        choices = ['win', 'lose']
        result['early_competition'] = np.select(conditions, choices, default='draw')

        team_competition = result.groupby(['match_id', 'teamId'])['early_competition'].agg(lambda x: all(y == 'win' for y in x) or all(y == 'lose' for y in x)).reset_index()
        team_competition = team_competition[team_competition['early_competition'] == True]

        # 매치-팀별로 'early_competition'이 모두 동일한 행만 필터링
        filtered_df = pd.merge(result, team_competition[['match_id', 'teamId']], on=['match_id', 'teamId'])

        # result에서 filtered_df에 있는 데이터를 제거
        result = result[~result.index.isin(filtered_df.index)].reset_index(drop=True)

        result = result.drop(columns=['early_gold_diff', 'diff_early_level'])

        # 자리비움 컬럼 생성 후, no_moving_minute 컬럼 드롭
        result['afk'] = result['no_moving_minute'] > 1

        # smite 장착 시 True 아니면 False인 컬럼 생성 후 스펠 컬럼 드롭
        result['smiteUser'] = (result['summoner1Id'] == 'Smite') | (result['summoner2Id'] == 'Smite')

        result = result[['match_id', 'tier', 'timePlayed', 'teamId', 'win', 'summonerName','puuid', 'teamPosition', 
                         'championName', 'mainRole', 'subRole', 'summoner1Id', 'summoner2Id',
                         'smiteUser', 'item0','item1','item2','item3','item4','item5', 'kda', 'deaths', 
                         'kill_involve_ratio','death_ratio' , 'tower_involve_ratio',
                         'object_involve_ratio','cs', 'goldEarned', 'goldSpent',
                         'average_transaction_margin_per_10min','champExperience',
                         'item_duplication' ,'totalDamageTaken','totalDamageDealtToChampions',
                         'damageDealtToObjectives','visionScore', 'alliedJungleMonsterKills', 
                         'enemyJungleMonsterKills', 'afk','goldBefore10Minutes',
                         'xp10Minutes', 'jungleCsBefore10Minutes', 'laneMinionsFirst10Minutes']]

    # print('완료')    
    return result
