import os
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import pandas as pd
import multiprocessing as mp
import logging
import pmdarima as pm
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import gc

import warnings
warnings.filterwarnings('ignore')

# 현재 스크립트가 저장된 위치를 기준으로 경로 설정
current_dir = os.path.dirname(os.path.realpath(__file__))

print(os.getcwd())
# 파일 호출
data_path: str = "./data"
train_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "train.csv")).assign(_type="train") # train 에는 _type = train 
test_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "test.csv")).assign(_type="test") # test 에는 _type = test
submission_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "test.csv")) # ID, target 열만 가진 데이터 미리 호출
df: pd.DataFrame = pd.concat([train_df, test_df], axis=0)

# HOURLY_ 로 시작하는 .csv 파일 이름을 file_names 에 할딩
file_names: List[str] = [
    f for f in os.listdir(data_path) if f.startswith("HOURLY_") and f.endswith(".csv")
]

# 파일명 : 데이터프레임으로 딕셔너리 형태로 저장
file_dict: Dict[str, pd.DataFrame] = {
    f.replace(".csv", ""): pd.read_csv(os.path.join(data_path, f)) for f in file_names
}

# 1. 모든 파일에서 날짜 범위 확인
min_date = None
max_date = None
for _df in file_dict.values():
    # _df['datetime'] = pd.to_datetime(_df['datetime'])  # datetime 형식으로 변환
    if min_date is None or _df['datetime'].min() < min_date:
        min_date = _df['datetime'].min()
    if max_date is None or _df['datetime'].max() > max_date:
        max_date = _df['datetime'].max()

# 2. min_date ~ max_date 까지의 범위로 ID 열 생성 (모든 시간을 포함)
full_id_range = pd.DataFrame({
    'ID': pd.date_range(start=min_date, end=max_date, freq='h')  # 1시간 단위로 범위 생성
})
full_id_range['ID'] = full_id_range['ID'].astype(str) 

# 3. ID 기준으로 각 파일과 병합
df = df.merge(full_id_range, on='ID', how='right')  # train과 test 데이터를 전체 범위에 맞춰 병합

# 4. 다른 파일들과 병합
for _file_name, _df in tqdm(file_dict.items()):
    # 열 이름 중복 방지를 위해 {_file_name.lower()}_{col.lower()}로 변경, datetime 열을 ID로 변경
    _rename_rule = {
        col: f"{_file_name.lower()}_{col.lower()}" if col != "datetime" else "ID"
        for col in _df.columns
    }
    _df = _df.rename(_rename_rule, axis=1)
    df = df.merge(_df, on="ID", how="left")  # ID 기준으로 left join
    
# 모델에 사용할 컬럼, 컬럼의 rename rule을 미리 할당함
cols_dict: Dict[str, str] = {
    "ID": "ID",
    "target": "target",
    "_type": "_type",
    "hourly_market-data_price-ohlcv_all_exchange_spot_btc_usd_close": "close",
    "hourly_market-data_coinbase-premium-index_coinbase_premium_gap": "coinbase_premium_gap",
    "hourly_market-data_coinbase-premium-index_coinbase_premium_index": "coinbase_premium_index",
    "hourly_market-data_funding-rates_all_exchange_funding_rates": "funding_rates",
    "hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations": "long_liquidations",
    "hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations_usd": "long_liquidations_usd",
    "hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations": "short_liquidations",
    "hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations_usd": "short_liquidations_usd",
    "hourly_market-data_open-interest_all_exchange_all_symbol_open_interest": "open_interest",
    "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_ratio": "buy_ratio",
    "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_sell_ratio": "buy_sell_ratio",
    "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_volume": "buy_volume",
    "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_ratio": "sell_ratio",
    "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_volume": "sell_volume",
    "hourly_network-data_addresses-count_addresses_count_active": "active_count",
    "hourly_network-data_addresses-count_addresses_count_receiver": "receiver_count",
    "hourly_network-data_addresses-count_addresses_count_sender": "sender_count",
}
df = df[cols_dict.keys()].rename(cols_dict, axis=1)

temp = df.loc[df['_type']=='train'][['ID', 'close', 'target']]
temp['previous'] = temp['close'].shift(1).bfill()
temp['close_change'] = (temp['close']/temp['previous']) - 1
df = df.merge(temp[['ID', 'close_change']], on='ID', how='left')

# 2019년 3월 30일부터 2024년 4월 26일까지의 데이터를 필터링: open_interest가 19년 3월 30일부터 기록됨
# 데이터가 너무 많음...
# 2022년 1월 1일부터 2024년 4월 26일까지의 데이터를 필터링
start_date = '2022-01-01'
end_date = '2024-04-26 07:00:00'
df_filtered = df[(pd.to_datetime(df['ID']) >= start_date) & (pd.to_datetime(df['ID']) <= end_date)]

# 로그 설정 (temp 폴더 안에 로그 파일 저장)
logging.basicConfig(filename=os.path.join(current_dir, 'model_log.log'), 
                    level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# MAPE 함수 정의
def MAPE(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true))

# # 병렬 처리로 롤링 예측 수행하는 함수
# def parallel_update_predict(test_slice, model):
#     y_pred_local = []
#     for new_ob in test_slice:
#         fc, conf = model.predict(n_periods=1, return_conf_int=True)
#         y_pred_local.append(fc.tolist()[0])
#         model.update(new_ob)
#     return y_pred_local

# 배치 단위로 롤링 예측을 수행하는 함수
def batch_update_predict(test_slice, model, batch_size=24):
    y_pred_local = []
    for i in range(0, len(test_slice), batch_size):
        # 배치 크기가 남은 데이터보다 클 수 있으므로 min(batch_size, 남은 데이터 수)로 조정
        n_periods = min(batch_size, len(test_slice) - i)
        fc, conf = model.predict(n_periods=n_periods, return_conf_int=True)
        y_pred_local.extend(fc.tolist())  # 예측 결과를 리스트로 추가

        # 배치 단위로 모델 업데이트
        for j in range(batch_size):
            if i + j < len(test_slice):
                model.update(test_slice.iloc[i + j])
        
        # 배치가 끝나면 로그 기록
        logging.info(f"Batch {i//batch_size + 1} completed with {min(batch_size, len(test_slice)-i)} samples.")
    return y_pred_local

# 데이터 분할
train_df = df_filtered[~(df_filtered['_type'] == 'train') & ~(df_filtered['_type'] == 'test')].reset_index(drop=True)
test_df = df_filtered[(pd.to_datetime(df_filtered['ID']) >= "2023-01-01 00:00:00") & (pd.to_datetime(df_filtered['ID']) <= end_date)].reset_index(drop=True)

# 여러 변수에 대한 롤링 예측 및 평가 함수
def rolling_forecast_for_variables(train, test, variables, batch_size=24):
    results = {}  # 각 변수의 예측 결과를 저장할 딕셔너리

    for var in variables:
        logging.info(f"Processing variable: {var}")
        
        # 해당 변수의 train, test 데이터
        train_var = train[var]
        test_var = test[var]

        # 최적 차분 수 결정 (KPSS & ADF 테스트)
        kpss_diffs = pm.arima.ndiffs(train_var, alpha=0.05, test='kpss', max_d=5)
        adf_diffs = pm.arima.ndiffs(train_var, alpha=0.05, test='adf', max_d=5)
        n_diffs = max(kpss_diffs, adf_diffs)
        
        logging.info(f"Optimal differencing for {var}: {n_diffs}")

        # 모델 생성 및 학습
        model = pm.auto_arima(train_var, d=n_diffs, seasonal=False, trace=False)

        # 코어 개수에 맞게 데이터 분할
        n_cores = mp.cpu_count()
        test_splits = np.array_split(test_var, n_cores)

        # 병렬 처리 실행 (배치 예측 적용)
        with mp.Pool(processes=n_cores) as pool:
            results_splits = pool.starmap(batch_update_predict, [(split, model, batch_size) for split in test_splits])

        # 결과 병합
        y_pred = [item for sublist in results_splits for item in sublist]

        # MAPE 계산
        mape_score = MAPE(test_var, y_pred)
        logging.info(f"MAPE for {var}: {mape_score:.3f}")
        
        # 예측 결과 저장
        results[var] = {'test': test_var, 'pred': y_pred, 'MAPE': mape_score}

        # 테스트 데이터의 ID, 실제값, 예측값을 저장할 데이터프레임 생성
        test_pred_df = pd.DataFrame({
            'ID': test['ID'].values,
            'test': test_var.values,
            'pred': y_pred
        })
        
        # 결과를 CSV로 저장 (temp 폴더 안에 저장)
        test_pred_df.to_csv(os.path.join(current_dir, f'model_results_{var}.csv'), index=False)
        
        # 시각화 (차트도 temp 폴더 안에 저장)
        plt.figure(figsize=(8, 6))
        # plt.plot(train_var.index, train_var, label='Train', color='blue')
        plt.plot(test_var.index, test_var, label='Test', color='orange')
        plt.plot(test_var.index, y_pred, label='Predicted', color='green')
        # plt.xlim([train_var.index[0], test_var.index[-1]])
        plt.legend()
        plt.title(f'{var} - Train, Test, and Predicted')
        plt.savefig(os.path.join(current_dir, f'plot_{var}.png'))
        plt.close()  # 백그라운드 작업 시 show() 대신 close()로 메모리 절약
        gc.collect()

    return results

# 주요 변수 리스트
# variables = ['coinbase_premium_index', 'funding_rates', 'long_liquidations_usd', 
#              'short_liquidations_usd', 'open_interest', 'buy_sell_ratio', 'buy_volume',
#              'sell_volume']  # 여기에 사용할 변수들을 추가
# variables = ['funding_rates', 'long_liquidations_usd', 
#              'short_liquidations_usd', 'open_interest', 'buy_sell_ratio', 'buy_volume',
#              'sell_volume', 'coinbase_premium_index'] 
variables = ['long_liquidations_usd', 
             'short_liquidations_usd', 'open_interest', 'buy_sell_ratio', 'buy_volume',
             'sell_volume', 'coinbase_premium_index'] 
# variables = ['funding_rates']

if __name__ == '__main__':
    # 데이터프레임 로드
    # df_filtered = pd.read_csv('your_data_file.csv')  # 데이터 로드 부분 예시
    
    # 변수별 롤링 예측 실행
    results = rolling_forecast_for_variables(train_df, test_df, variables, batch_size=1)
    
    # 완료 로그
    logging.info("All variables processed.")