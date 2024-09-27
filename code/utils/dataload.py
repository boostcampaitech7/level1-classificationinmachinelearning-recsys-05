import os
import pandas as pd
from typing import List, Dict
from tqdm import tqdm

def load_data(data_path: str) -> Dict[str, pd.DataFrame]:
    # 기본 데이터 로드
    train_df = pd.read_csv(os.path.join(data_path, "train.csv")).assign(_type="train")
    test_df = pd.read_csv(os.path.join(data_path, "test.csv")).assign(_type="test")
    submission_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    df = pd.concat([train_df, test_df], axis=0)

    # HOURLY_ 파일 로드
    file_names = [f for f in os.listdir(data_path) if f.startswith("HOURLY_") and f.endswith(".csv")]
    file_dict = {f.replace(".csv", ""): pd.read_csv(os.path.join(data_path, f)) for f in file_names}

    for _file_name, _df in tqdm(file_dict.items()):
        _rename_rule = {
            col: f"{_file_name.lower()}_{col.lower()}" if col != "datetime" else "ID"
            for col in _df.columns
        }
        _df = _df.rename(_rename_rule, axis=1)
        df = df.merge(_df, on="ID", how="left")

    return {
        "train": train_df,
        "test": test_df,
        "submission": submission_df,
        "combined": df
    }

def load_total_data(data_path: str) -> pd.DataFrame:
    # 기본 데이터 로드
    train_df = pd.read_csv(os.path.join(data_path, "train.csv")).assign(_type="train")
    test_df = pd.read_csv(os.path.join(data_path, "test.csv")).assign(_type="test")
    df = pd.concat([train_df, test_df], axis=0)

    # HOURLY_ 파일 로드
    file_names = [f for f in os.listdir(data_path) if f.startswith("HOURLY_") and f.endswith(".csv")]
    file_dict = {f.replace(".csv", ""): pd.read_csv(os.path.join(data_path, f)) for f in file_names}

    # 날짜 범위 확인
    min_date = None
    max_date = None
    for _df in file_dict.values():
        if min_date is None or _df['datetime'].min() < min_date:
            min_date = _df['datetime'].min()
        if max_date is None or _df['datetime'].max() > max_date:
            max_date = _df['datetime'].max()

    # 전체 ID 범위 생성
    full_id_range = pd.DataFrame({
        'ID': pd.date_range(start=min_date, end=max_date, freq='h')
    })
    full_id_range['ID'] = full_id_range['ID'].astype(str)

    # ID 기준으로 병합
    df = df.merge(full_id_range, on='ID', how='right')

    # 다른 파일들과 병합
    for _file_name, _df in tqdm(file_dict.items(), desc="Merging files"):
        _rename_rule = {
            col: f"{_file_name.lower()}_{col.lower()}" if col != "datetime" else "ID"
            for col in _df.columns
        }
        _df = _df.rename(_rename_rule, axis=1)
        df = df.merge(_df, on="ID", how="left")

    return df

def save_processed_data(df: pd.DataFrame, output_path: str):
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

# if __name__ == "__main__":
#     data_path = "../data"
#     data = load_data(data_path)
#     print("Data loaded successfully.")
#     print(f"Train shape: {data['train'].shape}")
#     print(f"Test shape: {data['test'].shape}")
#     print(f"Combined shape: {data['combined'].shape}")
    
# if __name__ == "__main__":
#     data_path = "../data"
#     output_path = "../data/processed_data.csv"
    
#     processed_df = load_total_data(data_path)
#     print("Data processing completed.")
#     print(f"Processed data shape: {processed_df.shape}")
    
#     save_processed_data(processed_df, output_path)