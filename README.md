![header](https://capsule-render.vercel.app/api?type=waving&color=gradient&width=max&height=250&section=header&text=비트코인-상승/하락-시계열분류예측&desc=RecSys05-오곡밥&fontSize=40&fontColor=FFFFFF&fontAlignY=40)

## 💡 프로젝트 개요

#### 프로젝트 소개
이번 프로젝트는 비트코인의 1시간 뒤 가격 변동을 예측하는 모델을 개발하는 것을 목표로 합니다. 암호화폐는 높은 변동성을 가지며, 가격 예측이 어려운 분야로 알려져 있습니다. 이를 해결하기 위해 다양한 머신러닝 모델과 기술을 적용하여 예측 성능을 향상시키고자 했습니다.

#### 데이터 소개
제공된 데이터는 크립토퀀트에서 정제된 온체인 데이터와 시장 데이터를 포함합니다. 네트워크 데이터는 블록체인 활동을, 시장 데이터는 거래량 및 청산량 등을 나타내며, 이 두 가지 데이터를 활용하여 예측 모델을 개발했습니다.

#### 예측 대상 (Target)
타겟은 '이전 시점 대비 현재 시점의 비트코인 가격 등락 범주'로, -0.5% 미만, -0.5% 이상 0% 미만, 0% 이상 0.5% 미만, 0.5% 이상의 네 가지 범주로 분류됩니다.

#### 프로젝트 목표
비트코인의 가격 변동성을 정확히 예측하여 투자자들의 의사 결정을 돕는 것이 목표입니다. 데이터 전처리, 모델 개발 및 튜닝을 통해 최적의 예측 성능을 달성하고자 노력했습니다.

<br>

## 🍚 팀원 소개

|문원찬|안규리|오소영|오준혁|윤건욱|황진욱|
|:---:|:---:|:---:|:---:|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/a29cbbd9-0cde-495a-bd7e-90f20759f3d1" width="100"/> | <img src="https://github.com/user-attachments/assets/c619ed82-03f3-4d48-9bba-dd60408879f9" width="100"/> | <img src="https://github.com/user-attachments/assets/1b0e54e6-57dc-4c19-97f5-69b7e6f3a9b4" width="100"/> | <img src="https://github.com/user-attachments/assets/67d19373-8cac-4676-bde1-b0637921cf7f" width="100"/> | <img src="https://github.com/user-attachments/assets/f91dd46e-9f1a-42e7-a939-db13692f4098" width="100"/> | <img src="https://github.com/user-attachments/assets/69bbb039-752e-4448-bcaa-b8a65015b778" width="100"/> |
| [![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/WonchanMoon)|[![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/notmandarin)|[![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/irrso)|[![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/ojunhyuk99)|[![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/YoonGeonWook)|[![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/hw01931)|

<br>

## ⚙️ 개발 환경
#### OS: Linux (5.4.0-99-generic, x86_64)
#### GPU: Tesla V100-SXM2-32GB (CUDA Version: 12.2)


<br>

## 🔧 기술 스택
#### 프로그래밍 언어 <img src="https://img.shields.io/badge/python-3776AB.svg?style=flat-square&logo=python&logoColor=white"/>
#### 데이터 분석 및 전처리 <img src="https://img.shields.io/badge/numpy-013243.svg?style=flat-square&logo=numpy&logoColor=white"/> <img src="https://img.shields.io/badge/pandas-150458.svg?style=flat-square&logo=pandas&logoColor=white"/>
#### EDA(탐색적 데이터 분석) <img src="https://img.shields.io/badge/sweetviz-E76F51.svg?style=flat-square&logoColor=white"/> <img src="https://img.shields.io/badge/ydata_profiling-E76F51.svg?style=flat-square&logoColor=white"/>
#### 모델 학습 및 평가 <img src="https://img.shields.io/badge/LightGBM-00B050.svg?style=flat-square&logoColor=white"/> <img src="https://img.shields.io/badge/XGBoost-FF5722.svg?style=flat-square&logoColor=white"/> <img src="https://img.shields.io/badge/CatBoost-03A9F4.svg?style=flat-square&logoColor=white"/> <img src="https://img.shields.io/badge/scikit--learn-F7931E.svg?style=flat-square&logo=scikitlearn&logoColor=white"/> <img src="https://img.shields.io/badge/pytorch-EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white"/>
#### 시각화 도구 <img src="https://img.shields.io/badge/plotly-3F4F75.svg?style=flat-square&logo=plotly&logoColor=white"/> <img src="https://img.shields.io/badge/matplotlib-FFD700.svg?style=flat-square&logoColor=white"/> 
#### 개발환경 <img src="https://img.shields.io/badge/jupyter-F37626.svg?style=flat-square&logo=jupyter&logoColor=white"/>
#### 기타 <img src="https://img.shields.io/badge/tqdm-FFC107.svg?style=flat-square&logo=tqdm&logoColor=black"/>



