# AquaGuard AI Deep AI Model Report

## AI 모델 목적

AquaGuard AI에 실제 학습 기반 AI 모듈을 추가하기 위해 PyTorch 기반 GRU 예측 모델과 AutoEncoder 이상탐지 모델을 학습했다.

## 입력 데이터

- 입력 파일: data/processed/01_reservoir_sigungu_daily.csv
- 시퀀스 길이: 30일
- 예측 시점: 7일 후
- 사용 시·군 수: 14
- 전체 시퀀스 샘플 수: 25,060
- 학습 샘플 수: 20,048
- 검증 샘플 수: 5,012
- 학습 장치: cuda

## AI-1. GRU 저수율 예측 모델

- 모델: PyTorch GRU
- 목적: 최근 30일 저수율 패턴을 기반으로 7일 후 시·군 평균 저수율 예측
- 검증 MAE: 3.5567
- 검증 R2: 0.7204
- 출력 파일: data/processed/ai_gru_reservoir_forecast_by_sigungu.csv
- 모델 파일: models/gru_reservoir_forecast.pt

## AI-2. AutoEncoder 이상탐지 모델

- 모델: PyTorch Sequence AutoEncoder
- 목적: 최근 30일 저수율 패턴의 재구성 오차를 기반으로 평소와 다른 이상 패턴 탐지
- 출력 파일: data/processed/ai_autoencoder_anomaly_by_sigungu.csv
- 모델 파일: models/autoencoder_reservoir_anomaly.pt

## 최종 Deep AI 위험도

deep_ai_risk_score =
0.60 * forecast_risk_score
+ 0.40 * autoencoder_anomaly_score

## 발표 표현

과거 공공데이터는 AI 모델의 기준 패턴 학습에 사용했고,
향후 올담 API의 오늘 또는 최근 1개월 데이터는 최신 추론 입력으로 사용할 수 있다.

MVP에서는 GRU 기반 7일 후 저수율 예측과 AutoEncoder 기반 이상탐지를 통해
단순 규칙 기반 위험도에서 학습 기반 AI 위험도 보정 구조로 확장했다.
