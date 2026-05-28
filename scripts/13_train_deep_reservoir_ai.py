from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORT_TABLES = ROOT / "reports" / "tables"
META = ROOT / "data" / "metadata"
MODELS = ROOT / "models"

REPORT_TABLES.mkdir(parents=True, exist_ok=True)
META.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PROCESSED / "01_reservoir_sigungu_daily.csv"

FORECAST_OUT = PROCESSED / "ai_gru_reservoir_forecast_by_sigungu.csv"
ANOMALY_OUT = PROCESSED / "ai_autoencoder_anomaly_by_sigungu.csv"
SUMMARY_OUT = REPORT_TABLES / "ai_sigungu_deep_summary.csv"
REPORT_OUT = META / "deep_ai_model_report.md"

SEQ_LEN = 30
HORIZON = 7
BATCH_SIZE = 128
EPOCHS_GRU = 80
EPOCHS_AE = 80
LR = 1e-3
RANDOM_SEED = 42


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def read_data():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    required = [
        "date",
        "sigungu",
        "avg_reservoir_rate",
        "min_reservoir_rate",
        "max_reservoir_rate",
        "reservoir_risk_score",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns={missing}. Current columns={df.columns.tolist()}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = [
        "avg_reservoir_rate",
        "min_reservoir_rate",
        "max_reservoir_rate",
        "reservoir_count",
        "low_reservoir_count_40",
        "low_reservoir_count_30",
        "total_effective_capacity",
        "reservoir_risk_score",
    ]

    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date", "sigungu", "avg_reservoir_rate"]).copy()
    df = df.sort_values(["sigungu", "date"]).reset_index(drop=True)

    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear
    df["sin_day"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["cos_day"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

    return df


FEATURE_COLS = [
    "avg_reservoir_rate",
    "min_reservoir_rate",
    "max_reservoir_rate",
    "reservoir_count",
    "low_reservoir_count_40",
    "low_reservoir_count_30",
    "total_effective_capacity",
    "reservoir_risk_score",
    "sin_day",
    "cos_day",
]


def build_sequences(df):
    samples = []

    for sigungu, g in df.groupby("sigungu"):
        g = g.sort_values("date").reset_index(drop=True)
        values = g[FEATURE_COLS].copy()

        values = values.replace([np.inf, -np.inf], np.nan)
        values = values.fillna(values.median(numeric_only=True)).fillna(0)

        target = g["avg_reservoir_rate"].values
        dates = g["date"].values

        max_i = len(g) - SEQ_LEN - HORIZON + 1
        if max_i <= 0:
            continue

        for i in range(max_i):
            x = values.iloc[i:i + SEQ_LEN].values.astype(np.float32)
            y = float(target[i + SEQ_LEN + HORIZON - 1])
            target_date = pd.Timestamp(dates[i + SEQ_LEN + HORIZON - 1])
            base_date = pd.Timestamp(dates[i + SEQ_LEN - 1])
            current_rate = float(target[i + SEQ_LEN - 1])

            samples.append({
                "sigungu": sigungu,
                "base_date": base_date,
                "target_date": target_date,
                "current_rate": current_rate,
                "x": x,
                "y": y,
            })

    if not samples:
        raise RuntimeError("No sequence samples generated. Check input data length.")

    return samples


class SeqDataset(Dataset):
    def __init__(self, samples, x_scaler=None, fit_scaler=False):
        self.meta = []
        xs = np.stack([s["x"] for s in samples], axis=0)
        ys = np.array([s["y"] for s in samples], dtype=np.float32)

        n, t, f = xs.shape
        flat = xs.reshape(-1, f)

        if fit_scaler:
            self.x_scaler = StandardScaler()
            self.x_scaler.fit(flat)
        else:
            self.x_scaler = x_scaler

        xs_scaled = self.x_scaler.transform(flat).reshape(n, t, f).astype(np.float32)

        self.xs = xs_scaled
        self.ys = ys

        for s in samples:
            self.meta.append({
                "sigungu": s["sigungu"],
                "base_date": s["base_date"],
                "target_date": s["target_date"],
                "current_rate": s["current_rate"],
            })

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return torch.tensor(self.xs[idx], dtype=torch.float32), torch.tensor(self.ys[idx], dtype=torch.float32)


class GRUForecaster(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.15):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class SequenceAutoEncoder(nn.Module):
    def __init__(self, seq_len, n_features, latent=32):
        super().__init__()
        input_dim = seq_len * n_features

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

        self.seq_len = seq_len
        self.n_features = n_features

    def forward(self, x):
        b = x.shape[0]
        flat = x.reshape(b, -1)
        z = self.encoder(flat)
        recon = self.decoder(z)
        return recon.reshape(b, self.seq_len, self.n_features)


def split_by_time(samples):
    samples = sorted(samples, key=lambda s: s["base_date"])
    split_idx = int(len(samples) * 0.8)
    train = samples[:split_idx]
    valid = samples[split_idx:]
    return train, valid


def train_gru(train_ds, valid_ds, device):
    model = GRUForecaster(n_features=len(FEATURE_COLS)).to(device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    best_mae = float("inf")
    best_state = None
    history = []

    for epoch in range(1, EPOCHS_GRU + 1):
        model.train()
        losses = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        model.eval()
        preds = []
        trues = []

        with torch.no_grad():
            for x, y in valid_loader:
                x = x.to(device)
                pred = model(x).cpu().numpy()
                preds.extend(pred.tolist())
                trues.extend(y.numpy().tolist())

        mae = mean_absolute_error(trues, preds)
        r2 = r2_score(trues, preds) if len(set(trues)) > 1 else 0.0

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "valid_mae": float(mae),
            "valid_r2": float(r2),
        })

        if mae < best_mae:
            best_mae = mae
            best_state = model.state_dict()

        if epoch % 10 == 0 or epoch == 1:
            print(f"[GRU] epoch={epoch:03d} loss={np.mean(losses):.4f} valid_mae={mae:.4f} r2={r2:.4f}", flush=True)

    model.load_state_dict(best_state)

    return model, pd.DataFrame(history)


def train_autoencoder(train_ds, valid_ds, device):
    model = SequenceAutoEncoder(seq_len=SEQ_LEN, n_features=len(FEATURE_COLS), latent=32).to(device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    history = []

    for epoch in range(1, EPOCHS_AE + 1):
        model.train()
        losses = []

        for x, _ in train_loader:
            x = x.to(device)

            optimizer.zero_grad()
            recon = model(x)
            loss = loss_fn(recon, x)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        model.eval()
        valid_losses = []

        with torch.no_grad():
            for x, _ in valid_loader:
                x = x.to(device)
                recon = model(x)
                loss = loss_fn(recon, x)
                valid_losses.append(loss.item())

        valid_loss = float(np.mean(valid_losses))
        train_loss = float(np.mean(losses))

        history.append({
            "epoch": epoch,
            "train_recon_loss": train_loss,
            "valid_recon_loss": valid_loss,
        })

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = model.state_dict()

        if epoch % 10 == 0 or epoch == 1:
            print(f"[AE] epoch={epoch:03d} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f}", flush=True)

    model.load_state_dict(best_state)

    return model, pd.DataFrame(history)


def predict_dataset(model, ds, device):
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()

    preds = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            p = model(x).cpu().numpy()
            preds.extend(p.tolist())

    return np.array(preds)


def reconstruction_errors(model, ds, device):
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()

    errors = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            recon = model(x)
            err = ((recon - x) ** 2).mean(dim=(1, 2)).cpu().numpy()
            errors.extend(err.tolist())

    return np.array(errors)


def build_latest_dataset(samples, x_scaler):
    latest = {}

    for s in samples:
        sigungu = s["sigungu"]
        if sigungu not in latest or s["base_date"] > latest[sigungu]["base_date"]:
            latest[sigungu] = s

    latest_samples = list(latest.values())
    ds = SeqDataset(latest_samples, x_scaler=x_scaler, fit_scaler=False)

    return latest_samples, ds


def risk_level(score):
    if score >= 80:
        return "심각"
    if score >= 60:
        return "경계"
    if score >= 40:
        return "주의"
    return "낮음"


def main():
    seed_everything(RANDOM_SEED)
    device = get_device()
    print(f"[AquaGuard AI] Deep AI training started. device={device}")

    df = read_data()
    print(f"[INFO] input rows={len(df):,}, sigungu={df['sigungu'].nunique()}, date={df['date'].min()}~{df['date'].max()}")

    samples = build_sequences(df)
    print(f"[INFO] sequence samples={len(samples):,}, seq_len={SEQ_LEN}, horizon={HORIZON}")

    train_samples, valid_samples = split_by_time(samples)

    train_ds = SeqDataset(train_samples, fit_scaler=True)
    valid_ds = SeqDataset(valid_samples, x_scaler=train_ds.x_scaler, fit_scaler=False)

    print(f"[INFO] train={len(train_ds):,}, valid={len(valid_ds):,}")

    gru, gru_history = train_gru(train_ds, valid_ds, device)
    ae, ae_history = train_autoencoder(train_ds, valid_ds, device)

    latest_samples, latest_ds = build_latest_dataset(samples, train_ds.x_scaler)

    pred_7d = predict_dataset(gru, latest_ds, device).clip(0, 100)
    recon_err = reconstruction_errors(ae, latest_ds, device)

    err_scaler = MinMaxScaler(feature_range=(0, 100))
    anomaly_score = err_scaler.fit_transform(recon_err.reshape(-1, 1)).reshape(-1)

    forecast_rows = []
    anomaly_rows = []

    for sample, pred, err, score in zip(latest_samples, pred_7d, recon_err, anomaly_score):
        current_rate = sample["current_rate"]
        forecast_drop = max(0.0, current_rate - pred)

        forecast_risk_score = np.clip((100 - pred) * 0.70 + forecast_drop * 0.30, 0, 100)
        anomaly_risk_score = float(np.clip(score, 0, 100))

        forecast_rows.append({
            "sigungu": sample["sigungu"],
            "base_date": sample["base_date"],
            "target_date": sample["target_date"],
            "current_avg_reservoir_rate": current_rate,
            "pred_avg_reservoir_rate_7d": float(pred),
            "forecast_drop_7d": float(forecast_drop),
            "forecast_risk_score": float(forecast_risk_score),
            "forecast_risk_level": risk_level(float(forecast_risk_score)),
        })

        anomaly_rows.append({
            "sigungu": sample["sigungu"],
            "base_date": sample["base_date"],
            "current_avg_reservoir_rate": current_rate,
            "reconstruction_error": float(err),
            "autoencoder_anomaly_score": anomaly_risk_score,
            "autoencoder_anomaly_level": risk_level(anomaly_risk_score),
        })

    forecast_df = pd.DataFrame(forecast_rows).sort_values("forecast_risk_score", ascending=False)
    anomaly_df = pd.DataFrame(anomaly_rows).sort_values("autoencoder_anomaly_score", ascending=False)

    summary = forecast_df.merge(
        anomaly_df[["sigungu", "autoencoder_anomaly_score", "autoencoder_anomaly_level"]],
        on="sigungu",
        how="left",
    )

    summary["deep_ai_risk_score"] = (
        summary["forecast_risk_score"] * 0.60
        + summary["autoencoder_anomaly_score"] * 0.40
    ).clip(0, 100)

    summary["deep_ai_risk_level"] = summary["deep_ai_risk_score"].apply(risk_level)
    summary = summary.sort_values("deep_ai_risk_score", ascending=False).reset_index(drop=True)
    summary["deep_ai_rank"] = np.arange(1, len(summary) + 1)

    summary = summary[[
        "deep_ai_rank",
        "sigungu",
        "deep_ai_risk_score",
        "deep_ai_risk_level",
        "current_avg_reservoir_rate",
        "pred_avg_reservoir_rate_7d",
        "forecast_drop_7d",
        "forecast_risk_score",
        "forecast_risk_level",
        "autoencoder_anomaly_score",
        "autoencoder_anomaly_level",
        "base_date",
        "target_date",
    ]]

    forecast_df.to_csv(FORECAST_OUT, index=False, encoding="utf-8-sig")
    anomaly_df.to_csv(ANOMALY_OUT, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")

    torch.save({
        "model_state_dict": gru.state_dict(),
        "feature_cols": FEATURE_COLS,
        "seq_len": SEQ_LEN,
        "horizon": HORIZON,
        "model_type": "GRUForecaster",
    }, MODELS / "gru_reservoir_forecast.pt")

    torch.save({
        "model_state_dict": ae.state_dict(),
        "feature_cols": FEATURE_COLS,
        "seq_len": SEQ_LEN,
        "model_type": "SequenceAutoEncoder",
    }, MODELS / "autoencoder_reservoir_anomaly.pt")

    joblib.dump(train_ds.x_scaler, MODELS / "deep_ai_feature_scaler.joblib")

    gru_history.to_csv(REPORT_TABLES / "ai_gru_training_history.csv", index=False, encoding="utf-8-sig")
    ae_history.to_csv(REPORT_TABLES / "ai_autoencoder_training_history.csv", index=False, encoding="utf-8-sig")

    valid_pred = predict_dataset(gru, valid_ds, device)
    valid_y = np.array([s["y"] for s in valid_samples], dtype=np.float32)

    mae = mean_absolute_error(valid_y, valid_pred)
    r2 = r2_score(valid_y, valid_pred) if len(set(valid_y)) > 1 else 0.0

    report = f"""# AquaGuard AI Deep AI Model Report

## AI 모델 목적

AquaGuard AI에 실제 학습 기반 AI 모듈을 추가하기 위해 PyTorch 기반 GRU 예측 모델과 AutoEncoder 이상탐지 모델을 학습했다.

## 입력 데이터

- 입력 파일: data/processed/01_reservoir_sigungu_daily.csv
- 시퀀스 길이: {SEQ_LEN}일
- 예측 시점: {HORIZON}일 후
- 사용 시·군 수: {df['sigungu'].nunique()}
- 전체 시퀀스 샘플 수: {len(samples):,}
- 학습 샘플 수: {len(train_ds):,}
- 검증 샘플 수: {len(valid_ds):,}
- 학습 장치: {device}

## AI-1. GRU 저수율 예측 모델

- 모델: PyTorch GRU
- 목적: 최근 30일 저수율 패턴을 기반으로 7일 후 시·군 평균 저수율 예측
- 검증 MAE: {mae:.4f}
- 검증 R2: {r2:.4f}
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
"""

    REPORT_OUT.write_text(report, encoding="utf-8")

    print()
    print("[Saved]")
    print(f"- {FORECAST_OUT}")
    print(f"- {ANOMALY_OUT}")
    print(f"- {SUMMARY_OUT}")
    print(f"- {REPORT_OUT}")
    print(f"- {MODELS / 'gru_reservoir_forecast.pt'}")
    print(f"- {MODELS / 'autoencoder_reservoir_anomaly.pt'}")

    print()
    print("[Deep AI Summary]")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
