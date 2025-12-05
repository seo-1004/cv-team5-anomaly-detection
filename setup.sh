#!/bin/bash
set -e  # 에러 발생 시 즉시 종료

########################################
# 기본 경로 설정 (현재 작업 디렉토리)
########################################
BASE_DIR="$(pwd)"
echo "[INFO] BASE_DIR = $BASE_DIR"

########################################
# 데이터 다운로드
########################################

DATA_ZIP="$BASE_DIR/data.zip"
DATA_URL="https://github.com/seo-1004/cv-team5-anomaly-detection/releases/download/v1.0/data_release.zip"

echo "[INFO] Downloading dataset..."
wget -O "$DATA_ZIP" "$DATA_URL"

echo "[INFO] Extracting dataset..."
mkdir -p "$BASE_DIR/data"
unzip -o "$DATA_ZIP" -d "$BASE_DIR/data"

rm "$DATA_ZIP"

echo "[INFO] Data download complete!"
echo "[INFO] Data saved under: $BASE_DIR/data"

########################################
# 모델 다운로드
########################################

MODEL_DIR="$BASE_DIR/checkpoints/autoencoder"
mkdir -p "$MODEL_DIR"
MODEL_FILE="best_model_epoch_100.pth"

MODEL_URL="https://github.com/seo-1004/cv-team5-anomaly-detection/releases/download/v1/best_model_epoch_100.pth"

echo "[INFO] Downloading model..."
echo "From: $MODEL_URL"
echo "To:   $MODEL_DIR/$MODEL_FILE"

wget -O "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL"

echo "[INFO] Model download complete!"
echo "[INFO] Saved at: $MODEL_DIR/$MODEL_FILE"

########################################
# Python 패키지 설치
########################################
REQ_FILE="$BASE_DIR/requirements.txt"

if [ -f "$REQ_FILE" ]; then
  echo "[INFO] Installing Python dependencies..."
  pip install -r "$REQ_FILE"
else
  echo "[WARN] requirements.txt not found at $REQ_FILE. Skipping pip install."
fi

echo ""
echo "=============================================="
echo "[SETUP COMPLETE] All data & model are ready!"
echo "=============================================="
