# RF-DETR + ByteTrack リアルタイムトラッキング

このプロジェクトは、RF-DETR（リアルタイムセグメンテーション）とByteTrackを組み合わせた、リアルタイムマルチオブジェクトトラッキングのサンプル実装です。

## 概要

- **RF-DETR Seg**: Roboflowが開発した高速セグメンテーション対応の物体検出モデル
- **ByteTrack**: シンプルで効果的なマルチオブジェクトトラッキングアルゴリズム
- **Supervision**: 物体検出とトラッキング結果の可視化ライブラリ

## 機能

✅ リアルタイム物体検出  
✅ インスタンスセグメンテーション  
✅ マルチオブジェクトトラッキング  
✅ トラッキングID表示  
✅ マスク可視化  

## セットアップ

### 必要な環境

- Python >= 3.10
- GPU (NVIDIA GPU推奨、CPU でも動作可能)

### インストール

```bash
# リポジトリのクローン
git clone <repository-url>
cd tracker_test

# 依存パッケージのインストール
uv pip install -r requirements.txt
```

または、個別にインストール：

```bash
uv pip install rfdetr supervision opencv-python matplotlib pyqt5
```

## 使用方法

### 基本的な実行

```bash
uv run main.py
```

プログラムは `/dev/video4` からビデオストリームを取得し、検出とトラッキングを行います。

### ビデオファイルからの処理

[main.py](main.py) の以下の行を編集してビデオファイルパスを指定：

```python
video_capture = cv2.VideoCapture("path/to/video.mp4")
```

## コード説明

### 初期化

```python
import cv2
import supervision as sv
from rfdetr import RFDETRSegMedium

# モデルとトラッカーの初期化
tracker = sv.ByteTrack()
model = RFDETRSegMedium()

# アノテーター（描画ツール）の初期化
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
mask_annotator = sv.MaskAnnotator()
```

### メインループ

```python
while True:
    success, frame_bgr = video_capture.read()
    if not success:
        break

    # BGR から RGB に変換
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # 物体検出とセグメンテーション
    detections = model.predict(frame_rgb)
    
    # トラッキング更新
    detections = tracker.update_with_detections(detections)

    # 結果の描画
    annotated_frame = mask_annotator.annotate(frame_bgr.copy(), detections)
    annotated_frame = box_annotator.annotate(annotated_frame, detections)
    annotated_frame = label_annotator.annotate(
        annotated_frame, 
        detections, 
        labels=detections.tracker_id
    )

    # 表示
    img_display.set_data(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    plt.pause(0.001)
```

## トラッキングアルゴリズム：ByteTrack

ByteTrackは、以下の特徴を持つマルチオブジェクトトラッキング手法です：

- **シンプル**: 複雑なモデルやデータ関連付けを必要としない
- **高速**: リアルタイム処理に対応
- **効果的**: 各種ベンチマークで高いスコアを獲得

### ByteTrack の性能比較

| トラッカー | IDF1 | HOTA | DetA | AssA | MOTA |
|----------|------|------|------|------|------|
| SORT     | 58.4 | 69.9 | 67.2 | 70.9 | 81.6 |
| ByteTrack| 60.1 | 73.2 | 74.1 | 73.0 | 84.0 |

## Supervision ライブラリについて

Supervisionは、物体検出とトラッキングの結果を簡単に処理・可視化するツールです：

### 主な用途

- **Detections**: 検出結果の統一的な表現
- **BoxAnnotator**: バウンディングボックスの描画
- **LabelAnnotator**: ラベルの描画
- **MaskAnnotator**: セグメンテーションマスクの描画
- **Tracker**: マルチオブジェクトトラッキング

### 使用例

```python
# 検出結果の取得
detections = model.predict(image)

# トラッキング
detections = tracker.update_with_detections(detections)

# 描画
annotated = box_annotator.annotate(image, detections)
```

## トラッキング対応の他のアルゴリズム

Roboflow Trackersライブラリは、複数のトラッキングアルゴリズムをサポートしています：

- **SORT**: シンプルなカルマンフィルターベースのトラッカー
- **ByteTrack**: 高性能なニューラルネットワークベースのトラッカー
- **OC-SORT**: ByteTrackの改良版（近日対応予定）
- **BoT-SORT**: ボテックスベースのトラッカー（近日対応予定）

## トラブルシューティング

### GUI表示エラー

```
cv2.error: ... The function is not implemented
```

**解決策**: 依存ライブラリをインストール

```bash
sudo apt-get install libgtk2.0-dev pkg-config
```

### Matplotlibバックエンド エラー

Tkinterが使用できない場合は、Qt5を使用：

```python
matplotlib.use('Qt5Agg')
```

## パフォーマンス最適化

### 入力フレームのリサイズ

```python
# フレームをリサイズして高速化
frame_resized = cv2.resize(frame_rgb, (576, 576))
detections = model.predict(frame_resized)
```

### モデルの最適化

```python
# 推論用に最適化（オプション）
model.optimize_for_inference()
```

## 参考資料

- **RF-DETR**: https://github.com/roboflow/rf-detr
- **Roboflow Trackers**: https://github.com/roboflow/trackers
- **Supervision**: https://github.com/roboflow/supervision
- **ByteTrack**: https://arxiv.org/abs/2110.06864

## ライセンス

このプロジェクトは Apache 2.0 ライセンスの下でリリースされています。

## 関連プロジェクト

- RF-DETR: リアルタイムセグメンテーション対応検出モデル
- Roboflow Trackers: マルチオブジェクトトラッキングライブラリ
- Supervision: 検出・トラッキング結果の処理・可視化

## サポート

問題が発生した場合は、GitHubのIssueページで報告してください。
