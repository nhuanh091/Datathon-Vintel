# 📊 Sales Forecasting — Kaggle Competition

Dự đoán doanh thu (`Revenue`) và chi phí hàng bán (`COGS`) theo ngày cho giai đoạn **2023-01-01 → 2024-07-01** dựa trên dữ liệu lịch sử từ 2012–2022.

---

## Cấu trúc thư mục

```
├── baseline_full.ipynb        # Notebook chính — toàn bộ pipeline
├── submission.csv             # File nộp bài cuối cùng
├── README.md                  # File này
└── data/                      # Thư mục chứa dữ liệu (không upload lên GitHub)
    ├── sales.csv
    ├── sample_submission.csv
    ├── promotions.csv
    ├── orders.csv
    ├── returns.csv
    ├── reviews.csv
    ├── web_traffic.csv
    ├── customers.csv
    ├── geography.csv
    ├── inventory.csv
    ├── order_items.csv
    ├── payments.csv
    ├── products.csv
    └── shipments.csv
```

---

## Yêu cầu môi trường

| Thư viện | Phiên bản khuyến nghị |
|---|---|
| Python | ≥ 3.9 |
| pandas | ≥ 1.5 |
| numpy | ≥ 1.23 |
| xgboost | ≥ 1.7 |
| scikit-learn | ≥ 1.2 |
| matplotlib | ≥ 3.6 |
| seaborn | ≥ 0.12 |
| shap | ≥ 0.42 *(tuỳ chọn)* |

### Cài đặt nhanh

```bash
pip install pandas numpy xgboost scikit-learn matplotlib seaborn shap
```

---

## Hướng dẫn chạy lại kết quả

### Trên Kaggle (khuyến nghị)

1. Fork notebook `baseline_full.ipynb` lên Kaggle
2. Attach dataset của competition vào notebook
3. Mở notebook → **Run All**
4. File `submission.csv` sẽ được tạo tự động trong thư mục output

> Nếu không có GPU, đổi dòng `'device': 'cuda'` → `'device': 'cpu'` trong `MODEL_PARAMS` (Cell 13)

### Chạy local

```bash
# 1. Clone repo
git clone <url-repo-này>
cd <tên-repo>

# 2. Tạo thư mục data và đặt tất cả file CSV vào đó
mkdir data
# copy tất cả file .csv vào thư mục data/

# 3. Cài thư viện
pip install pandas numpy xgboost scikit-learn matplotlib seaborn shap

# 4. Chạy notebook
jupyter notebook baseline_full.ipynb
# Chọn Kernel → Restart & Run All
```

---

##  Pipeline tổng quan

```
Raw Data (13 CSV files)
        │
        ▼
┌─────────────────────────────────────────┐
│  Section 2: Load & Inspect Data         │
│  • Đọc 7 bảng chính có liên quan        │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Section 3: Feature Engineering         │
│  • Calendar features (11 features)      │
│  • Promotion features (7 features)      │
│  • Order features (6 features)          │
│  • Return features (2 features)         │
│  • Review features (2 features)         │
│  • Web traffic features (4 features)    │
│  • Lag features — 1/7/14/30 ngày        │
│  • Rolling mean/std — 7/14/30 ngày      │
│                        ──────────────── │
│                        Tổng: ~45 features│
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Section 4: Prepare Train/Test          │
│  • Log-transform targets (log1p)        │
│  • Tách X_train, y_rev, y_cogs, X_test  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Section 5: Cross-Validation            │
│  • TimeSeriesSplit (5 folds)            │
│  • Báo cáo MAE / RMSE / R² mỗi fold    │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Section 6: Train Final Models          │
│  • XGBRegressor cho Revenue             │
│  • XGBRegressor cho COGS                │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Section 7: Evaluate (Hold-out 2021-22) │
│  • MAE / RMSE / R² cho Revenue & COGS   │
│  • Biểu đồ Actual vs Predicted          │
│  • Biểu đồ Residuals                    │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Section 8: Feature Importance & SHAP   │
│  • XGBoost plot_importance (gain)       │
│  • SHAP bar plot + beeswarm             │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Section 9: Export submission.csv       │
└─────────────────────────────────────────┘
```

---

## Metrics đánh giá (theo đề bài)

| Metric | Công thức | Mục tiêu |
|--------|-----------|----------|
| **MAE** | $\frac{1}{n}\sum_{i=1}^{n}\|F_i - A_i\|$ | Càng thấp càng tốt |
| **RMSE** | $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(F_i - A_i)^2}$ | Càng thấp càng tốt |
| **R²** | $1 - \frac{\sum(A_i-F_i)^2}{\sum(A_i-\bar{A})^2}$ | Càng gần 1 càng tốt |

---

## Mô hình sử dụng

**XGBoost Regressor** với cấu hình:
- `n_estimators = 1000` với early stopping (50 rounds)
- `learning_rate = 0.05`
- `max_depth = 6`
- `subsample = 0.8`, `colsample_bytree = 0.8`
- Target được log-transform (`log1p`) trước khi train → `expm1` khi predict

---

## Ghi chú

- File dữ liệu **không được** upload lên GitHub (kích thước lớn, thuộc về Kaggle competition)
- Chạy toàn bộ notebook mất khoảng **10–15 phút** tuỳ cấu hình máy
- SHAP visualization có thể bỏ qua nếu không cài `shap` — notebook vẫn chạy bình thường
