# Project 11 — Ghi chú nhanh Midterm AI Core

## Phạm vi

Midterm có 40 điểm theo README gốc: CNN/training loop (15), ablation và phân tích overfitting (10), vấn đáp/live demo (15). Bản này chỉ tập trung vào AI Core, chưa mở rộng sang Final.

## File cần biết

- `starter/student_core.py`: CNN PyTorch, `Dataset`, train/evaluate, checkpoint và prediction.
- `scripts/prepare_dataset.py`: tải và kiểm tra TrashNet, lọc metadata/ảnh lỗi trước khi split.
- `scripts/run_experiments.py`: 2 baseline, 10 cấu hình CNN, metrics, curves và confusion matrix.
- `starter/app.py`: demo Streamlit, dùng `models/waste_cnn.pth`.
- `scripts/build_presentation.py`: dựng lại `presentation.pdf` từ kết quả PyTorch.

## CNN mặc định

RGB 64×64 → 4 blocks Conv2d(3×3) + BatchNorm2d + ReLU + MaxPool2d (16, 32, 64, 128 channels) → Adaptive Global Average Pooling → Dropout(0.3) → Dense(128) + ReLU → Dropout(0.3) → Dense(6 logits). Có **114.966 tham số trainable** theo model hiện tại. Không dùng pretrained backbone.

`CrossEntropyLoss` nhận logits; chỉ áp dụng Softmax khi cần xác suất dự đoán. Class weights được tính từ nhãn train, không dùng validation/test.

## Lệnh PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/prepare_dataset.py --confirm
python scripts/run_experiments.py --epochs 25 --seed 42
pytest tests -q
python scripts/build_presentation.py --team "Tên nhóm — thành viên"
streamlit run starter/app.py
```

Nếu gặp `ModuleNotFoundError: matplotlib`, cài dependencies bằng `python -m pip install -r requirements.txt` trong đúng môi trường đang chạy script.

## Quy tắc thực nghiệm

1. Mỗi cấu hình khôi phục epoch có validation loss thấp nhất.
2. Chọn model cuối bằng validation macro-F1 trong các cấu hình có dropout > 0; `no_reg` và `drop_0.0` chỉ là ablation.
3. Test set chỉ dùng cho model đã chọn và các baseline cần báo cáo; không chọn model theo test metrics.
4. Giữ cùng split/seed khi so sánh các cấu hình. Một seed không chứng minh ưu thế thống kê.
5. Ảnh trong `data/sample/` là ảnh tổng hợp để kiểm tra hạ tầng; không dùng kết quả `--quick` làm accuracy/F1 trên TrashNet.

## Những ý cần giải thích khi bảo vệ

- **BatchNorm:** ổn định phân phối kích hoạt và hỗ trợ tối ưu.
- **Global Average Pooling:** giảm số tham số so với Flatten trong kiến trúc này.
- **Class weighting:** tăng ảnh hưởng tương đối của lớp ít mẫu.
- **Overfitting của `no_reg`:** train cao hơn validation không có nghĩa regularization vô ích.
- **Macro-F1:** đánh giá cân bằng hơn giữa các lớp khi số ảnh mỗi lớp không đều.
- **Giới hạn:** ảnh 64×64, một seed, dataset test hữu hạn và nhầm lẫn giữa glass/metal/plastic.

## Trạng thái lúc đóng gói

- Unit tests trong môi trường đóng gói: **17 passed**.
- CPU smoke test đã chạy trên ảnh tổng hợp.
- **Full TrashNet chưa chạy** trong môi trường đóng gói do không tải được dataset; vì vậy chưa có test metrics PyTorch chính thức hay checkpoint demo huấn luyện đầy đủ. Trên máy nhóm, hãy chạy full pipeline và kiểm tra `results/summary.md` trước khi đưa số liệu vào slide.
