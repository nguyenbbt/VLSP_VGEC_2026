# VGEC S0 — BARTpho attention + ViT5-base

Hệ thống phân loại lỗi và sửa câu tiếng Việt, tách riêng từ project gốc. Repo chứa code, cấu hình và tests; dữ liệu train/dev/test, prediction và trọng số model được cung cấp riêng.

**Kiến trúc và hành vi S0**

- BARTpho-syllable học đồng thời correction và 5 nhãn lỗi. Encoder → attention pooling có mask → Dropout 0,1 → Linear 5 lớp. Loss = correction CE + 0,2 × classification CE.
- Hai checkpoint BARTpho được chọn độc lập: `best_classification` cho nhãn và `best_correction` cho câu sửa.
- ViT5-base học correction-only và tạo thêm ứng viên.
- Pool giữ đúng luồng cũ: câu gốc, output BARTpho corrector, ViT5 corrector và decoder của checkpoint BARTpho classifier; mỗi generator lấy top-1 với beam search 4.
- **S0 chọn output BARTpho corrector nếu hợp lệ; nếu không, giữ câu gốc.** ViT5 không bỏ phiếu và không được dùng làm fallback trong selector S0. Các ứng viên còn lại phục vụ phân tích pool; không có neural reranker.
- Không có Qwen, BamiBERT, train fold hoặc model phụ trong entrypoint của repo này. Nhãn 0 không tự ép câu sửa trở thành copy.

**Dữ liệu và validation**

Đặt dữ liệu của bạn tại:

```text
data/raw/final_data.json
data/raw/dev.json
data/raw/private_test.json     # chỉ cần khi inference trên tập của bạn
```

Các đường dẫn train/dev nằm trong `configs/data.yaml`. Schema supervised là:

```json
[
  {
    "ID": 1,
    "Input": "Ví dụ câu đầu vào.",
    "Output": {"Error_Type_Id": 0, "Fixed Sentence": []}
  }
]
```

Inference chỉ cần `ID` và `Input`; ví dụ tự tạo nằm trong `examples/input.json`.

Audit loại nguồn trùng/overlap dev, gộp nguồn trùng và ghi hash. Train chia khoảng 85/15 theo `group_id`, phân tầng nhãn, seed 42. Validation bên trong dùng chọn checkpoint/early stopping; dev giữ riêng để báo cáo sau đó. Không chứng minh được lineage đầy đủ của dữ liệu tổng hợp thì kết quả thuộc track experimental, không tự nhận constrained-eligible.

Snapshot dữ liệu của lần chạy gốc có 7.153 dòng → 7.063 sau loại 90 dòng trùng dev. BARTpho: train 6.003 / validation 1.060. ViT5 bỏ 84 dòng thiếu target hoàn chỉnh: train 5.931 / validation 1.048. Đây là thông tin lịch sử, không phải dữ liệu được phân phối trong repo. Cấu hình `expected_train` chỉ ghi số tham khảo; preflight kiểm tra dữ liệu hiện tại theo audit đã tạo.

Hai model pretrained đã khóa commit theo manifest lần chạy gốc:

| Model | Revision |
|---|---|
| vinai/bartpho-syllable | `36eee8b4d648dd99da56462edcda3c5c97f7f3de` |
| VietAI/vit5-base | `2209a38d735ede63e88f5aa52bcdc11a05a37b85` |

**Cài đặt — PowerShell, Python 3.11**

Chạy các lệnh dưới đây tại thư mục S0. Nếu máy có Python Launcher:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[modal]"
python -m modal setup
```

Máy local chỉ cần thư viện nhẹ để audit/submit Modal. GPU image dùng các phiên bản khóa trong `requirements-modal-core.lock`. Để chạy tiny-model tests hoặc inference local, cài thêm:

```powershell
python -m pip install -e ".[train]"
```

**Chuẩn bị dữ liệu và kiểm tra trước train**

```powershell
python -X utf8 -m vgec audit --config configs/data.yaml
python -X utf8 scripts/preflight.py
.\run_final_attention.ps1 -CheckOnly
```

Audit tạo `data/prepared/` và `reports/data/`. Chưa đặt dữ liệu thì preflight dừng với thông báo đường dẫn thiếu, không submit training. Đổi raw/prepared data sau audit thì phải audit lại.

**Train trên Modal**

Kiểm tra profile để dùng đúng tài khoản:

```powershell
python -m modal profile list
```

Runner mặc định dùng L4 như bản S0 gốc. Chạy cả hai smoke rồi cả hai full training theo thứ tự, orchestration detached:

```powershell
.\run_final_attention.ps1 -Gpu L4
```

Có thể chỉ định tài khoản và A100:

```powershell
.\run_final_attention.ps1 -Profile TEN_PROFILE -Gpu A100
```

Các run mới dùng tên `s0-bartpho-attention-smoke-42`, `s0-vit5-base-smoke-42`, `s0-bartpho-attention-full-42`, `s0-vit5-base-full-42`. Các run ID này khác tên checkpoint lịch sử. Workflow tái dùng run đã finished hoặc resume từ checkpoint epoch gần nhất khi identity khớp; không chạy hai workflow cùng run ID đồng thời.

Chạy riêng smoke BARTpho:

```powershell
python -X utf8 -m modal run modal_train.py --model bartpho_attention --run-id s0-bartpho-attention-smoke-42 --smoke-steps 20 --resume
```

Chạy riêng full ViT5:

```powershell
python -X utf8 -m modal run --detach modal_train.py --model vit5 --run-id s0-vit5-base-full-42 --resume
```

`modal_train.py` giữ thông số học S0; việc tách repo không tự tối ưu thời gian train. Nếu đổi config/dữ liệu, dùng run ID mới. Không ép resume bằng cách sửa manifest.

**Inference bằng checkpoint S0 đã có**

Mặc định đọc Volume `vgec-artifacts` của profile Modal đang dùng; cache `vgec-hf-cache`. Clone repo không tải checkpoint và không cấp quyền vào Volume của tác giả.

```powershell
python -X utf8 -m modal run modal_predict.py --input examples/input.json --output outputs/example_s0.jsonl
```

Mặc định tìm các run lịch sử:

```text
vgec-artifacts/
  bartpho-attention-final-full-42/
    best_classification/
    best_correction/
  vit5-base-attention-final-full-42/
    best_correction/
```

Mỗi checkpoint cần đầy đủ model weights/config/tokenizer/`resolved_config.json`; BARTpho joint cần thêm `classifier.pt` và `attention_pool.pt`. Repo cũng đọc được checkpoint pointer dạng JSON đã có checksum. Người dùng mới cần tự train hoặc tự nhập checkpoint được phép chia sẻ vào Volume của mình.

Inference với các run vừa train trong repo này:

```powershell
python -X utf8 -m modal run modal_predict.py --input data/raw/private_test.json --output outputs/private_s0.jsonl --run-id s0-bartpho-attention-full-42 --vit5-run s0-vit5-base-full-42
```

Chọn profile bằng `--profile TEN_PROFILE` ngay sau `modal run`. Đổi Volume hoặc GPU trước khi chạy nếu cần:

```powershell
$env:VGEC_ARTIFACT_VOLUME = 'vgec-artifacts'
$env:VGEC_CACHE_VOLUME = 'vgec-hf-cache'
$env:VGEC_GPU = 'L4'
```

Chỉ ID/Input được gửi sang inference worker. Prediction lưu tại đường dẫn local đã chỉ định; wrapper tách repo không ghi đè file prediction cũ trong Volume. `--evaluate` chỉ dùng với input có nhãn và giữ gold local để tính metric.

**Export và đánh giá**

```powershell
python -X utf8 -m vgec evaluate --predictions outputs/dev_s0.jsonl --references data/prepared/dev.jsonl --output outputs/dev_metrics.json
python -X utf8 -m vgec export --predictions outputs/private_s0.jsonl --input data/raw/private_test.json --output outputs/submission_private_s0.json --draft
```

Lệnh evaluate yêu cầu predictions được tạo với namespace `dev`, ví dụ `modal_predict.py --input data/raw/dev.json --output outputs/dev_s0.jsonl --namespace dev --evaluate`. Các metric là provisional khi chưa có scorer chính thức. Export draft hiện luôn tạo `Fixed Sentence: [selected_text]`, kể cả nhãn 0; chỉ thay `--draft` bằng `--schema-verified` sau khi đã đối chiếu schema BTC.

**Tests offline**

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
python -X utf8 -m unittest discover -s tests -v
```

Tests dùng tokenizer/model nhỏ tạo tại chỗ, không tải BARTpho/ViT5 hoặc chạy GPU Modal. Kiểm tra loss/gradient, attention pooling, save/reload/resume, chọn câu S0, export, audit, import từ checkout độc lập và các đường dẫn không được mount/upload.

**Đưa lên GitHub**

Tạo repo trống rồi chạy tại folder S0:

```powershell
git init -b main
git add .
git diff --cached --stat
git diff --cached --name-only
git commit -m "Package VGEC S0 training and inference"
git remote add origin https://github.com/TEN_GITHUB/vgec-s0.git
git push -u origin main
```

Thay `TEN_GITHUB` bằng tài khoản của bạn. Kiểm tra danh sách staged trước commit: `.gitignore` loại raw/prepared data, outputs, reports, weights, token và môi trường Python. Repo chưa có license do chủ sở hữu chưa chọn; kiểm tra quyền chia sẻ dữ liệu/trọng số riêng nếu bổ sung sau này.
