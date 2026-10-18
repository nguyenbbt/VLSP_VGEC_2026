# VGEC 0.5702 — code của bản ensemble 50/50

Thư mục này gom code của cấu hình đã được người dùng báo điểm hệ thống **0.5702**. Điểm đó là kết quả nộp đã có, không phải điểm tự tính trong repo. Repo **chưa chứa trọng số model, dữ liệu, prediction private hoặc thông tin Modal**. Chỉ có code và identity của ba checkpoint để tránh vô tình chạy sai bản.

## Cấu hình chính xác

```text
Input
 ├─ BamiBERT final classifier ────────── p_bami[0..4] ─┐
 ├─ BARTpho S0 best_classification ──── p_s0[0..4] ────┤
 │       trained attention pooling + classifier head   ├─ argmax((p_bami+p_s0)/2)
 └─ BARTpho correction-only mới, 2 beam ───────────────┘
         beam hợp lệ đầu tiên → câu sửa; nếu không có → Copy Input

Nhãn 0 → Fixed Sentence: []
Nhãn 1–4 → Fixed Sentence: [câu sửa đã chọn]
```

Không dùng BARTpho S0 `best_correction` làm classifier; không dùng decoder của `best_classification` làm corrector. ViT5, Qwen và reranker không tham gia bản 0.5702. BARTpho correction-only mới **khác trọng số S0**.

| Vai trò | Checkpoint cần cung cấp |
|---|---|
| Classifier BamiBERT | `step-00002632-551882ef4edc` |
| Classifier BARTpho S0 | `bartpho-attention-final-full-42/best_classification` |
| Corrector BARTpho mới | `step-00002238-09c10cc9af21` |

Identity và SHA256 các file checkpoint/tokenizer cần thiết nằm trong [`checkpoints.json`](checkpoints.json). Hash Bami/BART mới lấy từ audit snapshot lịch sử; hash S0 lấy từ bản checkpoint local đã dùng đối chiếu. Ensemble trung bình **xác suất sau softmax** theo nhãn `[0,1,2,3,4]`, mỗi model 0.5, temperature 1. Corrector chọn output hợp lệ đầu tiên trong trace generator theo thứ tự beam, rồi mới áp dụng quy tắc nhãn 0. Việc chọn câu sửa không sử dụng gold/reference hay điểm reranker.

## Cấu trúc

- `components/s0/src/vgec/`: source S0, gồm BARTpho joint model, attention pooling và loader checkpoint.
- `components/sub1/src/vgec_sub1/`: source của nhánh BamiBERT và BARTpho correction-only; vẫn có module cho các thí nghiệm khác nhưng entrypoint 0.5702 không gọi chúng.
- `scripts/infer_components.py`: chạy **từng model một** từ checkpoint có sẵn, ghi xác suất hoặc trace BART.
- `scripts/assemble.py`: ghép theo typed ID, kiểm input hash/identity, tính ensemble và xuất JSON/ZIP.
- `tests/`: test CPU cho phép ghép, beam, fallback, schema và checkpoint head/pooling.

Hai thư mục `components` là snapshot source để giữ kiến trúc gốc. [`SOURCE_MANIFEST.json`](SOURCE_MANIFEST.json) ghi SHA256 của từng file. `scripts/verify_bundle.py` kiểm các hash này. Không sửa source snapshot rồi vẫn gọi đó là đúng run lịch sử.

## Chạy lại với checkpoint của bạn

Dùng Python 3.11 và môi trường có PyTorch/Transformers phù hợp. Lệnh dưới đây chỉ minh họa đường dẫn **bạn tự đặt**; cần cung cấp đúng ba checkpoint và Input. BamiBERT/BARTpho mới còn cần pretrained base đúng revision trong config hoặc Hugging Face cache. Script inference mặc định dùng CUDA, nên nên chạy trên A100 nếu cần tái hiện numerical behavior lịch sử; Bami dùng batch 32, S0 batch 8, BART mới batch 32.

```powershell
cd 'D:\VGEC FINETUNE\VGEC-0.5702'
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e components/s0 -e components/sub1

python scripts/infer_components.py --role bami --input data/private_test.json --checkpoint checkpoints/bami-final/step-00002632-551882ef4edc --output outputs/bami_probs.jsonl
python scripts/infer_components.py --role s0 --input data/private_test.json --checkpoint checkpoints/bartpho-attention-final-full-42/best_classification --output outputs/s0_probs.jsonl
python scripts/infer_components.py --role bart --input data/private_test.json --checkpoint checkpoints/bart-final/step-00002238-09c10cc9af21 --output outputs/bart_predictions.jsonl

python scripts/assemble.py --input data/private_test.json --bami-probs outputs/bami_probs.jsonl --s0-probs outputs/s0_probs.jsonl --bart-predictions outputs/bart_predictions.jsonl --output-dir outputs/submission
```

`outputs/submission/results.json` có đúng schema; `outputs/submission/results.zip` chỉ chứa `results.json` tại root. Script không gửi submission. Khi chạy từng role, script kiểm checkpoint name, identity, architecture, trained head/pooling và output coverage; assembler tiếp tục kiểm source hash, xác suất, label order và fallback. Nếu muốn dùng Modal, mount/tải đúng checkpoint vào các đường dẫn này trước; repo không chứa tên tài khoản hay Volume của lượt chạy lịch sử.

Bạn cũng có thể chạy riêng `scripts/assemble.py` trên xác suất và prediction nội bộ đã có, không cần load weights. Không dùng submission cũ có `Fixed Sentence: []` làm nguồn correction, vì nó làm mất text của các câu nhãn 0.

## Kiểm tra và phạm vi kết quả

```powershell
python scripts/verify_bundle.py
python -X utf8 -m unittest discover -s tests -v
```

Trên 90 câu dev lịch sử, cấu hình này đạt Macro-F1 `0.520085`, GLEU proxy `0.675969`, raw EM `45/90`. GLEU ở đây là scorer proxy nội bộ, **chưa xác minh là scorer BTC**. Trên 2.034 input private lịch sử, script ghép trong folder này đã được so với prediction baseline: không khác nhãn, câu sửa BART hoặc text sau hậu xử lý. Đây là kiểm CPU với cache đã có; **chưa chạy lại inference từ trọng số trong folder mới**. Không thể tái tạo điểm hệ thống 0.5702 chỉ bằng code khi thiếu trọng số và cùng môi trường inference.

## Đưa folder lên GitHub sau khi xem lại

`.gitignore` chặn dữ liệu, trọng số, output và credential. Trước khi push, xem `git status` và chỉ đưa code/source manifest lên. Khi bạn đã tạo repo trống trên GitHub:

```powershell
cd 'D:\VGEC FINETUNE\VGEC-0.5702'
git init
git add .
git status --short
git commit -m "Publish VGEC 0.5702 inference code"
git branch -M main
git remote add origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin main
```

Chưa có repo URL nên folder này **chưa được git init hoặc push**. Nếu định public, bạn nên chọn license phù hợp trước khi công bố.
