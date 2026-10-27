# Full-data refit của hệ thống 0.5702

Đây là **run mới**, train lại từ pretrained ba thành phần của cấu hình có điểm hệ thống được người dùng báo là 0.5702: classifier BamiBERT, classifier BARTpho S0 joint với attention pooling + head, và corrector BARTpho correction-only. Sau train, công thức ensemble vẫn là `0.5 * p_bami + 0.5 * p_s0` theo nhãn `[0,1,2,3,4]`; BARTpho sửa câu vẫn lấy beam hợp lệ đầu tiên, fallback Copy; nhãn 0 export `[]`. Không dùng R-Drop, Qwen, ViT5 hoặc reranker.

**Dữ liệu:** `the_final_data (1).json` có 7.153 nguồn duy nhất và **đã bao gồm 90 Input trong `dev.json`**. `data/prepared_final/train.jsonl` là 7.063 nguồn đã loại 90 dev. Do ID của cả 90 câu khác nhau giữa file final và dev, phép ghép theo Input chuẩn NFC/khoảng trắng, kiểm raw Input/label/reference, rồi dùng mỗi nguồn final một lần. Không nối dev vào final lần nữa. Bami/S0 dùng 7.153 nguồn; BART corrector dùng 7.097 nguồn có target sửa hợp lệ. `audit_data.py` và `prepare_full_data.py` tạo audit, ba file JSONL và manifest hash, không sửa file dữ liệu cũ.

**Chọn checkpoint:** Dev trở thành dữ liệu optimizer, nên không đánh giá/chọn checkpoint bằng dev. Runner khóa số epoch từ các checkpoint đã được chọn ở run trước: Bami **14/15**, S0 classifier **10/12**, BART corrector **6/8** (số trước là epoch dừng; số sau là horizon scheduler cũ). Các mốc này là quyết định có trước refit; model mới chưa được chứng minh tốt hơn bản 0.5702. Không còn validation chưa thấy nào trong lần refit này.

Bằng chứng mốc epoch: `experiments/bami_rdrop_alpha05/baseline/best_progress.json` (Bami cũ), `modal_backup_20261015/artifacts/bartpho-attention-final-full-42/inner_metrics_epoch_*.json` (S0), `reports/vgec_diagnostic_20261017/checkpoint_audit.json` (BART). Tokenizer S0 đã được kiểm CPU trên 7.153 nguồn: source dài nhất 64 token, target dài nhất 60 token, dưới giới hạn 256; runner tiếp tục kiểm toàn bộ token trước bước optimizer.

Checkpoint mới nằm ở Modal Volume `vgec-05702-refit-artifacts/vgec-full-data-refit-05702/<run-id>`; model/cache cũ không bị ghi đè. Bami/BART lưu snapshot bất biến và pointer `best`; S0 lưu `best_classification` gồm encoder, `classifier.pt` và `attention_pool.pt`. Mỗi run có identity dữ liệu/config, trạng thái optimizer để resume khi dùng **cùng run-id**. Runner từ chối prepared data đổi hash, sai số nguồn, run-id không an toàn và resume sai identity. Smoke và full dùng run-id khác nhau. Sau khi chạy, phải đối chiếu manifest/checkpoint mới rồi mới infer; script inference trong `VGEC-0.5702` đang khóa identity checkpoint cũ, không được lấy checkpoint refit trá hình làm checkpoint 0.5702 lịch sử.

Profile `tuandatdao16052007` dùng hai Volume riêng: `vgec-05702-refit-artifacts` cho checkpoint và `vgec-05702-refit-hf-cache` cho pretrained cache. Runner tạo chúng nếu chưa có trên account mới. Lần chạy đầu cần tải pretrained model nên có thể chậm hơn.

Chạy trên PowerShell từ root workspace:

```powershell
cd 'D:\VGEC FINETUNE'
python -X utf8 experiments/full_data_refit_05702/audit_data.py
python -X utf8 experiments/full_data_refit_05702/prepare_full_data.py
python -X utf8 -m unittest discover -s experiments/full_data_refit_05702 -p 'test_*.py' -v
$env:PYTHONIOENCODING = 'utf-8'
```

Lệnh Modal bên dưới **chưa chạy full train**. Chạy smoke lần lượt trước; chỉ chạy full run của từng role khi smoke của role đó trả `checkpoint_reload: passed`:

```powershell
modal run --profile tuandatdao16052007 experiments/full_data_refit_05702/modal_refit.py --role bami --run-id smoke-bami-05702 --smoke-steps 2
modal run --profile tuandatdao16052007 experiments/full_data_refit_05702/modal_refit.py --role s0 --run-id smoke-s0-05702 --smoke-steps 2
modal run --profile tuandatdao16052007 experiments/full_data_refit_05702/modal_refit.py --role bart --run-id smoke-bart-05702 --smoke-steps 2

modal run --profile tuandatdao16052007 experiments/full_data_refit_05702/modal_refit.py --role bami --run-id bami-full-05702
modal run --profile tuandatdao16052007 experiments/full_data_refit_05702/modal_refit.py --role s0 --run-id s0-full-05702
modal run --profile tuandatdao16052007 experiments/full_data_refit_05702/modal_refit.py --role bart --run-id bart-full-05702
```

Nếu Modal mất kết nối giữa chừng, chạy lại **chính lệnh full với cùng run-id** để resume từ checkpoint cuối. Không chạy nhiều role đồng thời với cùng run-id. Ba lệnh full bắt đầu train từ pretrained, không resume checkpoint 0.5702.

Không dùng dev cũ để báo điểm sau refit: cả 90 câu đã được học. Điểm hệ thống 0.5702 là của các checkpoint cũ, chưa phải điểm của checkpoint mới. Bài toán nguồn synthetic/lineage vẫn giữ giới hạn đánh giá đã ghi ở các audit trước.

Infer private sau khi cả ba full train hoàn tất:

```powershell
cd 'D:\VGEC FINETUNE'
$env:PYTHONIOENCODING = 'utf-8'
modal run --profile tuandatdao16052007 experiments/full_data_refit_05702/modal_private.py
```

Runner kiểm tra CPU trước khi cấp GPU: đúng ba run `bami-full-05702`, `s0-full-05702`, `bart-full-05702`, identity dữ liệu/config/code, epoch/optimizer steps đủ, checkpoint/hash weights/tokenizer và trained pooling/head. Smoke hoặc train chưa đủ sẽ bị từ chối. Worker chỉ nhận ID/Input từ `private_test.json`, lần lượt infer Bami → S0 → BART trên một A100; không train hay đọc gold. Giữ softmax/ensemble 50/50, decoding/validator/selector BART và nhãn 0 → `[]` như baseline. Cache hoàn tất được dùng lại khi chạy lại cùng lệnh; checkpoint/input/code khác phải dùng `--output-id` mới.

Kết quả tải tự động về `experiments/full_data_refit_05702/outputs/ensemble50-full-private-v1/`: ba cache xác suất/correction, `predictions.jsonl`, `results.json`, `ensemble50_full_refit_bart_top1.zip`, `identity.json`, `manifest.json`. ZIP chỉ chứa `results.json` ở root. Export kiểm typed ID, Input hash, coverage, thứ tự, candidate đầy đủ và hash JSON/ZIP. Đây là bản refit mới, không gán điểm 0.5702 cho nó; script không tự submit.
