# Kiểm tra gói code 0.5702

- 50 file source/config/prompt trong `components/` đã được copy giữ nguyên byte từ hai folder S0 và Sub1; SHA256 nằm trong `SOURCE_MANIFEST.json`.
- Cấu hình Sub1 copy ra cho cùng hash `270f544a79c4406ca18e857de3c1b09b694c86f694d5a5abf866bb19bcc9d840` với bản đã train.
- Script ghép mới được chạy trên cache lịch sử theo typed ID: **90/90 dev** và **2.034/2.034 private** khớp nhãn ensemble, câu sửa BART và text sau hậu xử lý với prediction của baseline 0.5702.
- Selector BART trong script inference được replay trên 90 raw trace dev và 2.034 raw trace final-model private: **0 câu khác text, 0 câu khác trạng thái fallback** so với bản BART top-1 lịch sử.
- Checkpoint loader nay kiểm SHA256 của weights, tokenizer, S0 head và attention pooling theo snapshot lịch sử; phép thử tráo weights phải bị từ chối.
- `python scripts/verify_bundle.py`: đạt. `python -X utf8 -m unittest discover -s tests -v`: 7/7 test đạt. Hai CLI đều trả `--help` thành công.

Các phép kiểm trên chỉ chạy CPU với artifact đã lưu. Chưa load ba checkpoint và chưa chạy lại inference end-to-end trong folder mới. Không có dataset, weights, prediction private hoặc credential trong gói code.
