Đóng gói S0 ngày 16/10/2026.

- Tách các module cần cho BARTpho attention, ViT5-base, audit, train, inference, metric và export; bỏ các entrypoint V2/Qwen/BamiBERT.
- Giữ cách tính loss, chia inner validation, attention pooling và chọn câu S0 của bản gốc. Bỏ nhánh import classifier độc lập không dùng trong S0.
- Viết lại wrapper Modal/routing để folder độc lập, không yêu cầu dữ liệu cleaned hoặc file ở thư mục cha. Chỉ mount đúng train/dev/prepared/audit đã khai báo; không mount cả thư mục chứa private test.
- Khóa revision pretrained từ manifest checkpoint gốc. Tên run cho training mới có prefix s0 để không nhầm với các run lịch sử. Cấu hình đã đổi revision nên không ép resume training cũ; inference vẫn đọc checkpoint cũ.
- Tách bộ đọc checkpoint pointer khỏi adapter trainer V2. Inference trả prediction về file local, không ghi đè prediction lịch sử trong Volume.
- Không kèm dataset, weights, credentials, output thật hoặc kết quả private. Ví dụ input là câu tự tạo.
- Không khởi tạo Git, tạo repo, push GitHub hoặc submit job GPU.

Kiểm tra đã thực hiện: 39 tests CPU/offline đạt; các test tiny-model có kiểm tra gradient, save/load và resume so với chạy liên tục. Checkout tạm độc lập import được mà không có dữ liệu hoặc đường dẫn tới project cha. Cả hai entrypoint Modal trả --help thành công. Kiểm tra cú pháp và danh sách file không có dataset/checkpoint nhạy cảm đạt.

Tests chạy bằng môi trường Python hiện có trên máy; chưa build lại image Python 3.11/requirements-modal-core.lock trên Modal và chưa chạy inference với trọng số thật trong folder mới. Muốn train cần tự đặt dữ liệu rồi chạy audit/preflight. Muốn infer bằng checkpoint cũ cần đúng Modal profile/Volume chứa model.

Theo y?u c?u gi? code c?: ?? ho?n t?c c?c thay ??i m?i trong l??t re-check v? UID, validation input/output, hashing v? split. ??i chi?u 11 module x? l? d? li?u/inference/metric/export gi?ng nguy?n b?n theo byte; l?p attention/model gi?ng theo AST; trainer ch? kh?c ???ng import b? ??c checkpoint pointer ?? folder ??c l?p. Gi? wrapper/path ??ng g?i ?? n?u ph?a tr?n. Ch?y l?i ?? 39 tests CPU/offline sau ho?n t?c: t?t c? ??t.
