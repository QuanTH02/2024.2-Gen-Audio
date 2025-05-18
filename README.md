# Vietnamese Speech Recognition Project

## Mô tả bài toán
Dự án này tập trung vào việc nhận dạng giọng nói tiếng Việt (ASR - Automatic Speech Recognition) sử dụng các mô hình Whisper và PhoWhisper. Mục tiêu là chuyển đổi các file âm thanh tiếng Việt thành văn bản với độ chính xác cao.

## Cách chạy code

### 1. Code của Giao (PhoWhisper)
1. Di chuyển vào thư mục Code/Giao hoặc truy cập trực tiếp: [Google Colab](https://colab.research.google.com/drive/1GOVVEi2y8EnZKSSvgcO81mjllBucOeRQ?usp=sharing)
2. Cài đặt các thư viện cần thiết:
```bash
pip install transformers pandas gdown
```
3. Mở file notebook trong Jupyter Notebook hoặc Google Colab
4. Chạy các cell theo thứ tự:
   - Cell 1: Cài đặt dependencies và tải dữ liệu
     + Nhấn Shift + Enter hoặc click nút Run để chạy cell
     + Đợi cho đến khi cell hoàn thành việc cài đặt
   - Cell 2: Chạy thử nghiệm với 10 mẫu
     + Chạy cell và kiểm tra kết quả
   - Cell 3: Chạy toàn bộ dữ liệu test
     + Chạy cell và đợi quá trình xử lý hoàn tất

### 2. Code của Phúc (Whisper)
1. Di chuyển vào thư mục Code/Phuc hoặc truy cập trực tiếp: [Kaggle Notebook](https://huggingface.co/habuiphuc/whiper-small-finetune-fullpara)
2. Cài đặt các thư viện cần thiết:
```bash
pip install transformers datasets evaluate peft huggingface_hub
```
3. Mở file notebook trong Jupyter Notebook hoặc Google Colab
4. Chạy các cell theo thứ tự:
   - Cell 1-3: Chuẩn bị dữ liệu
     + Chạy từng cell một và đảm bảo không có lỗi
     + Kiểm tra kết quả sau mỗi cell
   - Cell 4-7: Cài đặt model và các công cụ xử lý
     + Chạy tuần tự các cell
     + Đợi cho đến khi model được tải xong
   - Cell 8-12: Huấn luyện và đánh giá model
     + Chạy các cell theo thứ tự
     + Theo dõi quá trình huấn luyện
     + Kiểm tra kết quả đánh giá

### 3. Code của Sơn
1. Di chuyển vào thư mục Code/Son hoặc truy cập trực tiếp: [Kaggle Notebook](https://www.kaggle.com/code/dinhthaisonle/phowhisper-test)
2. Mở file notebook trong Jupyter Notebook hoặc Google Colab
3. Chạy các cell theo thứ tự:
   - Cell 1: Cài đặt thư viện
     + Chạy cell và đợi cài đặt hoàn tất
   - Cell 2: Tải dữ liệu
     + Chạy cell và kiểm tra dữ liệu đã được tải
   - Cell 3: Tiền xử lý dữ liệu
     + Chạy cell và xác nhận dữ liệu đã được xử lý
   - Cell 4: Chạy thử nghiệm
     + Chạy cell và kiểm tra kết quả

### 4. Code của Quân
1. Di chuyển vào thư mục Code/Quan hoặc truy cập trực tiếp:
   - Wav2Vec2: [GitHub Repository](https://github.com/QuanTH02/2024.2-Gen-Audio)
   - PhoWhisper: [Google Colab](https://colab.research.google.com/drive/16adYRnseVyp0PWu3G8A9AckD5ufqEmQ8?authuser=0#scrollTo=MTeSfPvkoHp-)
2. Mở file notebook trong Jupyter Notebook hoặc Google Colab
3. Chạy các cell theo thứ tự:
   - Cell 1: Cài đặt thư viện
     + Chạy cell và đợi cài đặt hoàn tất
   - Cell 2: Tải model PhoWhisper
     + Chạy cell và đợi model được tải
   - Cell 3: Chạy thử nghiệm
     + Chạy cell và kiểm tra kết quả
   - Cell 4: Đánh giá kết quả
     + Chạy cell và xem báo cáo đánh giá

## Phân công công việc

### Sơn
- Thử nghiệm code
- Tiền xử lý dữ liệu private test
- Chọn mô hình phù hợp
- Làm slide thuyết trình

### Giao
- Phát triển code
- Sử dụng mô hình PhoWhisper
- Tối ưu hóa kết quả

### Phúc
- Phát triển code
- Sử dụng mô hình Whisper
- Tối ưu hóa kết quả

### Quân
- Thử nghiệm code PhoWhisper
- Thử nghiệm code Wav2Vec2
- Viết báo cáo chi tiết

## Các model sử dụng

1. **Whisper-small**
   - Model gốc từ OpenAI
   - Được fine-tune cho tiếng Việt
   - Sử dụng PEFT (Parameter-Efficient Fine-Tuning)

2. **PhoWhisper-small**
   - Model được phát triển bởi VINAI
   - Được huấn luyện đặc biệt cho tiếng Việt
   - Dựa trên kiến trúc Whisper

## Kết quả chạy

| Mô hình | WER trên Public Test | WER trên Private Test chưa tiền xử lý | WER trên Private Test đã tiền xử lý |
|---------|---------------------|----------------------|----------------------|
| Whisper-small | Chưa đánh giá | 73.34% | Chưa đánh giá |
| PhoWhisper-small | 6.78% | 40.45% | 38.8% |

*WER (Word Error Rate): Tỷ lệ lỗi từ, càng thấp càng tốt* 