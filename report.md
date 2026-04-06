# 🎯 SYSTEM PROMPT CHO AI CODING AGENT: TỐI ƯU HÓA COVERT TIMING CHANNEL (TIMEGAN + RL)

**[VAI TRÒ VÀ BỐI CẢNH]**
Bạn là một AI Research Engineer và Red Teamer. Nhiệm vụ của bạn là sửa đổi mã nguồn Jupyter Notebook (dựa trên kiến trúc TimeGAN) để biến nó thành một **Bộ Tiêm Nhiễu Thời Gian (Additive Noise Injection)** cho kênh Covert Timing Channel.
Mục tiêu: Ngụy trang các mốc thời gian cứng (Bit 0 = 0.5s, Bit 1 = 1.0s) thành dải phân phối $\mathcal{N}(0.8, 0.15)$ của lưu lượng IoT, không làm lật bit khi giải mã (Bit 0 phải $\le 0.75s$).

---

## 🛑 PHẦN GIỮ NGUYÊN (DO NOT MODIFY)
Tuyệt đối **KHÔNG SỬA ĐỔI** cấu trúc mạng và luồng huấn luyện của 2 phase đầu tiên:
1. **Phase 1 (Autoencoder Training):** Mạng Embedder và Recovery nén/giải nén không gian ẩn (Latent Space) đang hoạt động rất tốt.
2. **Phase 2 (Supervisor Training):** Khả năng học liên kết thời gian (Temporal Dynamics) của Supervisor đã chuẩn. 

---

## 🛠 CÁC MODULE CẦN VIẾT LẠI (IMPLEMENTATION TASKS)

### BƯỚC 1: Xây dựng DataLoader Giả lập & Chuẩn hóa Cố định (Fixed Scaler)
* **Xóa bỏ:** Không đọc dữ liệu từ file CSV (`custom_traffic_data.csv`).
* **Sinh dữ liệu động:** Tạo hàm sinh batch dữ liệu từ phân phối chuẩn: `data = torch.normal(mean=0.8, std=0.15, size=(batch_size, seq_len, 1))`.
* **Ép cứng hệ quy chiếu (Fixed MinMax Scaler):** KHÔNG dùng max/min của batch. Ép không gian vật lý tĩnh để tránh nghẽn `Sigmoid` ở mạng Recovery.
  * Khai báo hằng số toàn cục: `PHYS_MIN = 0.0`, `PHYS_MAX = 2.5`.
  * `X_scaled = (X - PHYS_MIN) / (PHYS_MAX - PHYS_MIN)`.

### BƯỚC 2: Định hình Đầu vào & Cơ chế Tiêm Nhiễu (Additive Noise)
Thay đổi logic của hàm `forward` trong mạng `Generator`:
* **Condition Vật lý ($C$):** $C$ đầu vào sẽ chứa xen kẽ các mốc thời gian vật lý của thông điệp (0.5s cho Bit 0, 1.0s cho Bit 1).
* **Scale Condition:** Quy đổi $C \rightarrow C_{scaled}$ qua hàm Scaler ở Bước 1. (Ví dụ: 0.5s $\rightarrow$ 0.2).
* **Kiến trúc Đầu vào:** Tại lớp đầu tiên, `torch.cat([Z, C_scaled], dim=-1)` với $Z$ là nhiễu ngẫu nhiên.
* **Tiêm Nhiễu Cộng gộp:** * Generator (AI) KHÔNG xuất ra IPD cuối cùng. Nó chỉ xuất ra lượng nhiễu: `delta_t = self.network(input)`.
  * **Công thức output cuối cùng:** `final_ipd_scaled = C_scaled + delta_t`.



### BƯỚC 3: Xây dựng Hàm Loss Kép (Phase 3 - Joint Training)
Trong vòng lặp huấn luyện `Joint Training` (Phase 3), thiết kế lại `Total_G_Loss` bằng tổng của 4 thành phần:
1. **GAN Loss (`gan_loss`):** Loss cơ bản lừa Discriminator (giữ nguyên logic gốc).
2. **Conditioning Loss (`mse_loss`):** Ép lượng nhiễu bám sát mốc vật lý.
   * `mse_loss = nn.MSELoss()(final_ipd_scaled, C_scaled)`.
3. **RL Penalty Loss (Bộ lọc Vật lý):**
   * Ngưỡng vật lý an toàn cho Bit 0 là `0.75s`. Đổi sang scale: `THRESHOLD_SCALED = 0.3`.
   * Tìm index của các mốc Bit 0: `mask_bit0 = (C_scaled == 0.2)`.
   * Nếu tại `mask_bit0` có `final_ipd_scaled > THRESHOLD_SCALED` $\rightarrow$ Áp dụng hình phạt: `penalty_loss += 100.0 * số_lượng_vi_phạm`.
4. **Entropy Regularization Loss (`entropy_loss`):** * Tối đa hóa phương sai để chống Mode Collapse (Mô phỏng Soft Actor-Critic).
   * `variance = torch.var(final_ipd_scaled)`
   * `entropy_loss = -1.0 * variance * alpha` (Khởi tạo `alpha = 0.1`).
* **Tổng hợp:** `Total_G_Loss = gan_loss + mse_loss + penalty_loss + entropy_loss`.

### BƯỚC 4: Vòng lặp Validation & Early Stopping
Ở cuối MỖI Epoch của Phase 3, tạm ngắt quá trình train để chạy Validation:
1. **Mô phỏng Inference:** Sinh 1000 mẫu $Z$ ngẫu nhiên và vector $C$ xen kẽ (Bit 0/Bit 1). Đưa qua Generator để lấy `final_ipd_scaled`.
2. **Inverse Scale:** Dịch ngược `final_ipd_scaled` về hệ quy chiếu vật lý (tính bằng giây) $\rightarrow$ `final_ipd_phys`.
3. **Kiểm tra Reliability (Độ tin cậy):** Đếm xem có bao nhiêu Bit 0 vi phạm (giá trị > `0.75s`).
4. **Kiểm tra Stealth (Tàng hình):** Chạy kiểm định thống kê: `p_value = scipy.stats.kstest(final_ipd_phys, 'norm', args=(0.8, 0.15)).pvalue`.
5. **Điều kiện dừng sớm (Early Stopping):** * NẾU (`Số vi phạm == 0`) VÀ (`p_value > 0.05`):
   * Lưu lại file mô hình: `torch.save(generator.state_dict(), 'generator_optimal.pth')`.
   * Lập tức thoát khỏi vòng lặp huấn luyện (`break`). Không tiếp tục train để tránh Mode Collapse.

**[HÀNH ĐỘNG YÊU CẦU CHO AGENT]**
Hãy đọc kỹ file `notebook951a76ef77.ipynb`, sau đó viết lại mã nguồn Python hoàn chỉnh cho các Cell tương ứng với Data Preparation và Phase 3 (Joint Training & Validation) dựa trên bản thiết kế này. Code cần có comment rõ ràng bằng tiếng Anh hoặc tiếng Việt ở các phần tính Loss và Tiêm nhiễu.