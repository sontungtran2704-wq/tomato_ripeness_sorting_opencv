"""
PHẦN 1 - TIỀN XỬ LÝ, TÁCH NỀN VÀ PHÂN LOẠI ĐỘ CHÍN
Các bước xử lý:
  B1. Làm mượt ảnh màu bằng MẶT NẠ GAUSSIAN tự sinh, áp dụng riêng cho từng kênh R, G, B
  B2. Chuyển RGB -> HSI bằng công thức arccos
  B3. Tạo mặt nạ từ kênh độ bão hòa S bằng ngưỡng (tách quả khỏi nền)
  B4. Hình thái học nhị phân: mở, đóng, lấp lỗ, tìm vùng liên thông
  B5. Histogram kênh H bên trong từng quả
  B6. Đếm % điểm đỏ, cam, vàng  ->  Chín / Ương / Xanh
"""

import cv2
import numpy as np


# Nền giấy trắng có S < 0,08; vỏ cà chua có S >= 0,24. Chọn S_MIN = 55/255 (~0,22) nằm giữa 2 nhóm
# -> Mép mặt nạ trùng mép quả. Đã thử nghiệm 5 mức (38, 55, 70, 85, 100) và chọn mức cho
# Sai số đường kính nhỏ nhất 
# I_MIN loại ô vuông đen (vật chuẩn):
S_MIN = 55    # Ngưỡng độ bão hòa trên thang 0-255 (S trong [0, 1] nhân 255 cho giống ảnh 8 bit)
I_MIN = 22    # Ngưỡng cường độ: bỏ vật quá tối (ô vuông đen vật chuẩn). Thang 0-255 giống I 8 bit.
              # Thử 15 / 20 / 25 / 30: 20 - 25 chạy tốt cả khi tối 60%; 30 cắt mất phần mép quả bị tối.

def s_min_theo_ti_le(hsi, ti_le=0.10):
    """Ngưỡng S tự động: ti_le (mặc định 10 %) giá trị LỚN NHẤT của ảnh saturation (thang 0-255).
    Chương trình mặc định dùng hằng số S_MIN = 55 ; hàm này dùng để đối chiếu
    và dùng khi ảnh thật có độ bão hòa khác ảnh mô phỏng:  tach_nen(hsi, s_min_theo_ti_le(hsi))."""
    return float(ti_le * hsi[:, :, 1].max())

# Chia vòng tròn màu H thành 4 vùng
# H là GÓC (độ) đo từ trục đỏ: Đỏ 0, Vàng 60, Lục 120, Cyan 180, Lam 240, Magenta 300.
# Cà chua chín đi qua màu: XANH (70-179) -> VÀNG (50-69) -> CAM (20-49) -> ĐỎ (0-19 và 320-359).

H_DO = ((0, 20), (320, 360))
H_CAM = ((20, 50),)
H_VANG = ((50, 70),)
H_XANH = ((70, 180),)

TI_LE_CHIN = 0.60   # điểm đỏ >= 60%                  -> Chín
TI_LE_UONG = 0.50   # điểm đỏ + cam + vàng >= 50%     -> Ương,  còn lại -> Xanh

TEN_DO_CHIN = ["Xanh", "Uong", "Chin"]   # độ chín 0, 1, 2

# Phần tử cấu trúc B
# B là 'đầu dò' nhỏ (hình elip 5x5, gần giống đĩa) trượt trên ảnh để co / giãn.
# B càng lớn thì xóa nhiễu / lấp lỗ càng mạnh nhưng cũng làm mất chi tiết nhỏ của quả.
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Phép CO 
def co(A, B=KERNEL):
    """Phép CO  A ⊖ B = { z | (B)z ⊆ A }: chỗ nào B nằm LỌT hẳn trong A thì giữ lại."""
    return cv2.erode(A, B, borderType=cv2.BORDER_CONSTANT, borderValue=0)

# Phép GIÃN (dilation)
def gian(A, B=KERNEL):
    """Phép GIÃN  A ⊕ B = { z | (B̂)z ∩ A ≠ ∅ }: B phản xạ (quay 180°) rồi trượt,
    chạm A ở 1 điểm là đánh dấu. OpenCV không tự phản xạ B nên ta lật B trước."""
    B_phan_xa = cv2.flip(B, -1)
    return cv2.dilate(A, B_phan_xa, borderType=cv2.BORDER_CONSTANT, borderValue=0)

# Phép MỞ 
# Xóa đốm nhiễu nhỏ, cắt eo hẹp nhưng giữ nguyên kích thước các vật lớn (quả).
def mo(A, B=KERNEL):
    """Phép MỞ  A ∘ B = (A ⊖ B) ⊕ B: xóa vật nhỏ hơn B, cắt eo hẹp, bỏ nhánh mảnh."""
    return gian(co(A, B), B)

# Phép ĐÓNG 
# Hàn khe hẹp, xóa lỗ nhỏ trên quả.
def dong(A, B=KERNEL):
    """Phép ĐÓNG  A • B = (A ⊕ B) ⊖ B: hàn khe hẹp, xóa lỗ nhỏ."""
    return co(gian(A, B), B)

# Lấp lỗ
def lap_lo(A):
    """Điền đầy lỗ.bắt đầu từ 1 điểm trong lỗ rồi lặp X_k = (X_{k-1} ⊕ B) ∩ A^c (B là dấu cộng).
    điểm nền nào KHÔNG được tô tới thì nằm trong lỗ -> thêm vào A."""
    cao, rong = A.shape
    nen = cv2.copyMakeBorder(A, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)   # viền 0 để nền ngoài nối liền nhau
    cv2.floodFill(nen, np.zeros((cao + 4, rong + 4), np.uint8), (0, 0), 128, flags=4)
    lo = nen[1:-1, 1:-1] == 0
    ra = A.copy()
    ra[lo] = 255
    return ra

# B1 - Mặt nạ Gaussian
# Mẫu m x m ( 5 x 5, sigma = 1).
# Chia cho TỔNG hệ số để tổng = 1 -> ảnh chỉ mờ nhẹ chứ không sáng lên / tối đi.
# Gaussian lấy tâm làm trọng số lớn nhất nên khử nhiễu mà ít làm nhòe mép hơn lọc trung bình.
def tao_mat_na_gaussian(m=5, sigma=1.0):
    """Sinh mặt nạ Gaussian m x m:  h(x, y) = exp(-(x² + y²) / 2σ²),
    lấy mẫu quanh tâm rồi CHIA CHO TỔNG hệ số (tổng = 1) nên hằng số 1/(2πσ²) triệt tiêu."""
    a = (m - 1) // 2
    y, x = np.mgrid[-a:a + 1, -a:a + 1]
    h = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    return (h / h.sum()).astype(np.float32)

MAT_NA = tao_mat_na_gaussian(5, 1.0)

# Chuyển RGB -> HSI 
# I = (R+G+B)/3 | S = 1 - 3*min(R,G,B)/(R+G+B) | H = theta nếu B<=G, ngược lại 360 - theta
# Có 2 trường hợp mẫu số = 0 (điểm xám R=G=B, điểm đen R+G+B=0) -> xử lý riêng cho khỏi chia 0.
# H chỉ tính cho điểm có màu (S >= 0,03) để tiết kiệm thời gian arccos + sqrt.
def bgr_sang_hsi(anh):
    """Chuyển ảnh BGR (OpenCV) sang HSI. R, G, B chuẩn hóa về [0, 1].
        θ = arccos{ ½[(R-G) + (R-B)] / [ (R-G)² + (R-B)(G-B) ]^½ }
        H = θ nếu B ≤ G,  H = 360° - θ nếu B > G
        S = 1 - 3·min(R, G, B) / (R + G + B)
        I = (R + G + B) / 3
    Hai trường hợp mẫu số bằng 0 xử lý riêng:
        R = G = B (điểm xám thuần) -> đặt H = 0, S = 0.     R + G + B = 0 (điểm đen thuần) -> S = 0.
    Trả về mảng float32 [cao, rong, 3] gồm  H (độ, 0-360),  S (nhân 255),  I (nhân 255)."""
    b, g, r = [anh[:, :, i].astype(np.float32) / 255.0 for i in range(3)]

    # S và I tính cho MỌI điểm - chỉ có phép cộng, nhân, chia
    tong = r + g + b
    co_tong = tong > 1e-6
    S = np.where(co_tong, 1.0 - 3.0 * np.minimum(np.minimum(r, g), b) / np.where(co_tong, tong, 1.0), 0.0)
    I = tong / 3.0

    # H tốn arccos + sqrt nên chỉ tính cho điểm CÓ MÀU (S >= 0,03). Điểm nhợt màu thì H vô nghĩa
    # (mẫu số tiến về 0) và dù sao cũng bị mặt nạ S loại, nên để H = 0.
    H = np.zeros_like(S)
    chon = S >= 0.03
    rr, gg, bb = r[chon], g[chon], b[chon]
    tu = 0.5 * ((rr - gg) + (rr - bb))
    mau = np.sqrt((rr - gg) ** 2 + (rr - bb) * (gg - bb))
    co_mau = mau > 1e-6                                       # R = G = B: mẫu số 0 -> H = 0
    cos_theta = np.where(co_mau, tu / np.where(co_mau, mau, 1.0), 1.0)
    theta = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    H[chon] = np.where(co_mau, np.where(bb <= gg, theta, 360.0 - theta), 0.0)
    return np.dstack([H, S * 255.0, I * 255.0]).astype(np.float32)


# B1 + B2
def tien_xu_ly(anh):
    anh_muot = cv2.filter2D(anh, -1, MAT_NA, borderType=cv2.BORDER_REPLICATE)
    return bgr_sang_hsi(anh_muot)

# Tạo mặt nạ nhị phân
# Thứ tự: ngưỡng (S, I) -> MỞ (bỏ đốm nhiễu) -> ĐÓNG (lấp khe) -> LẤP LỖ -> MỞ lớn (bỏ cuống, lá đài).
# Mở lớn: phần tử cấu trúc cỡ 1/40 cạnh ngắn ảnh; cuống / lá nhỏ hơn B nên bị xóa,
# Thân quả to tròn thì gần như giữ nguyên.
# Đầu ra: ảnh 0 / 255, 255 (trắng) = quả cà chua.
def tach_nen(hsi, s_min=S_MIN, i_min=I_MIN):
    """Trả về ảnh nhị phân: 255 (trắng) = quả cà chua, 0 (đen) = nền."""
    S = hsi[:, :, 1]
    I = hsi[:, :, 2]

    # B3. Mặt nạ từ ảnh saturation: S lớn hơn ngưỡng -> 1, còn lại -> 0.
    # Hue không ổn định khi S thấp (mẫu số của công thức H tiến về 0) nên phải loại điểm nhợt màu TRƯỚC khi xét H.
    # Thêm ngưỡng I để bỏ vật quá tối (ô vuông đen): điểm đen/xám không có màu nên S không đáng tin.
    mask = np.where((S >= s_min) & (I >= i_min), 255, 0).astype(np.uint8)

    # Phép MỞ: xóa các đốm nhiễu nhỏ trên nền
    mask = mo(mask)
    # Phép ĐÓNG: lấp các khe hở nhỏ trên quả
    mask = dong(mask)
    # Điền đầy lỗ: vệt phản sáng trắng trên vỏ có S thấp nên bị thủng lỗ
    mask = lap_lo(mask)

    # Bỏ cuống và lá đài: chúng màu xanh đậm nên cũng lọt qua ngưỡng S.
    # Phép MỞ với phần tử cấu trúc LỚN (cỡ 1/40 cạnh ngắn ảnh): mọi chi tiết nhỏ hơn B bị loại
    # (Đây chính là tác dụng "lọc hình thái"), thân quả to tròn thì mở xong gần như giữ nguyên.
    kich_thuoc = max(5, int(min(mask.shape) / 40) | 1)          # số lẻ
    B_lon = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kich_thuoc, kich_thuoc))
    return mo(mask, B_lon)

# Tách từng quả
# Mỗi vùng trắng liên thông (8 hướng) là 1 quả. stats cho (x, y, rộng, cao, diện tích) của từng vùng.
# Nhãn 0 là nền nên các vòng lặp phía sau chạy từ 1.
def tim_vung_lien_thong(mask):
    """Trích các vùng liên thông . Trả về (số nhãn, ảnh nhãn, bảng thống kê).
    Mỗi dòng stats: x, y, rộng, cao, diện tích (nhãn 0 là nền)."""
    so_nhan, nhan, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return so_nhan, nhan, stats

# Histogram kênh H trong 1 quả
# Chỉ lấy điểm NẰM TRONG quả và có S >= S_MIN (bỏ vệt phản sáng vì H ở đó vô nghĩa).
# hist[i] = số điểm ảnh có góc H = i độ -> 360 ô, mỗi ô 1 độ.
def tinh_histogram_H(hsi_qua, mask_qua, s_min=S_MIN):
    """Histogram kênh H (360 mức, mỗi mức 1°) của các điểm bên trong 1 quả (bỏ vệt sáng trắng)."""
    H = hsi_qua[:, :, 0]
    S = hsi_qua[:, :, 1]
    # Chỉ lấy điểm nằm trong quả VÀ có màu rõ (bỏ vệt phản sáng có S thấp, vì H ở đó vô nghĩa)
    chon = (mask_qua > 0) & (S >= s_min)
    cac_gia_tri_H = H[chon].astype(np.int32) % 360
    return np.bincount(cac_gia_tri_H, minlength=360)

def _dem(hist, cac_khoang):
    return sum(int(hist[a:b].sum()) for a, b in cac_khoang)

# Quy đổi histogram thành kết luận
# Cộng các ô hist theo khoảng góc đỏ / cam / vàng / xanh rồi chia tổng = TỈ LỆ %.
# Sau đó áp luật ở phần TI_LE_CHIN. Trả về (0 Xanh | 1 Ương | 2 Chín) + 4 tỉ lệ để hiển thị.
# Nếu quả không có điểm nào hợp lệ (tong = 0) -> trả về Xanh để tránh chia 0.
def phan_loai_do_chin(hsi_qua, mask_qua, s_min=S_MIN):
    """Trả về (độ chín 0/1/2, (tỉ lệ đỏ, tỉ lệ cam, tỉ lệ vàng, tỉ lệ xanh))."""
    hist = tinh_histogram_H(hsi_qua, mask_qua, s_min)
    tong = hist.sum()
    if tong == 0:
        return 0, (0.0, 0.0, 0.0, 0.0)

    ti_le_do = _dem(hist, H_DO) / tong
    ti_le_cam = _dem(hist, H_CAM) / tong
    ti_le_vang = _dem(hist, H_VANG) / tong
    ti_le_xanh = _dem(hist, H_XANH) / tong

    if ti_le_do >= TI_LE_CHIN:
        do_chin = 2          # Chín
    elif ti_le_do + ti_le_cam + ti_le_vang >= TI_LE_UONG:
        do_chin = 1          # Ương (đang chuyển màu: cam / vàng / pha đỏ)
    else:
        do_chin = 0          # Xanh
    return do_chin, (ti_le_do, ti_le_cam, ti_le_vang, ti_le_xanh)

# Ảnh minh họa kênh H 
# Tô mỗi điểm đúng màu của góc H (S, I đặt cực đại) -> nhìn là thấy quả đỏ hay xanh.
def anh_kenh_H(hsi, mask):
    """Ảnh để XEM kênh H: tô mỗi điểm bằng đúng màu của góc hue (bão hòa và cường độ đặt cực đại), nền tô đen.
    Chỉ dùng để hiển thị, không tham gia tính toán."""
    hien = np.zeros(hsi.shape, np.uint8)
    hien[:, :, 0] = np.clip(hsi[:, :, 0] / 2.0, 0, 179)       # OpenCV chỉ cần H/2 để vẽ màu
    hien[:, :, 1] = 255
    hien[:, :, 2] = 255
    anh = cv2.cvtColor(hien, cv2.COLOR_HSV2BGR)
    anh[mask == 0] = 0
    return anh

# Ghép 4 ảnh
def _them_tieu_de(anh, chu):
    anh = cv2.resize(anh, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.rectangle(anh, (0, 0), (anh.shape[1], 24), (40, 30, 20), -1)
    cv2.putText(anh, chu, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return anh

def ve_4_buoc_phan1(anh):
    """Ghép 4 ảnh: gốc | mặt nạ | kênh H | kết quả. Trả về (ảnh ghép, bảng tỉ lệ từng quả)."""
    hsi = tien_xu_ly(anh)
    mask = tach_nen(hsi)
    anh_H = anh_kenh_H(hsi, mask)

    ket_qua = anh.copy()
    mau_ve = [(60, 190, 40), (20, 150, 250), (60, 60, 235)]   # BGR: xanh, cam, đỏ
    bang = []
    so_nhan, nhan, stats = tim_vung_lien_thong(mask)
    for l in range(1, so_nhan):
        x, y, w, h, dien_tich = stats[l]
        if dien_tich < 600:                   # bỏ vùng quá nhỏ (nhiễu)
            continue
        mask_qua = np.where(nhan[y:y + h, x:x + w] == l, 255, 0).astype(np.uint8)
        do_chin, (do, cam, vang, xanh) = phan_loai_do_chin(hsi[y:y + h, x:x + w], mask_qua)
        bang.append((x, y, TEN_DO_CHIN[do_chin], do, cam, vang, xanh))
        vien, _ = cv2.findContours(mask_qua, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(int(x), int(y)))
        cv2.drawContours(ket_qua, vien, -1, mau_ve[do_chin], 3)
        cv2.putText(ket_qua, f"{TEN_DO_CHIN[do_chin]} do{do:.0%} cam{cam:.0%} vang{vang:.0%}", (x, max(y - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mau_ve[do_chin], 2, cv2.LINE_AA)

    o1 = _them_tieu_de(anh, "1. Anh goc")
    o2 = _them_tieu_de(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), "2. Mat na: S>=S_MIN, I>=I_MIN + mo/dong/lap lo")
    o3 = _them_tieu_de(anh_H, "3. Kenh H (HSI) ben trong qua")
    o4 = _them_tieu_de(ket_qua, "4. Ti le do / cam / vang -> Chin / Uong / Xanh")
    ghep = np.vstack([np.hstack([o1, o2]), np.hstack([o3, o4])])
    return ghep, bang


if __name__ == "__main__":
    import sys
    from part_3_simulation import tao_anh_tinh
    if len(sys.argv) > 1:
        anh = cv2.imdecode(np.fromfile(sys.argv[1], np.uint8), cv2.IMREAD_COLOR)   # đọc được đường dẫn có dấu
    else:
        anh, _, _ = tao_anh_tinh(so_qua=9, seed=7)

    ghep, bang = ve_4_buoc_phan1(anh)
    print(f"{'Qua tai (x, y)':<16}{'Ket qua':<9}{'Do':>7}{'Cam':>7}{'Vang':>7}{'Xanh':>7}")
    for x, y, ten, do, cam, vang, xanh in sorted(bang):
        print(f"({x:4d},{y:4d})     {ten:<9}{do:>7.0%}{cam:>7.0%}{vang:>7.0%}{xanh:>7.0%}")
    cv2.imshow("PHAN 1 - Tach nen & phan loai do chin (nhan phim bat ky de thoat)", ghep)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
