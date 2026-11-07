"""
PHẦN 2 - ĐO KÍCH THƯỚC VÀ ĐẾM SỐ LƯỢNG
Ý tưởng:
  - Máy tính chỉ biết kích thước theo PIXEL. Muốn đổi ra mm cần 1 vật biết trước
    kích thước: ô vuông đen 20 x 20 mm. Nếu cạnh ô vuông = 50 pixel thì
    k = 50 / 20 = 2.5 pixel/mm  ->  quả rộng 150 pixel = 150 / 2.5 = 60 mm.
  - Mỗi quả trong mặt nạ (Phần 1) là 1 vùng trắng: đếm số pixel = diện tích A.
    Coi quả là hình tròn:  A = pi * (D/2)^2   =>   D = 2 * sqrt(A / pi)

Các bước:
  B1. Tìm ô vuông đen 20 x 20 mm -> hệ số k (pixel/mm)
  B2. Lấy từng vùng trắng trong mặt nạ; nếu 2 quả dính nhau thì tách ra
  B3. Mỗi quả: diện tích A (đếm pixel), trọng tâm (trung bình tọa độ), đường kính D
  B4. D_mm = D_px / k  ->  cỡ S (< 50 mm), M (50 - 65 mm), L (> 65 mm)
  B5. Đếm: ảnh tĩnh -> bảng 3 x 3 (độ chín x cỡ)
           băng tải -> theo dõi từng quả, chỉ đếm 1 lần khi tâm quả đi qua vạch
"""
import math

import cv2
import numpy as np

from part_1_classification import tien_xu_ly, tach_nen, tim_vung_lien_thong, phan_loai_do_chin, S_MIN, I_MIN


# Tham số
# Vật chuẩn 20 x 20 mm | cỡ: S < 50, M 50-65, L > 65 mm
# Vùng < 600 pixel = nhiễu
CANH_VAT_CHUAN_MM = 20.0     # cạnh ô vuông đen (mm)
CO_S = 50.0                  # đường kính < 50 mm  -> cỡ S
CO_L = 65.0                  # đường kính > 65 mm  -> cỡ L, còn lại là M
TEN_CO = ["S", "M", "L"]     # cột 0, 1, 2 của bảng đếm
DIEN_TICH_MIN = 600          # vùng nhỏ hơn 600 pixel coi là nhiễu, bỏ qua


# B1 - Tìm vật chuẩn -> hệ số k
# Ngưỡng = 0,5 x trung vị I -> vùng tối
# Lọc: diện tích hợp lý | gần vuông (tỉ lệ cạnh <= 1,15) | đặc (>= 85%)
# MinAreaRect: vật chuẩn đặt nghiêng vẫn nhận được
# k = căn(diện tích) / 20 mm  (pixel/mm)
def tim_vat_chuan(hsi):
    """Tìm ô vuông đen. Trả về (k pixel/mm, (x, y, w, h)) hoặc (None, None) nếu không thấy."""
    # Kênh I của HSI chính là ảnh mức xám theo phương pháp trung bình: I = (R + G + B) / 3.
    # Tìm vật đen = chuyển ảnh xám sang ảnh nhị phân bằng 1 ngưỡng.
    V = hsi[:, :, 2]
    # Ngưỡng = 1/2 độ sáng nền (trung vị I của cả ảnh vì nền chiếm nhiều nhất).
    # Ảnh tối đi thì ngưỡng cũng tự giảm theo -> không phải chỉnh tay.
    nguong = 0.5 * np.median(V)
    vung_toi = np.where(V < nguong, 255, 0).astype(np.uint8)

    vien, _ = cv2.findContours(vung_toi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    tot_nhat = None
    for c in vien:
        x, y, w, h = cv2.boundingRect(c)
        if w < 10 or h < 10:
            continue
        # Diện tích = số pixel tối nằm bên trong đường viền
        to_kin = np.zeros((h, w), np.uint8)
        cv2.drawContours(to_kin, [c], -1, 255, -1, offset=(-x, -y))
        dien_tich = np.count_nonzero((to_kin > 0) & (vung_toi[y:y + h, x:x + w] > 0))
        if dien_tich < 150 or dien_tich > 0.05 * V.size:
            continue
        # Hình chữ nhật (có thể xoay) nhỏ nhất bao vùng tối -> vật chuẩn đặt nghiêng vẫn nhận được
        _, (rw, rh), _ = cv2.minAreaRect(c)
        if min(rw, rh) < 5:
            continue
        gan_vuong = max(rw, rh) / min(rw, rh) <= 1.15                 # 2 cạnh gần bằng nhau
        dac = dien_tich / ((rw + 1) * (rh + 1)) >= 0.85                # bên trong tô kín, không rỗng
        if gan_vuong and dac and (tot_nhat is None or dien_tich > tot_nhat[0]):
            tot_nhat = (dien_tich, (x, y, w, h))

    if tot_nhat is None:
        return None, None
    dien_tich, khung = tot_nhat
    canh_pixel = math.sqrt(dien_tich)              # hình vuông: cạnh = căn bậc 2 diện tích
    k = canh_pixel / CANH_VAT_CHUAN_MM
    return k, khung


# B2 - Tách quả dính nhau
# Độ đặc = diện tích / diện tích bao lồi; > 0,95 -> 1 quả
# Biến đổi khoảng cách (distanceTransform): tâm quả = đỉnh, giá trị đỉnh = bán kính
# Tìm đỉnh: điểm không đổi sau phép GIÃN
# Bỏ đỉnh trùng: cách nhau < 0,7 (r1 + r2) -> cùng 1 quả
# Chia điểm: gán cho quả có (khoảng cách tới tâm - bán kính) nhỏ nhất
def tach_qua_cham_nhau(vung):
    """vung: ảnh nhị phân của 1 vùng trắng. Trả về danh sách ảnh nhị phân, mỗi ảnh = 1 quả."""
    vien, _ = cv2.findContours(vung, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(vien, key=cv2.contourArea)

    # Độ đặc = diện tích / diện tích bao lồi. 1 quả tròn: ~ 0.99.
    # 2 quả dính nhau tạo chỗ "eo" lõm vào nên độ đặc nhỏ hơn (~ 0.85 - 0.9).
    do_dac = cv2.contourArea(c) / max(cv2.contourArea(cv2.convexHull(c)), 1)
    if do_dac > 0.95:
        return [vung]                                  # chỉ có 1 quả

    # Biến đổi khoảng cách (distance transform): mỗi điểm trắng = khoảng cách tới điểm đen gần nhất.
    # Đây là hàm có sẵn của OpenCV (cv2.distanceTransform), không tự cài đặt.
    # -> Tâm mỗi quả là chỗ xa mép nhất = ĐỈNH, giá trị đỉnh = bán kính quả.
    dist = cv2.distanceTransform(vung, cv2.DIST_L2, 5)

    # Tìm đỉnh (cực đại cục bộ): sau phép GIÃN ảnh xám, mỗi điểm nhận giá trị lớn nhất
    # trong lân cận. Điểm nào KHÔNG đổi sau khi giãn chính là điểm lớn nhất vùng đó.
    # Cửa sổ nhỏ (0.3 bán kính quả lớn nhất) để quả nhỏ nằm sát quả to vẫn có đỉnh riêng.
    # Các đỉnh thừa trong cùng 1 quả sẽ bị gộp ở bước "bỏ đỉnh trùng" ngay dưới.
    kich_thuoc = max(5, int(0.3 * dist.max()) | 1)                  # số lẻ
    # Phần tử VUÔNG thay vì tròn: OpenCV giãn với hình vuông nhanh hơn nhiều
    kernel = np.ones((kich_thuoc, kich_thuoc), np.uint8)
    dist_gian = cv2.dilate(dist, kernel)
    dinh = np.where((dist >= dist_gian) & (dist > 0.4 * dist.max()), 255, 0).astype(np.uint8)

    # Mỗi cụm điểm đỉnh -> 1 ứng viên tâm (cx, cy) và bán kính r
    vien_dinh, _ = cv2.findContours(dinh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ung_vien = []
    for d in vien_dinh:
        xs, ys = d[:, 0, 0], d[:, 0, 1]
        ung_vien.append((xs.mean(), ys.mean(), dist[ys, xs].max()))

    # 2 quả chạm nhau thì 2 tâm cách nhau khoảng r1 + r2.
    # Đỉnh nào ở quá gần 1 đỉnh lớn hơn (< 0.7 (r1 + r2)) thì cùng 1 quả (quả hơi bầu dục) -> bỏ.
    ung_vien.sort(key=lambda t: -t[2])                              # xét đỉnh lớn trước
    tam = []
    for cx, cy, r in ung_vien:
        trung = False
        for cx2, cy2, r2 in tam:
            if math.sqrt((cx - cx2) ** 2 + (cy - cy2) ** 2) < 0.7 * (r + r2):
                trung = True
        if not trung:
            tam.append((cx, cy, r))
    if len(tam) < 2:
        return [vung]

    # Gán mỗi điểm cho quả mà nó nằm "sâu" nhất: khoảng cách Euclid tới tâm (tính trong mặt phẳng ảnh) trừ bán kính.
    # Với 2 hình tròn chạm nhau, đường chia đi đúng qua điểm tiếp xúc.
    Y, X = np.indices(vung.shape)
    bang_khoang_cach = []
    for cx, cy, r in tam:
        bang_khoang_cach.append(np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - r)
    thuoc_qua = np.argmin(np.array(bang_khoang_cach), axis=0)       # số thứ tự quả gần nhất

    cac_qua = []
    for i in range(len(tam)):
        cac_qua.append(np.where((thuoc_qua == i) & (vung > 0), 255, 0).astype(np.uint8))
    return cac_qua


# B4 - Chia cỡ
# D < 50 mm: S | D > 65 mm: L | còn lại: M | không có vật chuẩn: ?
def phan_loai_co(d_mm):
    if d_mm is None:
        return "?"                       # không có vật chuẩn -> không biết mm
    if d_mm < CO_S:
        return "S"
    if d_mm > CO_L:
        return "L"
    return "M"


# B3 - Phân tích 1 ảnh (Phần 1 + Phần 2)
# tien_xu_ly -> tach_nen -> tim_vat_chuan -> tách từng vùng -> tách quả dính
# Diện tích A = số pixel | tâm = trung bình tọa độ
# D_px = 2 x căn(A / pi) | D_mm = D_px / k
# Độ chín: phan_loai_do_chin (Phần 1)
# cham_bien: quả bị cắt ở mép ảnh | kết quả sắp từ trái sang phải
def phan_tich_anh(anh, s_min=S_MIN, i_min=I_MIN):
    """Chạy toàn bộ Phần 1 + Phần 2 trên 1 ảnh.
    Trả về dict gồm:
       qua      : danh sách quả, mỗi quả là 1 dict (tâm, đường kính, cỡ, độ chín, ...)
       mask     : mặt nạ nhị phân
       hsi      : ảnh HSI (H độ, S và I thang 0-255)
       k        : hệ số pixel/mm (None nếu không thấy vật chuẩn)
       vat_chuan: khung (x, y, w, h) của vật chuẩn"""
    hsi = tien_xu_ly(anh)                       # Phần 1 - B1, B2
    mask = tach_nen(hsi, s_min, i_min)          # Phần 1 - B3, B4
    k, vat_chuan = tim_vat_chuan(hsi)           # Phần 2 - B1
    cao, rong = mask.shape

    ds_qua = []
    so_nhan, nhan, stats = tim_vung_lien_thong(mask)            # trích vùng liên thông
    for l in range(1, so_nhan):
        x, y, w, h, dien_tich = stats[l]
        if dien_tich < DIEN_TICH_MIN:
            continue
        vung = np.where(nhan[y:y + h, x:x + w] == l, 255, 0).astype(np.uint8)   # cắt riêng vùng này ra

        for mask_qua in tach_qua_cham_nhau(vung):              # Phần 2 - B2
            ys, xs = np.nonzero(mask_qua)                        # tọa độ các điểm trắng của quả
            A = len(xs)                                          # diện tích = số pixel
            if A < DIEN_TICH_MIN:
                continue
            # Trọng tâm = trung bình tọa độ (chính là mô-men m10/m00, m01/m00)
            cx = x + xs.mean()
            cy = y + ys.mean()
            d_px = 2 * math.sqrt(A / math.pi)                    # đường kính tương đương (pixel)
            d_mm = d_px / k if k else None                       # đổi ra mm
            do_chin, ti_le = phan_loai_do_chin(hsi[y:y + h, x:x + w], mask_qua, s_min)   # Phần 1 - B5, B6

            x0, y0 = x + xs.min(), y + ys.min()
            x1, y1 = x + xs.max(), y + ys.max()
            vien_qua, _ = cv2.findContours(mask_qua, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                                           offset=(int(x), int(y)))
            ds_qua.append({
                "tam": (cx, cy),
                "dien_tich": A,
                "d_px": d_px,
                "d_mm": d_mm,
                "co": phan_loai_co(d_mm),
                "do_chin": do_chin,
                "ti_le": ti_le,
                "khung": (x0, y0, x1 - x0 + 1, y1 - y0 + 1),
                "vien": max(vien_qua, key=cv2.contourArea),      # để vẽ đường viền
                "cham_bien": x0 <= 1 or y0 <= 1 or x1 >= rong - 2 or y1 >= cao - 2,  # bị cắt ở mép ảnh
            })

    ds_qua.sort(key=lambda q: q["tam"][0])                       # sắp từ trái sang phải
    return {"qua": ds_qua, "mask": mask, "hsi": hsi, "k": k, "vat_chuan": vat_chuan}


# Tính lại mm với k khác
# Camera: k = trung vị nhiều khung cho ổn định
def cap_nhat_kich_thuoc(kq, k):
    """Tính lại mm với hệ số k khác (camera: k lấy trung vị nhiều khung cho ổn định)."""
    kq["k"] = k
    for q in kq["qua"]:
        q["d_mm"] = q["d_px"] / k if k else None
        q["co"] = phan_loai_co(q["d_mm"])


# B5 - Đếm ảnh tĩnh
# Bảng 3 x 3: hàng = độ chín (Xanh, Ương, Chín) | cột = cỡ (S, M, L)
def dem_anh_tinh(ds_qua):
    """Bảng đếm 3 x 3: hàng = độ chín (0 Xanh, 1 Ương, 2 Chín), cột = cỡ (S, M, L)."""
    bang = np.zeros((3, 3), int)
    for q in ds_qua:
        if q["co"] in TEN_CO:
            bang[q["do_chin"], TEN_CO.index(q["co"])] += 1
    return bang


# B5 - Bộ đếm băng tải
# Mỗi quả 1 ID: tâm, vận tốc, số khung mất dấu, phiếu độ chín, đường kính, cờ da_dem
def tao_bo_dem():
    """Bộ đếm cho băng tải. 'vet' lưu thông tin từng quả đang được theo dõi (theo ID)."""
    return {"id_tiep_theo": 1, "vet": {}, "bang": np.zeros((3, 3), int)}


# B5 - Theo dõi và đếm qua vạch (mỗi khung hình gọi 1 lần)
# 1. Dự đoán vị trí = vị trí cũ + vận tốc -> ghép với ID gần nhất (cặp gần nhất ghép trước)
# 2. Không ghép được -> quả mới -> cấp ID mới
# 3. Mất dấu quá 6 khung -> xóa ID
# 4. Qua vạch: (x_cũ - vạch) x (x_mới - vạch) <= 0 -> đếm 1 lần
# Độ chín = đa số phiếu | đường kính = trung vị các khung
def cap_nhat_bo_dem(bo_dem, ds_qua, x_vach, k):
    """Gọi 1 lần cho mỗi khung hình. Trả về (danh sách ID của từng quả, danh sách ID vừa được đếm).

    Ý tưởng: quả ở khung này sẽ gần chỗ nó ở khung trước + quãng đường vừa đi (vận tốc).
    -> Ghép mỗi quả với ID cũ gần vị trí dự đoán nhất. Khi tâm đi qua vạch đếm thì đếm 1 lần."""
    vet = bo_dem["vet"]

    # 1. Tính khoảng cách giữa vị trí DỰ ĐOÁN của mỗi ID cũ và tâm mỗi quả mới
    cac_cap = []
    for id_qua, v in vet.items():
        du_doan_x = v["tam"][0] + v["van_toc"][0]
        du_doan_y = v["tam"][1] + v["van_toc"][1]
        for j, q in enumerate(ds_qua):
            kc = math.sqrt((du_doan_x - q["tam"][0]) ** 2 + (du_doan_y - q["tam"][1]) ** 2)
            if kc < max(70, 0.5 * q["d_px"]):                   # đủ gần mới cho ghép
                cac_cap.append((kc, id_qua, j))
    cac_cap.sort()                                              # cặp gần nhất ghép trước

    ids = [None] * len(ds_qua)
    id_da_ghep = []
    for kc, id_qua, j in cac_cap:
        if id_qua not in id_da_ghep and ids[j] is None:
            ids[j] = id_qua
            id_da_ghep.append(id_qua)

    # 2. Quả chưa ghép được -> quả mới xuất hiện -> cấp ID mới
    for j, q in enumerate(ds_qua):
        if ids[j] is None:
            ids[j] = bo_dem["id_tiep_theo"]
            bo_dem["id_tiep_theo"] += 1
            vet[ids[j]] = {"tam": q["tam"], "van_toc": (0, 0), "so_khung_mat": 0,
                           "phieu_do_chin": [], "duong_kinh_px": [], "da_dem": False}
            id_da_ghep.append(ids[j])

    # 3. ID nào mất dấu quá 6 khung hình (quả đã ra khỏi ảnh) -> xóa
    for id_qua in list(vet.keys()):
        if id_qua not in id_da_ghep:
            vet[id_qua]["so_khung_mat"] += 1
            if vet[id_qua]["so_khung_mat"] > 6:
                del vet[id_qua]

    # 4. Cập nhật từng quả và kiểm tra có vừa đi qua vạch không
    vua_dem = []
    for j, q in enumerate(ds_qua):
        v = vet[ids[j]]
        x_cu = v["tam"][0]
        x_moi = q["tam"][0]
        if len(v["phieu_do_chin"]) > 0:
            v["van_toc"] = (q["tam"][0] - v["tam"][0], q["tam"][1] - v["tam"][1])
        v["tam"] = q["tam"]
        v["so_khung_mat"] = 0
        v["phieu_do_chin"].append(q["do_chin"])                # mỗi khung "bỏ phiếu" 1 lần
        if not q["cham_bien"]:                                  # quả bị cắt ở mép thì không lấy kích thước
            v["duong_kinh_px"].append(q["d_px"])

        # Đi qua vạch: khung trước ở 1 bên vạch, khung này ở bên kia  ->  (x_cu - vạch) * (x_moi - vạch) <= 0
        qua_vach = (x_cu - x_vach) * (x_moi - x_vach) <= 0 and x_cu != x_moi
        if qua_vach and not v["da_dem"] and len(v["phieu_do_chin"]) > 1:
            v["da_dem"] = True
            do_chin = np.bincount(v["phieu_do_chin"]).argmax()             # độ chín được nhiều phiếu nhất
            if v["duong_kinh_px"] and k:
                d_mm = np.median(v["duong_kinh_px"]) / k                   # trung vị: bỏ khung đo lệch
            else:
                d_mm = None
            co = phan_loai_co(d_mm)
            if co in TEN_CO:
                bo_dem["bang"][do_chin, TEN_CO.index(co)] += 1
            vua_dem.append(ids[j])
    return ids, vua_dem

def _them_tieu_de(anh, chu):
    anh = cv2.resize(anh, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.rectangle(anh, (0, 0), (anh.shape[1], 24), (40, 30, 20), -1)
    cv2.putText(anh, chu, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return anh


# Ghép 4 ảnh minh họa
# 1 vật chuẩn -> k | 2 biến đổi khoảng cách | 3 tách quả + trọng tâm | 4 đường kính -> cỡ S/M/L
def ve_4_buoc_phan2(anh):
    kq = phan_tich_anh(anh)

    # Ô 1: vật chuẩn và hệ số k
    o1 = anh.copy()
    if kq["vat_chuan"] is not None:
        x, y, w, h = kq["vat_chuan"]
        cv2.rectangle(o1, (x - 3, y - 3), (x + w + 2, y + h + 2), (255, 200, 0), 3)
        cv2.putText(o1, f"canh {kq['k'] * CANH_VAT_CHUAN_MM:.0f}px / 20mm = {kq['k']:.3f} px/mm",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 120, 0), 2, cv2.LINE_AA)

    # Ô 2: biến đổi khoảng cách (đỏ = xa mép = tâm quả)
    dist = cv2.distanceTransform(kq["mask"], cv2.DIST_L2, 5)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    o2 = cv2.applyColorMap(dist, cv2.COLORMAP_JET)
    o2[kq["mask"] == 0] = 0

    # Ô 3: mỗi quả 1 màu + dấu cộng ở trọng tâm.  Ô 4: vòng tròn đường kính tương đương
    o3 = np.zeros_like(anh)
    o4 = anh.copy()
    mau = [(255, 120, 120), (120, 255, 120), (120, 120, 255), (255, 255, 120), (255, 120, 255), (120, 255, 255)]
    for i, q in enumerate(kq["qua"]):
        cx, cy = int(q["tam"][0]), int(q["tam"][1])
        cv2.drawContours(o3, [q["vien"]], -1, mau[i % len(mau)], -1)
        cv2.drawMarker(o3, (cx, cy), (0, 0, 0), cv2.MARKER_CROSS, 16, 2)
        cv2.circle(o4, (cx, cy), int(q["d_px"] / 2), (255, 0, 255), 2, cv2.LINE_AA)
        chu = f"{q['d_mm']:.1f}mm {q['co']}" if q["d_mm"] else "?"
        cv2.putText(o4, chu, (cx - 45, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    ghep = np.vstack([np.hstack([_them_tieu_de(o1, "1. Vat chuan 20x20mm -> he so k"),
                                 _them_tieu_de(o2, "2. Bien doi khoang cach: dinh = tam qua")]),
                      np.hstack([_them_tieu_de(o3, f"3. Tach tung qua + trong tam: {len(kq['qua'])} qua"),
                                 _them_tieu_de(o4, "4. D = 2*sqrt(A/pi) / k -> co S/M/L")])])
    return ghep, kq


if __name__ == "__main__":
    import sys
    from part_3_simulation import tao_anh_tinh, danh_gia_anh_tinh
    dap_an = None
    if len(sys.argv) > 1:          # ảnh thật:  python part_2_measuring_and_counting.py picture/picture_test_1.jpg
        anh = cv2.imdecode(np.fromfile(sys.argv[1], np.uint8), cv2.IMREAD_COLOR)
    else:                          # ảnh mô phỏng có quả chạm nhau
        anh, dap_an, _ = tao_anh_tinh(so_qua=9, seed=6, cho_cham=True)

    ghep, kq = ve_4_buoc_phan2(anh)
    print(f"He so k = {kq['k']:.3f} px/mm" if kq["k"] else "Khong thay vat chuan")
    print(f"{'Tam (x, y)':<15}{'A (px)':>8}{'D (px)':>8}{'D (mm)':>8}{'Co':>4}{'Dap an (mm)':>13}")
    for q in kq["qua"]:
        dung = 0
        if dap_an:
            g = min(dap_an, key=lambda g: (g["tam"][0] - q["tam"][0]) ** 2 + (g["tam"][1] - q["tam"][1]) ** 2)
            dung = g["d_mm"]
        print(f"({q['tam'][0]:5.0f},{q['tam'][1]:5.0f})  {q['dien_tich']:>8}{q['d_px']:>8.1f}"
              f"{q['d_mm'] or 0:>8.1f}{q['co']:>4}{dung:>13.1f}")
    print("Bang dem (hang: Xanh, Uong, Chin  |  cot: S, M, L):")
    print(dem_anh_tinh(kq["qua"]))
    if dap_an:
        dg = danh_gia_anh_tinh(kq["qua"], dap_an)
        print(f"Dap an {dg['so_qua_that']} qua, dem duoc {dg['so_qua_dem']}; "
              f"sai so duong kinh TB {np.mean(dg['sai_so_mm']):.2f} mm")
    cv2.imshow("PHAN 2 - Do kich thuoc & dem (Nhan phim bat ky de thoat)", ghep)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
