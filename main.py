"""
GIAO DIỆN CHƯƠNG TRÌNH 
PHÂN LOẠI ĐỘ CHÍN, ĐO KÍCH THƯỚC VÀ ĐẾM CÀ CHUA BẰNG XỬ LÝ ẢNH
Bao gồm:
  - Lấy ảnh từ 3 nguồn: ảnh mô phỏng / băng tải mô phỏng / camera điện thoại
  - Gọi phan_tich_anh() (Phần 1 + Phần 2) cho mỗi ảnh
  - Vẽ kết quả lên ảnh, hiện lên cửa sổ, cập nhật bảng đếm

Giao diện dùng thư viện Tkinter (có sẵn trong Python). Ý chính:
  - tk.Button(..., command=ham): bấm nút thì gọi hàm 'ham'
  - root.after(t, ham): hẹn sau t mili giây gọi 'ham' -> dùng để chạy vòng lặp video

Phím tắt:  N = tạo ảnh mới   B = chạy băng tải   Space = tạm dừng / tiếp tục

Camera điện thoại (điện thoại và máy tính chung 1 mạng Wi-Fi):
  - App DroidCam  : nhập  http://<IP điện thoại>:4747/video
  - App IP Webcam : nhập  http://<IP điện thoại>:8080/video
  - Webcam laptop / DroidCam Client: nhập số 0, 1 hoặc 2
  - Hoặc đường dẫn file video quay sẵn, ví dụ: picture/video_test_1.mp4
"""

import base64
import csv
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np

from part_1_classification import TEN_DO_CHIN, S_MIN, I_MIN, anh_kenh_H
from part_2_measuring_and_counting import (phan_tich_anh, dem_anh_tinh, tao_bo_dem, cap_nhat_bo_dem,
                             cap_nhat_kich_thuoc, TEN_CO)
from main_screen import ManHinhDau
from part_3_simulation import (tao_anh_tinh, tao_bang_tai, khung_hinh_bang_tai, bang_tai_da_xong,
                            danh_gia_anh_tinh, danh_gia_bang_tai)

THU_MUC = os.path.dirname(os.path.abspath(__file__))     # thư mục chứa file này (để lưu CSV, ảnh)


# Màu giao diện
# HEX cho Tkinter | BGR cho OpenCV
NEN = "#0b1220"        # nền cửa sổ
THE = "#141d2c"        # nền các khung
THE2 = "#1b2638"       # nền ô bảng
VIEN = "#2a3850"       # đường viền
CHU = "#e6ebf2"        # chữ chính
CHU_MO = "#8b9ab0"     # chữ phụ
DO = "#ef4444"         # màu nhấn (đỏ cà chua)
FONT = "Segoe UI"

MAU_DO_CHIN_HEX = ["#22c55e", "#f59e0b", "#ef4444"]               # Xanh, Ương, Chín (chữ giao diện)
MAU_DO_CHIN_BGR = [(94, 197, 34), (11, 158, 245), (68, 68, 239)]  # cùng màu, thứ tự BGR (OpenCV)
TEN_VN = ["Xanh", "Ương", "Chín"]


# Vẽ nhãn chữ
# Hộp màu + chữ trắng
def ve_nhan(anh, chu, x, y, mau):
    """Vẽ 1 nhãn chữ trắng trên nền màu."""
    (w, h), _ = cv2.getTextSize(chu, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(y, h + 8)
    cv2.rectangle(anh, (x, y - h - 8), (x + w + 10, y), mau, -1)
    cv2.putText(anh, chu, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# Vẽ kết quả lên ảnh
# Khung vật chuẩn + k | vạch đếm (nháy xanh khi vừa đếm) | viền + tâm + nhãn từng quả
# Băng tải: thêm số ID
def ve_ket_qua(anh, kq, ids=None, x_vach=None, vach_sang=False):
    """Vẽ: khung vật chuẩn, vạch đếm, đường viền + nhãn 'Chin M 55.3mm' cho từng quả."""
    anh = anh.copy()
    if kq["vat_chuan"] is not None:
        x, y, w, h = kq["vat_chuan"]
        cv2.rectangle(anh, (x - 2, y - 2), (x + w + 1, y + h + 1), (255, 200, 0), 2)
        ve_nhan(anh, f"20mm  k={kq['k']:.2f} px/mm", x + w + 6, y + h // 2 + 8, (180, 130, 0))
    if x_vach is not None:
        mau = (80, 220, 80) if vach_sang else (255, 80, 255)     # vạch nháy xanh khi vừa đếm 1 quả
        cv2.line(anh, (int(x_vach), 0), (int(x_vach), anh.shape[0]), mau, 3 if vach_sang else 2)
    for i, q in enumerate(kq["qua"]):
        mau = MAU_DO_CHIN_BGR[q["do_chin"]]
        cv2.drawContours(anh, [q["vien"]], -1, mau, 2, cv2.LINE_AA)
        cv2.circle(anh, (int(q["tam"][0]), int(q["tam"][1])), 3, (255, 255, 255), -1)
        duong_kinh = f"{q['d_mm']:.1f}mm" if q["d_mm"] is not None else "?mm"
        chu = f"{TEN_DO_CHIN[q['do_chin']]} {q['co']} {duong_kinh}"
        if ids is not None:
            chu = f"#{ids[i]} " + chu                              # băng tải: thêm số ID
        ve_nhan(anh, chu, q["khung"][0], q["khung"][1] - 4, mau)
    return anh


# Chế độ xem 4 ô
# Ảnh gốc | mặt nạ | kênh H | kết quả
def ghep_4_buoc(anh_goc, kq, anh_ket_qua):
    """Lưới 2 x 2 để giải thích: ảnh gốc | mặt nạ  /  kênh H | kết quả."""
    kenh_H = anh_kenh_H(kq["hsi"], kq["mask"])                 # kênh H của HSI
    kenh_H[kq["mask"] == 0] = (30, 24, 20)
    cac_o = [(anh_goc, "1. Anh goc"),
             (cv2.cvtColor(kq["mask"], cv2.COLOR_GRAY2BGR), "2. Mat na nhi phan (Phan 1)"),
             (kenh_H, "3. Kenh H cua HSI trong qua (Phan 1)"),
             (anh_ket_qua, "4. Ket qua: do chin + kich thuoc (Phan 2)")]
    nho = []
    for anh, ten in cac_o:
        anh = cv2.resize(anh, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        ve_nhan(anh, ten, 0, 22, (60, 45, 30))
        nho.append(anh)
    return np.vstack([np.hstack(nho[:2]), np.hstack(nho[2:])])


# Đổi ảnh OpenCV -> Tkinter
# Định dạng PPM (nhanh), lỗi thì dùng PNG
def sang_anh_tkinter(anh_bgr):
    """Đổi ảnh OpenCV (BGR) sang ảnh Tkinter hiển thị được.
    Dùng định dạng PPM (chỉ là header + dữ liệu RGB thô) nên rất nhanh; lỗi thì dùng PNG."""
    rgb = cv2.cvtColor(anh_bgr, cv2.COLOR_BGR2RGB)
    cao, rong = rgb.shape[:2]
    try:
        return tk.PhotoImage(width=rong, height=cao, data=b"P6 %d %d 255 " % (rong, cao) + rgb.tobytes(),
                             format="PPM")
    except tk.TclError:
        _, buf = cv2.imencode(".png", anh_bgr)
        return tk.PhotoImage(data=base64.b64encode(buf.tobytes()))


# Đọc camera ở luồng riêng (thread)
# cap.read() có thể chờ lâu -> luồng nền đọc liên tục, chỉ giữ khung mới nhất
# Lock: tránh 2 luồng cùng đọc / ghi self.khung
# Nguồn: số 0,1,2 (webcam) | http://... (DroidCam, IP Webcam) | file video
class CameraLuong:
    """Đọc camera trong 1 LUỒNG riêng (thread), chạy song song với giao diện.
    Luồng này đọc liên tục và chỉ giữ khung hình MỚI NHẤT, nên dù xử lý chậm hơn camera
    thì hình vẫn không bị trễ. Việc kết nối (có thể mất vài giây) cũng nằm trong luồng
    nên cửa sổ không bị treo."""

    def __init__(self, nguon):
        self.nguon = nguon.strip()
        self.trang_thai = "dang_mo"      # "dang_mo" -> "ok" hoặc "loi"
        self.khung = None                # khung hình mới nhất
        self.dang_chay = True
        self.khoa = threading.Lock()     # khóa: tránh 2 luồng cùng đọc/ghi self.khung 1 lúc
        threading.Thread(target=self.doc_lien_tuc, daemon=True).start()

    def doc_lien_tuc(self):
        if self.nguon.isdigit():                         # số 0, 1, 2: webcam / DroidCam Client
            if sys.platform.startswith("win"):
                cap = cv2.VideoCapture(int(self.nguon), cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(int(self.nguon))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:                                            # địa chỉ http://... hoặc file video
            cap = cv2.VideoCapture(self.nguon)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            self.trang_thai = "loi"
            return
        self.trang_thai = "ok"

        la_file = os.path.isfile(self.nguon)
        fps_file = cap.get(cv2.CAP_PROP_FPS) or 30
        while self.dang_chay:
            bat_dau = time.perf_counter()
            ok, khung = cap.read()
            if ok:
                with self.khoa:
                    self.khung = khung
            elif la_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)      # hết file video -> phát lại từ đầu
            else:
                time.sleep(0.02)
            if la_file:                                  # file: đọc đúng tốc độ thật của video
                time.sleep(max(0.0, 1 / fps_file - (time.perf_counter() - bat_dau)))
        cap.release()

    def doc(self):
        with self.khoa:
            return None if self.khung is None else self.khung.copy()

    def dong(self):
        self.dang_chay = False


# Cửa sổ chính
# Trạng thái: nguon (anh / bang_tai / camera), anh_goc, dap_an, dang_chay, bo_dem, bang_dem
# Nhóm hàm: dựng giao diện | hiển thị | ảnh tĩnh | băng tải | camera | xuất dữ liệu
class UngDung:
    # KHỞI TẠO: các biến lưu trạng thái + dựng giao diện
    def __init__(self, root):
        self.root = root
        root.title("Phân loại - Đo - Đếm cà chua | Xử lý ảnh")
        root.configure(bg=NEN)
        root.minsize(self.px(1200), self.px(720))
        try:
            root.state("zoomed")                 # phóng to toàn màn hình (Windows)
        except tk.TclError:
            root.geometry("1440x880")

        self.nguon = "anh"                       # nguồn đang dùng: "anh" / "bang_tai" / "camera"
        self.anh_goc = None                      # ảnh đang xem
        self.dap_an = None                       # đáp án ảnh mô phỏng (None nếu ảnh thật)
        self.seed = 1                            # số seed của ảnh mô phỏng đang xem
        self.bang_tai = None
        self.camera = None
        self.dang_chay = False                   # vòng lặp video (băng tải / camera) có đang chạy không
        self.bo_dem = tao_bo_dem()               # bộ đếm qua vạch (Phần 2)
        self.bang_dem = np.zeros((3, 3), int)    # bảng đếm đang hiển thị
        self.lich_su_k = []                      # k của các khung camera gần đây
        self.fps = 0.0
        self.moc_thoi_gian = time.perf_counter()
        self.nhay_vach = 0                       # số khung còn nháy vạch đếm
        self.ket_qua_cuoi = None                 # (ảnh gốc, kq, ảnh kết quả) để vẽ lại khi đổi chế độ xem
        self.hen_gio = None

        self.dung_giao_dien()

        # Phím tắt (bỏ qua khi đang gõ chữ trong ô nhập địa chỉ camera)
        root.bind("<Key-n>", lambda e: None if isinstance(e.widget, tk.Entry) else self.tao_anh_moi())
        root.bind("<Key-b>", lambda e: None if isinstance(e.widget, tk.Entry) else self.chay_bang_tai())
        root.bind("<space>", lambda e: None if isinstance(e.widget, tk.Entry) else self.tam_dung())
        root.protocol("WM_DELETE_WINDOW", self.thoat)        # bấm X: ngắt camera rồi mới đóng
        root.after(100, self.tao_anh_moi)                    # mở lên là có sẵn 1 ảnh mô phỏng
    # CÁC HÀM DỰNG GIAO DIỆN NHỎ

    # Đổi pixel theo DPI Windows
    def px(self, n):
        """Đổi số pixel thiết kế (ở 96 DPI) sang pixel thật theo độ phóng đại Windows."""
        return int(n * self.root.winfo_fpixels("1i") / 96)

    # Hàm dựng giao diện dùng lại
    # tao_khung | tao_nut | tao_thanh_truot
    def tao_khung(self, cha, tieu_de):
        """1 khung có tiêu đề. Trả về phần thân để đặt nút, thanh trượt vào."""
        khung = tk.Frame(cha, bg=THE, highlightbackground=VIEN, highlightthickness=1)
        khung.pack(fill="x", pady=(0, 10))
        tk.Label(khung, text=tieu_de.upper(), bg=THE, fg=CHU_MO,
                 font=(FONT, 9, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        than = tk.Frame(khung, bg=THE)
        than.pack(fill="x", padx=12, pady=(0, 12))
        return than

    def tao_nut(self, cha, chu, ham, mau=DO, **vi_tri):
        nut = tk.Button(cha, text=chu, command=ham, bg=mau, fg="white", activebackground=mau,
                        activeforeground="white", relief="flat", bd=0, font=(FONT, 10, "bold"),
                        cursor="hand2", padx=10, pady=7)
        nut.pack(fill="x", pady=3, **vi_tri)
        return nut

    def tao_thanh_truot(self, cha, ten, nho_nhat, lon_nhat, mac_dinh, buoc, ham_khi_doi, don_vi=""):
        """Thanh trượt có tên bên trái, giá trị bên phải. Trả về biến chứa giá trị (.get() để đọc)."""
        gia_tri = tk.DoubleVar(value=mac_dinh)
        hang = tk.Frame(cha, bg=THE)
        hang.pack(fill="x", pady=(6, 0))
        tk.Label(hang, text=ten, bg=THE, fg=CHU, font=(FONT, 10)).pack(side="left")
        nhan = tk.Label(hang, text=f"{mac_dinh:g}{don_vi}", bg=THE, fg=DO, font=(FONT, 10, "bold"))
        nhan.pack(side="right")

        def khi_keo(_):
            nhan.config(text=f"{gia_tri.get():g}{don_vi}")
            ham_khi_doi()

        tk.Scale(cha, from_=nho_nhat, to=lon_nhat, resolution=buoc, orient="horizontal", variable=gia_tri,
                 showvalue=False, bg=THE, troughcolor=VIEN, highlightthickness=0, bd=0, sliderrelief="flat",
                 activebackground=DO, width=10, sliderlength=18, command=khi_keo).pack(fill="x")
        return gia_tri

    # Dựng giao diện 3 cột
    # Trái: điều khiển (nguồn ảnh, camera, thông số mô phỏng, ngưỡng S / I, k thủ công, xuất dữ liệu)
    # Giữa: ảnh + chế độ xem (Kết quả / Mặt nạ / 4 ô)
    # Phải: tổng số quả, bảng đếm 3 x 3, so với đáp án, thông tin
    def dung_giao_dien(self):
        # ---- Đầu trang
        dau = tk.Frame(self.root, bg=NEN)
        dau.pack(fill="x", padx=18, pady=(14, 6))
        tk.Label(dau, text="●", bg=NEN, fg=DO, font=(FONT, 20)).pack(side="left")
        tk.Label(dau, text=" PHÂN LOẠI ĐỘ CHÍN · ĐO KÍCH THƯỚC · ĐẾM CÀ CHUA", bg=NEN, fg=CHU,
                 font=(FONT, 16, "bold")).pack(side="left")
        tk.Label(dau, text="Đề tài môn học Xử lý ảnh", bg=NEN, fg=CHU_MO,
                 font=(FONT, 10)).pack(side="right")

        than = tk.Frame(self.root, bg=NEN)
        than.pack(fill="both", expand=True, padx=18)

        # ---- Cột trái đặt trong Canvas để CUỘN được bằng con lăn chuột (màn hình laptop thấp)
        vo = tk.Frame(than, bg=NEN)
        vo.pack(side="left", fill="y")
        canvas = tk.Canvas(vo, bg=NEN, highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        trai = tk.Frame(canvas, bg=NEN)
        cua_so_trai = canvas.create_window(0, 0, window=trai, anchor="nw")
        trai.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(cua_so_trai, width=e.width))

        def lan_chuot(e):
            canvas.yview_scroll(-1 if (e.delta > 0 or e.num == 4) else 1, "units")
        canvas.bind("<Enter>", lambda e: (canvas.bind_all("<MouseWheel>", lan_chuot),
                                          canvas.bind_all("<Button-4>", lan_chuot),
                                          canvas.bind_all("<Button-5>", lan_chuot)))
        canvas.bind("<Leave>", lambda e: (canvas.unbind_all("<MouseWheel>"),
                                          canvas.unbind_all("<Button-4>"),
                                          canvas.unbind_all("<Button-5>")))

        phai = tk.Frame(than, bg=NEN)
        phai.pack(side="right", fill="y")
        tk.Frame(phai, bg=NEN, width=self.px(390), height=1).pack()     # giữ rộng tối thiểu (theo DPI)
        giua = tk.Frame(than, bg=NEN)
        giua.pack(side="left", fill="both", expand=True, padx=14)

        # ======== CỘT TRÁI ========
        k = self.tao_khung(trai, "Nguồn ảnh mô phỏng")
        self.tao_nut(k, "Tạo ảnh mô phỏng mới  (N)", self.tao_anh_moi)
        self.tao_nut(k, "Chạy băng tải  (B)", self.chay_bang_tai, mau="#2563eb")
        self.tao_nut(k, "Mở ảnh từ máy...", self.mo_anh, mau="#334155")

        k = self.tao_khung(trai, "Camera điện thoại (thực nghiệm)")
        self.v_dia_chi = tk.StringVar(value="http://192.168.1.10:4747/video")
        tk.Entry(k, textvariable=self.v_dia_chi, bg=THE2, fg=CHU, insertbackground=CHU, relief="flat",
                 font=(FONT, 10)).pack(fill="x", ipady=5)
        tk.Label(k, text="DroidCam:   http://IP:4747/video\nIP Webcam:  http://IP:8080/video\n"
                         "Webcam / DroidCam Client:  0, 1, 2\nHoặc đường dẫn file video quay sẵn (.mp4)",
                 bg=THE, fg=CHU_MO, font=(FONT, 8), justify="left").pack(anchor="w", pady=(4, 2))
        hang = tk.Frame(k, bg=THE)
        hang.pack(fill="x")
        self.tao_nut(hang, "Kết nối", self.ket_noi_camera, mau="#16a34a", side="left", expand=True, padx=(0, 3))
        self.tao_nut(hang, "Ngắt", self.ngat_camera, mau="#475569", side="left", expand=True, padx=(3, 0))
        self.v_kieu_dem = tk.StringVar(value="tinh")
        hang = tk.Frame(k, bg=THE)
        hang.pack(fill="x", pady=(6, 0))
        for gia_tri, chu in (("tinh", "Đếm tĩnh"), ("vach", "Đếm qua vạch")):
            tk.Radiobutton(hang, text=chu, value=gia_tri, variable=self.v_kieu_dem, command=self.dat_lai,
                           indicatoron=False, bg=THE2, fg=CHU, selectcolor="#2563eb", activebackground=VIEN,
                           activeforeground=CHU, relief="flat", bd=0, font=(FONT, 9, "bold"), pady=5,
                           cursor="hand2").pack(side="left", fill="x", expand=True, padx=1)
        self.v_vi_tri_vach = self.tao_thanh_truot(k, "Vị trí vạch đếm", 10, 90, 50, 1, lambda: None, "%")

        k = self.tao_khung(trai, "Thông số mô phỏng")
        self.v_so_qua = self.tao_thanh_truot(k, "Số quả", 1, 12, 9, 1, self.khi_doi_canh)
        self.v_do_sang = self.tao_thanh_truot(k, "Độ sáng", 40, 120, 100, 5, self.khi_doi_canh, "%")
        self.v_nhieu = self.tao_thanh_truot(k, "Nhiễu Gaussian σ", 0, 25, 0, 1, self.khi_doi_canh)
        self.v_toc_do = self.tao_thanh_truot(k, "Tốc độ băng tải", 4, 24, 10, 1, lambda: None, " px/khung")
        self.v_cham_nhau = tk.BooleanVar(value=False)
        tk.Checkbutton(k, text="Cho quả chạm nhau", variable=self.v_cham_nhau, command=self.khi_doi_canh,
                       bg=THE, fg=CHU, selectcolor=THE2, activebackground=THE, activeforeground=CHU,
                       font=(FONT, 10), highlightthickness=0, bd=0).pack(anchor="w", pady=(8, 0))

        k = self.tao_khung(trai, "Ngưỡng tách nền (Phần 1)")
        self.v_s_min = self.tao_thanh_truot(k, "S nhỏ nhất", 10, 150, S_MIN, 5, self.khi_doi_nguong)
        self.v_v_min = self.tao_thanh_truot(k, "I nhỏ nhất", 0, 100, I_MIN, 1, self.khi_doi_nguong)

        # Ảnh không có vật chuẩn (ví dụ ảnh trên mạng): không thể biết 1 pixel bằng bao nhiêu mm.
        # Cho phép nhập k bằng tay, ví dụ đo 1 quả bằng thước rồi tính k = đường kính (px) / đường kính (mm).
        k = self.tao_khung(trai, "Khi không có vật chuẩn")
        hang = tk.Frame(k, bg=THE)
        hang.pack(fill="x")
        tk.Label(hang, text="k thủ công (px/mm):", bg=THE, fg=CHU, font=(FONT, 10)).pack(side="left")
        self.v_k_tay = tk.StringVar(value="")
        o_k = tk.Entry(hang, textvariable=self.v_k_tay, width=7, bg=THE2, fg=CHU, insertbackground=CHU,
                       relief="flat", font=(FONT, 10))
        o_k.pack(side="right", ipady=3)
        o_k.bind("<Return>", lambda e: self.khi_doi_nguong())
        tk.Label(k, text="Để trống = không dùng. Nhập xong bấm Enter.\nChỉ dùng khi ảnh KHÔNG có ô vuông 20 mm.",
                 bg=THE, fg=CHU_MO, font=(FONT, 8), justify="left").pack(anchor="w", pady=(4, 0))

        k = self.tao_khung(trai, "Xuất dữ liệu")
        hang = tk.Frame(k, bg=THE)
        hang.pack(fill="x")
        for chu, ham in (("Đặt lại", self.dat_lai), ("Lưu CSV", self.luu_csv), ("Lưu ảnh", self.luu_anh)):
            tk.Button(hang, text=chu, command=ham, bg=THE2, fg=CHU, activebackground=VIEN, activeforeground=CHU,
                      relief="flat", bd=0, font=(FONT, 9, "bold"), cursor="hand2", pady=6
                      ).pack(side="left", fill="x", expand=True, padx=2)

        # ======== CỘT GIỮA: ẢNH ========
        thanh = tk.Frame(giua, bg=NEN)
        thanh.pack(fill="x", pady=(0, 6))
        self.v_che_do_xem = tk.StringVar(value="ket_qua")
        for gia_tri, chu in (("ket_qua", "Kết quả"), ("mask", "Mặt nạ"), ("4_buoc", "Các bước (4 ô)")):
            tk.Radiobutton(thanh, text=chu, value=gia_tri, variable=self.v_che_do_xem, indicatoron=False,
                           command=self.ve_lai, bg=THE, fg=CHU, selectcolor=DO, activebackground=THE2,
                           activeforeground=CHU, relief="flat", bd=0, font=(FONT, 10, "bold"), padx=14, pady=5,
                           cursor="hand2").pack(side="left", padx=(0, 4))
        self.lbl_trang_thai = tk.Label(thanh, text="", bg=NEN, fg=CHU_MO, font=(FONT, 10))
        self.lbl_trang_thai.pack(side="right")
        khung_anh = tk.Frame(giua, bg="#05080f", highlightbackground=VIEN, highlightthickness=1)
        khung_anh.pack(fill="both", expand=True)
        self.lbl_anh = tk.Label(khung_anh, bg="#05080f")
        self.lbl_anh.pack(fill="both", expand=True)
        self.lbl_anh.bind("<Configure>", lambda e: self.ve_lai())        # đổi kích thước cửa sổ -> vẽ lại

        # ======== CỘT PHẢI: KẾT QUẢ ========
        k = self.tao_khung(phai, "Tổng số quả")
        self.lbl_tong = tk.Label(k, text="0", bg=THE, fg=CHU, font=(FONT, 34, "bold"))
        self.lbl_tong.pack(anchor="w")
        self.lbl_che_do = tk.Label(k, text="", bg=THE, fg=CHU_MO, font=(FONT, 9), wraplength=self.px(350), justify="left")
        self.lbl_che_do.pack(anchor="w")

        k = self.tao_khung(phai, "Bảng đếm độ chín × cỡ")
        self.o_bang = {}                                   # ô số trong bảng: o_bang[(hàng, cột)]
        for cot, chu in enumerate(["", "S", "M", "L", "Tổng"]):
            tk.Label(k, text=chu, bg=THE, fg=CHU_MO, font=(FONT, 9, "bold"),
                     width=5).grid(row=0, column=cot, pady=(0, 4))
        for hang_so, do_chin in enumerate([2, 1, 0], start=1):          # hàng: Chín, Ương, Xanh
            tk.Label(k, text="● " + TEN_VN[do_chin], bg=THE, fg=MAU_DO_CHIN_HEX[do_chin],
                     font=(FONT, 10, "bold"), anchor="w", width=7).grid(row=hang_so, column=0, sticky="w", pady=3)
            for cot in range(1, 5):
                o = tk.Label(k, text="0", bg=THE2, fg=CHU, font=(FONT, 11, "bold"), width=5, pady=3)
                o.grid(row=hang_so, column=cot, padx=2, pady=3)
                self.o_bang[(do_chin, cot)] = o
        tk.Label(k, text="Tổng", bg=THE, fg=CHU_MO, font=(FONT, 10, "bold"),
                 anchor="w").grid(row=4, column=0, sticky="w")
        for cot in range(1, 5):
            o = tk.Label(k, text="0", bg=THE, fg=CHU, font=(FONT, 11, "bold"), width=5)
            o.grid(row=4, column=cot, padx=2, pady=(6, 0))
            self.o_bang[("tong", cot)] = o

        k = self.tao_khung(phai, "So với đáp án mô phỏng")
        self.lbl_danh_gia = tk.Label(k, text="—", bg=THE, fg=CHU, font=(FONT, 10), justify="left",
                                     anchor="w", wraplength=self.px(350))
        self.lbl_danh_gia.pack(fill="x")

        k = self.tao_khung(phai, "Thông tin")
        self.lbl_thong_tin = tk.Label(k, text="", bg=THE, fg=CHU_MO, font=(FONT, 10), justify="left", anchor="w",
                                      wraplength=self.px(350))
        self.lbl_thong_tin.pack(fill="x")

        self.root.update_idletasks()
        canvas.config(width=trai.winfo_reqwidth())          # cột trái rộng đúng bằng nội dung, không bị cắt chữ

        tk.Label(self.root, text="Hệ Thống Phân Loại Độ Chín, Đo Kích Thước Và Đếm Cà Chua",
                 bg=NEN, fg=CHU_MO, font=(FONT, 9)).pack(pady=(4, 8))

    # Hiển thị ảnh
    # Thu / phóng vừa khung | giữ ảnh trong self.anh_tk (nếu không Python xóa ảnh)
    def hien_anh(self, anh):
        """Thu/phóng ảnh cho vừa khung giữa rồi hiện lên."""
        rong_khung = max(self.lbl_anh.winfo_width() - 4, 200)
        cao_khung = max(self.lbl_anh.winfo_height() - 4, 150)
        ti_le = min(rong_khung / anh.shape[1], cao_khung / anh.shape[0])
        anh = cv2.resize(anh, (int(anh.shape[1] * ti_le), int(anh.shape[0] * ti_le)),
                         interpolation=cv2.INTER_AREA if ti_le < 1 else cv2.INTER_LINEAR)
        self.anh_tk = sang_anh_tkinter(anh)        # phải lưu vào self, nếu không Python xóa mất ảnh
        self.lbl_anh.config(image=self.anh_tk)

    # Chọn chế độ xem
    # Lưu ket_qua_cuoi để đổi chế độ xem không cần xử lý lại
    def hien_ket_qua(self, anh_goc, kq, anh_ket_qua):
        """Hiện theo chế độ xem đang chọn: Kết quả / Mặt nạ / 4 ô."""
        self.ket_qua_cuoi = (anh_goc, kq, anh_ket_qua)
        che_do = self.v_che_do_xem.get()
        if che_do == "mask":
            self.hien_anh(cv2.cvtColor(kq["mask"], cv2.COLOR_GRAY2BGR))
        elif che_do == "4_buoc":
            self.hien_anh(ghep_4_buoc(anh_goc, kq, anh_ket_qua))
        else:
            self.hien_anh(anh_ket_qua)

    def ve_lai(self):
        if self.ket_qua_cuoi is not None:
            self.hien_ket_qua(*self.ket_qua_cuoi)

    # Cập nhật bảng đếm
    # 3 x 3 + tổng hàng + tổng cột + tổng số quả
    def cap_nhat_bang(self, bang, tong=None):
        """Ghi bảng 3 x 3 (hàng = độ chín, cột = S, M, L) lên giao diện, kèm tổng hàng / cột."""
        self.bang_dem = bang
        for do_chin in range(3):
            for cot in range(3):
                self.o_bang[(do_chin, cot + 1)].config(text=str(bang[do_chin, cot]))
            self.o_bang[(do_chin, 4)].config(text=str(bang[do_chin].sum()))       # tổng hàng
        for cot in range(3):
            self.o_bang[("tong", cot + 1)].config(text=str(bang[:, cot].sum()))   # tổng cột
        self.o_bang[("tong", 4)].config(text=str(bang.sum()))
        self.lbl_tong.config(text=str(bang.sum() if tong is None else tong))

    def hien_thong_tin(self, kq, them=""):
        if kq["vat_chuan"] is not None:
            k = f"{kq['k']:.3f} px/mm (từ vật chuẩn)"
        elif kq["k"]:
            k = f"{kq['k']:.3f} px/mm (nhập tay / khung trước)"
        else:
            k = "không thấy vật chuẩn"
        self.lbl_thong_tin.config(text=f"Hệ số k: {k}\nTốc độ xử lý: {self.fps:.0f} khung/s{them}")

    # Tốc độ xử lý (khung/s)
    # Làm mượt: 90% giá trị cũ + 10% giá trị mới
    def do_fps(self):
        """FPS làm mượt: 90% giá trị cũ + 10% giá trị mới (để số không nhảy loạn)."""
        bay_gio = time.perf_counter()
        self.fps = 0.9 * self.fps + 0.1 / max(bay_gio - self.moc_thoi_gian, 1e-6)
        self.moc_thoi_gian = bay_gio

    def lay_nguong(self):
        return int(self.v_s_min.get()), int(self.v_v_min.get())

    def lay_k_tay(self):
        """Đọc ô 'k thủ công'. Trống hoặc gõ sai thì trả về None."""
        try:
            k = float(self.v_k_tay.get().replace(",", "."))
            return k if k > 0 else None
        except ValueError:
            return None

    # Nguồn 1 - Ảnh tĩnh
    # N: ảnh mô phỏng mới (seed mới)
    def tao_anh_moi(self):
        self.ngat_camera()
        self.seed = int(time.time() * 1000) % 100000           # seed mới -> ảnh mới
        self.ve_canh_mo_phong()

    def ve_canh_mo_phong(self):
        self.dang_chay = False
        self.nguon = "anh"
        self.anh_goc, self.dap_an, _ = tao_anh_tinh(
            so_qua=int(self.v_so_qua.get()), seed=self.seed, do_sang=self.v_do_sang.get() / 100,
            nhieu=self.v_nhieu.get(), cho_cham=self.v_cham_nhau.get())
        self.lbl_trang_thai.config(text=f"Ảnh mô phỏng  ·  seed {self.seed}")
        self.xu_ly_anh_tinh()

    # Kéo thanh trượt sáng / nhiễu / số quả
    # Vẽ lại CÙNG seed | hẹn 60 ms để kéo nhanh không vẽ lại liên tục
    def khi_doi_canh(self):
        """Kéo thanh trượt độ sáng / nhiễu / số quả -> vẽ lại CÙNG cảnh (cùng seed) để so sánh.
        Hẹn 60 ms sau mới vẽ để khi kéo nhanh không phải vẽ lại liên tục."""
        if self.nguon != "anh" or self.dap_an is None:
            return
        if self.hen_gio:
            self.root.after_cancel(self.hen_gio)
        self.hen_gio = self.root.after(60, self.ve_canh_mo_phong)

    def khi_doi_nguong(self):
        """Kéo ngưỡng S, V -> xử lý lại ảnh tĩnh (băng tải / camera tự đọc ngưỡng mới mỗi khung)."""
        if not self.dang_chay and self.anh_goc is not None:
            if self.hen_gio:
                self.root.after_cancel(self.hen_gio)
            self.hen_gio = self.root.after(40, self.xu_ly_anh_tinh)

    # Mở ảnh từ máy
    # imdecode: đọc được đường dẫn có dấu | ảnh > 1200 px thì thu nhỏ
    def mo_anh(self):
        self.ngat_camera()
        duong_dan = filedialog.askopenfilename(filetypes=[("Ảnh", "*.png *.jpg *.jpeg *.bmp")])
        if not duong_dan:
            return
        anh = cv2.imdecode(np.fromfile(duong_dan, np.uint8), cv2.IMREAD_COLOR)   # đọc được đường dẫn có dấu
        if anh is None:
            messagebox.showerror("Lỗi", "Không đọc được file ảnh.")
            return
        if anh.shape[1] > 1200:                                   # ảnh điện thoại rất lớn -> thu nhỏ
            ti_le = 1200 / anh.shape[1]
            anh = cv2.resize(anh, None, fx=ti_le, fy=ti_le, interpolation=cv2.INTER_AREA)
        self.dang_chay = False
        self.nguon = "anh"
        self.anh_goc, self.dap_an = anh, None
        self.lbl_trang_thai.config(text=os.path.basename(duong_dan))
        self.xu_ly_anh_tinh()

    # Xử lý 1 ảnh tĩnh
    # phan_tich_anh -> k nhập tay nếu không có vật chuẩn -> bảng 3 x 3 -> vẽ kết quả
    # Ảnh mô phỏng: so với đáp án (đúng độ chín, đúng cỡ, sai số mm)
    def xu_ly_anh_tinh(self):
        bat_dau = time.perf_counter()
        kq = phan_tich_anh(self.anh_goc, *self.lay_nguong())       # <<< GỌI PHẦN 1 + PHẦN 2
        if kq["k"] is None and self.lay_k_tay():                   # không có vật chuẩn -> dùng k nhập tay
            cap_nhat_kich_thuoc(kq, self.lay_k_tay())
        self.fps = 1 / max(time.perf_counter() - bat_dau, 1e-6)

        self.cap_nhat_bang(dem_anh_tinh(kq["qua"]), tong=len(kq["qua"]))
        self.lbl_che_do.config(text="Chế độ: đếm trên ảnh tĩnh")
        self.hien_ket_qua(self.anh_goc, kq, ve_ket_qua(self.anh_goc, kq))
        self.hien_thong_tin(kq)

        if self.dap_an is None:
            self.lbl_danh_gia.config(text="Ảnh từ máy: không có đáp án.", fg=CHU_MO)
            return
        dg = danh_gia_anh_tinh(kq["qua"], self.dap_an)
        n = dg["so_qua_that"]
        dau = "✔" if dg["so_qua_dem"] == n else "✘"
        if dg["sai_so_mm"]:
            sai_so = f"{np.mean(dg['sai_so_mm']):.2f} mm (lớn nhất {max(dg['sai_so_mm']):.2f})"
        else:
            sai_so = "—"
        self.lbl_danh_gia.config(fg=CHU, text=f"Đáp án: {n} quả   ·   Đếm được: {dg['so_qua_dem']}  {dau}\n"
                                              f"Đúng độ chín: {dg['dung_do_chin']}/{n}\n"
                                              f"Đúng cỡ S/M/L: {dg['dung_co']}/{n}\n"
                                              f"Sai số đường kính TB: {sai_so}")

    # Nguồn 2 - Băng tải mô phỏng
    # B: chạy | Space: tạm dừng / tiếp tục
    def chay_bang_tai(self):
        self.ngat_camera()
        self.bang_tai = tao_bang_tai(so_qua=int(self.v_so_qua.get()) + 6, seed=int(time.time()) % 100000)
        self.bo_dem = tao_bo_dem()
        self.nguon = "bang_tai"
        self.dap_an = None
        self.dang_chay = True
        self.moc_thoi_gian = time.perf_counter()
        self.lbl_trang_thai.config(text="Băng tải mô phỏng  ·  quả chạy từ trái sang phải")
        self.lbl_che_do.config(text="Chế độ: đếm qua vạch (mỗi ID đếm 1 lần)")
        self.lbl_danh_gia.config(text="Đang chạy...", fg=CHU_MO)
        self.vong_lap_bang_tai()

    # Vòng lặp video (mỗi lần gọi = 1 khung)
    # Vẽ khung -> phan_tich_anh -> cap_nhat_bo_dem (đếm qua vạch) -> vẽ -> băng tải tiến
    # root.after: hẹn gọi lại ~30 khung/s | hết băng tải: so với đáp án
    def vong_lap_bang_tai(self):
        """Xử lý 1 khung hình rồi hẹn giờ gọi lại chính nó -> thành vòng lặp video."""
        if not self.dang_chay or self.nguon != "bang_tai":
            return
        bt = self.bang_tai
        if bang_tai_da_xong(bt):                                # hết quả -> so với đáp án
            self.dang_chay = False
            dg = danh_gia_bang_tai(self.bo_dem["bang"], bt["dap_an"])
            dau = "✔" if dg["so_qua_dem"] == dg["so_qua_that"] else "✘"
            self.lbl_danh_gia.config(fg=CHU, text=f"Đáp án: {dg['so_qua_that']} quả   ·   Đếm được: "
                                                  f"{dg['so_qua_dem']}  {dau}\n"
                                                  f"Đúng loại (độ chín × cỡ): {dg['dung_loai']}/{dg['so_qua_that']}")
            return
        bat_dau = time.perf_counter()

        anh = khung_hinh_bang_tai(bt, self.v_do_sang.get() / 100, self.v_nhieu.get())
        kq = phan_tich_anh(anh, *self.lay_nguong())                              # <<< PHẦN 1 + PHẦN 2
        x_vach = bt["rong"] / 2
        ids, vua_dem = cap_nhat_bo_dem(self.bo_dem, kq["qua"], x_vach, kq["k"])  # <<< ĐẾM (PHẦN 2)
        self.nhay_vach = 4 if vua_dem else max(self.nhay_vach - 1, 0)
        self.hien_ket_qua(anh, kq, ve_ket_qua(anh, kq, ids, x_vach, self.nhay_vach > 0))
        self.cap_nhat_bang(self.bo_dem["bang"])
        bt["quang_duong"] += self.v_toc_do.get()                                 # băng tải chạy tiếp

        self.do_fps()
        tien_do = min(100, 100 * bt["quang_duong"] / bt["quang_duong_het"])
        self.hien_thong_tin(kq, f"\nTiến độ: {tien_do:.0f}%")
        cho = max(1, int(1000 / 30 - (time.perf_counter() - bat_dau) * 1000))   # giữ khoảng 30 khung/s
        self.root.after(cho, self.vong_lap_bang_tai)

    # Nguồn 3 - Camera điện thoại
    # CameraLuong (luồng riêng) | cho_camera: kiểm tra mỗi 150 ms, tối đa 20 giây
    def ket_noi_camera(self):
        self.ngat_camera()
        self.dang_chay = False
        self.camera = CameraLuong(self.v_dia_chi.get())
        self.luc_bat_dau_ket_noi = time.time()
        self.lbl_trang_thai.config(text=f"Đang kết nối {self.ten_nguon_camera()} ...")
        self.cho_camera()

    def cho_camera(self):
        """Kiểm tra mỗi 150 ms xem camera kết nối xong chưa (tối đa 20 giây)."""
        if self.camera is None:
            return
        if self.camera.trang_thai == "dang_mo" and time.time() - self.luc_bat_dau_ket_noi < 20:
            self.root.after(150, self.cho_camera)
            return
        if self.camera.trang_thai != "ok":
            self.ngat_camera()
            messagebox.showerror("Không kết nối được camera",
                                 "Kiểm tra:\n• Điện thoại và máy tính chung 1 mạng Wi-Fi\n"
                                 "• App DroidCam / IP Webcam đang mở và đã bấm Start\n"
                                 "• Địa chỉ đúng dạng http://IP:4747/video (DroidCam)\n"
                                 "  hoặc http://IP:8080/video (IP Webcam)\n"
                                 "• Thử mở địa chỉ đó bằng trình duyệt trên máy tính")
            return
        self.lich_su_k = []
        self.bo_dem = tao_bo_dem()
        self.nguon = "camera"
        self.dap_an = None
        self.dang_chay = True
        self.moc_thoi_gian = time.perf_counter()
        self.lbl_trang_thai.config(text=f"Camera: {self.ten_nguon_camera()}")
        self.lbl_danh_gia.config(text="Thực nghiệm: không có đáp án mô phỏng.\n"
                                      "Đối chiếu bằng cách đếm và đo thước trực tiếp.", fg=CHU_MO)
        self.vong_lap_camera()

    def ten_nguon_camera(self):
        n = self.camera.nguon
        return os.path.basename(n) if os.path.isfile(n) else n[:45]

    def ngat_camera(self):
        if self.camera is not None:
            self.camera.dong()
            self.camera = None
            self.dang_chay = False
            self.lbl_trang_thai.config(text="Đã ngắt camera")

    # Vòng lặp camera
    # Thu nhỏ rộng 960 px -> phan_tich_anh -> k = trung vị 15 khung gần nhất
    # Đếm tĩnh: các quả đang trong khung | đếm qua vạch: theo dõi ID
    def vong_lap_camera(self):
        if not self.dang_chay or self.camera is None:
            return
        bat_dau = time.perf_counter()
        anh = self.camera.doc()
        if anh is None:                                     # chưa có khung hình -> chờ 30 ms
            self.root.after(30, self.vong_lap_camera)
            return
        if anh.shape[1] > 960:                              # thu nhỏ còn rộng 960 để xử lý nhanh
            ti_le = 960 / anh.shape[1]
            anh = cv2.resize(anh, None, fx=ti_le, fy=ti_le, interpolation=cv2.INTER_AREA)

        kq = phan_tich_anh(anh, *self.lay_nguong())         # <<< PHẦN 1 + PHẦN 2
        # k mỗi khung dao động nhẹ -> lấy TRUNG VỊ 15 khung gần nhất cho ổn định.
        # Khung nào tay che mất vật chuẩn thì vẫn dùng k của các khung trước.
        if kq["k"]:
            self.lich_su_k = (self.lich_su_k + [kq["k"]])[-15:]
        cap_nhat_kich_thuoc(kq, np.median(self.lich_su_k) if self.lich_su_k else self.lay_k_tay())

        if self.v_kieu_dem.get() == "vach":                # đẩy quả đi qua vạch
            x_vach = anh.shape[1] * self.v_vi_tri_vach.get() / 100
            ids, vua_dem = cap_nhat_bo_dem(self.bo_dem, kq["qua"], x_vach, kq["k"])
            self.nhay_vach = 4 if vua_dem else max(self.nhay_vach - 1, 0)
            anh_kq = ve_ket_qua(anh, kq, ids, x_vach, self.nhay_vach > 0)
            self.cap_nhat_bang(self.bo_dem["bang"])
            self.lbl_che_do.config(text="Camera: đếm qua vạch (mỗi ID đếm 1 lần)")
        else:                                               # đếm các quả đang nằm trong khung
            anh_kq = ve_ket_qua(anh, kq)
            self.cap_nhat_bang(dem_anh_tinh(kq["qua"]), tong=len(kq["qua"]))
            self.lbl_che_do.config(text="Camera: đếm các quả đang trong khung hình")
        self.anh_goc = anh
        self.hien_ket_qua(anh, kq, anh_kq)

        self.do_fps()
        self.hien_thong_tin(kq, "" if kq["vat_chuan"] else "\n⚠ Khung này không thấy vật chuẩn")
        cho = max(1, int(1000 / 30 - (time.perf_counter() - bat_dau) * 1000))
        self.root.after(cho, self.vong_lap_camera)

    # Space: dừng / chạy tiếp
    def tam_dung(self):
        """Space: dừng / chạy tiếp băng tải hoặc camera."""
        if self.nguon == "camera" and self.camera is not None:
            self.dang_chay = not self.dang_chay
            if self.dang_chay:
                self.moc_thoi_gian = time.perf_counter()
                self.vong_lap_camera()
        elif self.nguon == "bang_tai" and not bang_tai_da_xong(self.bang_tai):
            self.dang_chay = not self.dang_chay
            if self.dang_chay:
                self.moc_thoi_gian = time.perf_counter()
                self.vong_lap_bang_tai()

    # Đặt lại
    # Bộ đếm mới, bảng về 0
    def dat_lai(self):
        self.bo_dem = tao_bo_dem()
        self.cap_nhat_bang(np.zeros((3, 3), int))

    # Lưu CSV
    # Thêm 1 dòng vào thong_ke.csv: thời gian, nguồn, 9 ô bảng, tổng
    def luu_csv(self):
        """Thêm 1 dòng (thời gian, nguồn, 9 ô của bảng, tổng) vào thong_ke.csv."""
        duong_dan = os.path.join(THU_MUC, "thong_ke.csv")
        file_moi = not os.path.exists(duong_dan)
        with open(duong_dan, "a", newline="", encoding="utf-8-sig") as f:
            ghi = csv.writer(f)
            if file_moi:
                ghi.writerow(["thoi_gian", "nguon"] + [f"{TEN_VN[d]}_{c}" for d in (2, 1, 0) for c in TEN_CO]
                             + ["tong"])
            o_bang = [int(self.bang_dem[d, c]) for d in (2, 1, 0) for c in range(3)]
            ghi.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), self.nguon] + o_bang + [int(self.bang_dem.sum())])
        messagebox.showinfo("Đã lưu", f"Đã thêm 1 dòng vào\n{duong_dan}")

    def luu_anh(self):
        if self.ket_qua_cuoi is None:
            return
        duong_dan = os.path.join(THU_MUC, time.strftime("ket_qua_%Y%m%d_%H%M%S.png"))
        cv2.imencode(".png", self.ket_qua_cuoi[2])[1].tofile(duong_dan)   # tofile: lưu được đường dẫn có dấu
        messagebox.showinfo("Đã lưu", duong_dan)

    def thoat(self):
        self.ngat_camera()
        self.root.destroy()


# Điểm vào chương trình
# python main.py  ->  màn hình đầu (man_hinh_dau.py)  ->  bấm START  ->  giao diện chính
def main():
    try:    # chữ sắc nét trên màn hình Windows có phóng to 125%, 150%
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    # Màn hình đầu tiên (logo, thông tin đề tài, nút START);
    # Bấm START mới mở giao diện chương trình chính UngDung.
    ManHinhDau(root, lambda: UngDung(root))
    root.mainloop()


if __name__ == "__main__":
    main()
