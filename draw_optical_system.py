"""
Sơ đồ hệ quang học kiểm tra PCB/PCBA (bản nháp, Nhiệm vụ B - Giai đoạn 1)
(a) Trường sáng ĐỒNG TRỤC  (b) Trường tối góc thấp  (c) Bảng gán chế độ chiếu sáng theo loại lỗi
Chạy:  python draw_optical_system.py   ->  optical_system_draft.png / .svg
Mọi tọa độ trong mỗi panel nằm trong khung 0..100 để dễ chỉnh sửa.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch, Ellipse

ORANGE, BLUE, GRAY, GREEN, GOLD = "#e67e22", "#2471a3", "#7f8c8d", "#1e8449", "#d4ac0d"


def arrow(ax, p, q, color, lw=1.8, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, color=color, lw=lw, ls=ls))


def lens(ax, cx, cy, w=18, h=5, label=None, side="r"):
    ax.add_patch(Ellipse((cx, cy), w, h, fc="#d6eaf8", ec="#1b4f72", lw=1.5, zorder=3))
    if label:
        ax.text(cx + (w / 2 + 2 if side == "r" else -w / 2 - 2), cy, label, fontsize=8.5,
                ha="left" if side == "r" else "right", va="center")


def camera(ax, label):
    ax.add_patch(Rectangle((40, 90), 20, 8, fc="#566573", ec="black", zorder=3))
    ax.text(50, 94, "Camera", color="white", ha="center", va="center", fontsize=9, weight="bold", zorder=4)
    ax.text(62, 94, label, fontsize=8.5, va="center")
    lens(ax, 50, 82, 16, 4.5, "Ống kính tele-centric / tube lens")


def pcb(ax, defect=True):
    ax.add_patch(Rectangle((12, 14), 76, 8, fc="#1d6f42", ec="black", zorder=2))      # nền FR4 + solder mask
    for x0, x1 in ((16, 34), (66, 84)):                                                 # đường mạch đồng
        ax.add_patch(Rectangle((x0, 22), x1 - x0, 2.2, fc=GOLD, ec="#7d6608", zorder=3))
    ax.add_patch(Rectangle((40, 22), 20, 2.2, fc=GOLD, ec="#7d6608", zorder=3))
    if defect:                                                                          # gai đồng (spur)
        ax.add_patch(Polygon([[48, 24.2], [50, 29.5], [52, 24.2]], fc=GOLD, ec="#7d6608", zorder=4))
    ax.text(50, 17.5, "PCB (nền FR4 + đồng)", ha="center", color="white", fontsize=8.5, zorder=5)


def panel_a(ax):
    ax.set_title("(a) Trường sáng đồng trục (coaxial bright-field)", fontsize=11, weight="bold")
    camera(ax, "(area-scan hoặc line-scan)")
    ax.plot([50, 50], [88, 30], ls=":", color=GRAY, lw=1)
    # nguồn LED + bộ tách chùm 45 độ
    ax.add_patch(Rectangle((2, 57), 17, 10, fc="#f9e79f", ec="black", zorder=3))
    ax.text(10.5, 62, "LED\n+ khuếch tán", ha="center", va="center", fontsize=8, zorder=4)
    arrow(ax, (19, 62), (44, 62), ORANGE)
    ax.add_patch(Polygon([[44, 56], [56, 56], [56, 68], [44, 68]], fc="#aed6f1", ec="black", alpha=.55, zorder=2))
    ax.plot([44, 56], [56, 68], color="black", lw=2, zorder=3)
    ax.text(58, 68, "Bộ tách chùm 50/50 (45°)", fontsize=8.5)
    # vật kính
    lens(ax, 50, 46, 18, 5, "Vật kính", "r")
    # tia tới (cam) và tia phản xạ (xanh)
    for x in (44, 56):
        arrow(ax, (x, 53), (x, 25.5), ORANGE)
    for x in (47.5, 52.5):
        arrow(ax, (x, 25.5), (x, 76), BLUE)
    pcb(ax)
    ax.text(50, 6, "Ánh sáng tới vuông góc, phản xạ gương từ đồng → vào vật kính\n"
                    "Đồng/mặt phẳng: SÁNG;  nền, khe hở, lỗ, mép lệch: TỐI", ha="center", fontsize=8.5)
    ax.plot([2, 8], [80, 80], color=ORANGE, lw=2); ax.text(10, 80, "tia tới", color=ORANGE, fontsize=8.5, weight="bold", va="center")
    ax.plot([2, 8], [75, 75], color=BLUE, lw=2); ax.text(10, 75, "tia phản xạ / tán xạ", color=BLUE, fontsize=8.5, weight="bold", va="center")


def panel_b(ax):
    ax.set_title("(b) Trường tối góc thấp (dark-field, đèn vòng)", fontsize=11, weight="bold")
    camera(ax, "(cùng bộ thu nhận)")
    ax.plot([50, 50], [88, 30], ls=":", color=GRAY, lw=1)
    lens(ax, 50, 46, 18, 5, "Vật kính", "l")
    # đèn vòng (mặt cắt hai bên)
    for x0, sgn in ((4, 1), (80, -1)):
        ax.add_patch(Rectangle((x0, 36), 16, 6, fc="#f9e79f", ec="black", zorder=3))
    ax.text(12, 44, "Đèn vòng LED\n(góc chiếu thấp)", ha="center", fontsize=8.5)
    ax.text(88, 44, "Đèn vòng LED\n(đối xứng)", ha="center", fontsize=8.5)
    # tia tới nghiêng, phản xạ gương đi ra ngoài (không vào vật kính)
    arrow(ax, (16, 38), (36, 25), ORANGE)
    arrow(ax, (36, 25), (58, 38), GRAY, ls="--")
    arrow(ax, (84, 38), (64, 25), ORANGE)
    arrow(ax, (64, 25), (42, 38), GRAY, ls="--")
    ax.text(2, 30, "phản xạ gương\nlệch khỏi vật kính", color=GRAY, fontsize=8, ha="left")
    # tán xạ từ gai đồng đi vào vật kính
    for x in (46, 50, 54):
        arrow(ax, (50, 28.5), (x, 43), BLUE)
    arrow(ax, (47.5, 49), (47.5, 76), BLUE)
    arrow(ax, (52.5, 49), (52.5, 76), BLUE)
    pcb(ax)
    ax.text(50, 6, "Mặt phẳng phản xạ gương đi chỗ khác → nền TỐI\n"
                    "Cạnh, gai, xước, dị vật tán xạ vào vật kính → nổi SÁNG", ha="center", fontsize=8.5)
    ax.annotate("gai đồng (spur)\ntán xạ mạnh", (51, 28.5), (64, 55), fontsize=8.5, color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=.8))


TABLE = [
    ("Loại lỗi (DsPCBSD+ / đề bài)", "Chế độ ưu tiên", "Lý do vật lý (giả thuyết thiết kế, cần kiểm chứng)"),
    ("Đứt mạch (OP), ngắn mạch (SH)", "Trường sáng đồng trục", "Đồng phản xạ mạnh, nền tối → hình dạng mạch rõ; mất/thừa đường dẫn lộ ngay"),
    ("Thiếu lỗ / vỡ viền lỗ (HB)", "Trường sáng đồng trục", "Lỗ và pad có tương phản sáng-tối ổn định với nền"),
    ("Gai đồng (SP), khuyết cạnh (MB)", "Trường tối", "Cạnh gồ ghề tán xạ mạnh, nổi bật trên nền tối; kết hợp trường sáng để định vị"),
    ("Đồng thừa (SC), xước dây dẫn (CS), dị vật (CFO/BMFO)", "Trường tối (+ trường sáng)", "Bề mặt không phẳng/vật lạ tán xạ; trường sáng cho vị trí so với mạch"),
    ("PCBA: thiếu/lệch linh kiện", "Trường sáng đồng trục", "So sánh với ảnh mẫu chuẩn; trường tối làm nổi mối hàn và mép linh kiện"),
]


def table(ax):
    import textwrap

    ax.axis("off")
    cols, wraps = [0.0, 0.31, 0.50], (36, 24, 70)
    ax.text(0, 0.97, "(c) Gán chế độ chiếu sáng theo loại lỗi", fontsize=11, weight="bold", transform=ax.transAxes, va="top")
    ys = [0.86] + [0.74 - 0.135 * k for k in range(len(TABLE) - 1)]
    for r, (row, y) in enumerate(zip(TABLE, ys)):
        for x, txt, wr in zip(cols, row, wraps):
            ax.text(x, y, "\n".join(textwrap.wrap(txt, wr)), fontsize=8.5, va="top",
                    weight="bold" if r == 0 else "normal", transform=ax.transAxes)
    ax.plot([0, 1], [0.775, 0.775], color="black", lw=.8, transform=ax.transAxes)
    ax.text(0.5, 0.0, "Có thể tích hợp 2 chế độ vào cùng một đầu đo: chụp 2 ảnh liên tiếp (bật lần lượt LED đồng trục / đèn vòng) "
            "rồi dùng riêng hoặc ghép kênh làm đầu vào cho mô hình.", ha="center", fontsize=9, style="italic",
            transform=ax.transAxes)


def main(out="optical_system_draft"):
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.7], hspace=0.06, wspace=0.08)
    for k, fn in enumerate((panel_a, panel_b)):
        ax = fig.add_subplot(gs[0, k])
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal"); ax.axis("off")
        fn(ax)
    table(fig.add_subplot(gs[1, :]))
    fig.suptitle("Hệ quang học đề xuất cho kiểm tra khuyết tật PCB/PCBA  →  Camera → AI (YOLO) → Bounding Box",
                 fontsize=13, weight="bold", y=0.93)
    fig.savefig(out + ".png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(out + ".svg", bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
