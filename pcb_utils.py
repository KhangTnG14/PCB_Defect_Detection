"""
pcb_utils.py  -  Hàm dùng chung cho dự án PCB/PCBA (Object Detection, YOLO/Ultralytics)
Nhiệm vụ B - Giai đoạn 1.

Nội dung
    1. Cấu hình, seed, logging, data.yaml
    2. Kiểm tra bộ dữ liệu YOLO (nhãn lỗi, rò rỉ split)
    3. Huấn luyện có checkpoint/resume + ghi log thời gian từng epoch
    4. Đánh giá val/test (P, R, F1, mAP, AP từng lớp, FPS)
    5. Thu thập dự đoán -> PR theo lớp, ROC mức phát hiện, FROC, ma trận nhầm lẫn
    6. Hàm vẽ: đường cong train, PR, ROC, confusion matrix, so sánh mô hình, ảnh dự đoán
    7. Tiện ích chuyển nhãn DeepPCB -> YOLO (chỉ dùng nếu quyết định gộp/kiểm tra chéo)

Module KHÔNG import ultralytics/torch ở đầu file, nên các hàm vẽ/thống kê chạy được ở bất kỳ máy nào.
Nhãn biểu đồ mặc định bằng tiếng Anh (đồng nhất với Ultralytics); đổi qua tham số title.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# ----------------------------------------------------------------------------
# 1. Cấu hình, seed, logging, data.yaml
# ----------------------------------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    """Cố định seed cho random / numpy / torch (nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def setup_logger(log_path: str | Path, name: str = "pcb") -> logging.Logger:
    """Logger ghi đồng thời ra màn hình và file."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def disable_trackers() -> None:
    """Tắt wandb/clearml/comet... để Ultralytics không hỏi đăng nhập trên Kaggle."""
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    try:
        from ultralytics import settings

        for k in ("wandb", "clearml", "comet", "dvc", "mlflow", "neptune", "raytune"):
            try:
                settings.update({k: False})
            except Exception:
                pass
    except ImportError:
        pass


def load_data_yaml(path: str | Path):
    """Đọc data.yaml -> (cfg, names:list[str], root:Path)."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    root = Path(cfg.get("path", path.parent))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    return cfg, list(names), root


def make_working_yaml(src_yaml: str | Path, dst_yaml: str | Path = "/kaggle/working/data_work.yaml",
                      root: str | Path | None = None) -> str:
    """
    Dataset Kaggle nằm ở /kaggle/input (chỉ đọc) và `path` trong data.yaml của A có thể là
    đường dẫn máy cá nhân. Hàm này tạo bản sao data.yaml với `path` trỏ đúng thư mục dataset.
    """
    src_yaml = Path(src_yaml)
    cfg = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
    cfg["path"] = str(root or src_yaml.parent)
    Path(dst_yaml).parent.mkdir(parents=True, exist_ok=True)
    Path(dst_yaml).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return str(dst_yaml)


# ----------------------------------------------------------------------------
# 2. Kiểm tra bộ dữ liệu YOLO
# ----------------------------------------------------------------------------
def label_path_for(img_path: str | Path) -> str:
    """Quy ước của Ultralytics: .../images/... -> .../labels/....txt"""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    s = str(img_path)
    s = sb.join(s.rsplit(sa, 1)) if sa in s else s
    return os.path.splitext(s)[0] + ".txt"


def list_images(root: Path, entry) -> list[str]:
    """entry trong data.yaml: thư mục, file .txt liệt kê ảnh, hoặc list của các mục đó."""
    entries = entry if isinstance(entry, (list, tuple)) else [entry]
    out: list[str] = []
    for e in entries:
        p = Path(e)
        p = p if p.is_absolute() else root / p
        if p.is_dir():
            out += [str(f) for f in sorted(p.rglob("*")) if f.suffix.lower() in IMG_EXT]
        elif p.is_file() and p.suffix == ".txt":
            for ln in p.read_text().splitlines():
                ln = ln.strip()
                if ln:
                    q = Path(ln)
                    out.append(str(q if q.is_absolute() else root / q))
    return out


def check_dataset(data_yaml: str | Path, splits=("train", "val", "test"), max_issues: int = 20):
    """
    Kiểm tra nhanh nhãn YOLO. Trả về (df_split, df_class, issues, leaks).
      df_split : số ảnh, số bbox, file nhãn thiếu/rỗng, dòng nhãn lỗi theo từng split
      df_class : số bbox theo lớp x split
      issues   : mẫu các dòng nhãn lỗi (class id sai, tọa độ ngoài [0,1], sai số cột, ...)
      leaks    : số tên ảnh trùng giữa các split (rò rỉ dữ liệu)
    """
    cfg, names, root = load_data_yaml(data_yaml)
    nc = len(names)
    rows, cls_cols, issues, stems = [], {}, [], {}
    for sp in splits:
        if not cfg.get(sp):
            continue
        imgs = list_images(root, cfg[sp])
        stems[sp] = {Path(i).stem for i in imgs}
        n_missing = n_empty = n_bad = n_boxes = 0
        counts = np.zeros(nc, dtype=int)
        for ip in imgs:
            lp = label_path_for(ip)
            if not os.path.exists(lp):
                n_missing += 1
                continue
            lines = [l for l in Path(lp).read_text().splitlines() if l.strip()]
            if not lines:
                n_empty += 1
                continue
            for ln in lines:
                parts = ln.split()
                ok = len(parts) == 5
                if ok:
                    try:
                        c = int(float(parts[0]))
                        x, y, w, h = map(float, parts[1:])
                        ok = 0 <= c < nc and 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
                    except ValueError:
                        ok = False
                if not ok:
                    n_bad += 1
                    if len(issues) < max_issues:
                        issues.append(f"[{sp}] {lp}: '{ln[:70]}'")
                    continue
                counts[c] += 1
                n_boxes += 1
        rows.append(dict(split=sp, images=len(imgs), boxes=n_boxes, missing_label_files=n_missing,
                         empty_label_files=n_empty, invalid_lines=n_bad))
        cls_cols[sp] = counts
    df_split = pd.DataFrame(rows).set_index("split")
    df_class = pd.DataFrame(cls_cols, index=names)
    df_class["total"] = df_class.sum(axis=1)
    keys = list(stems)
    leaks = {f"{a}&{b}": len(stems[a] & stems[b]) for i, a in enumerate(keys) for b in keys[i + 1:]}
    return df_split, df_class, issues, leaks


# ----------------------------------------------------------------------------
# 3. Huấn luyện có checkpoint / resume + log thời gian epoch
# ----------------------------------------------------------------------------
def load_model(weights, family: str = "auto"):
    """family: 'yolo' | 'rtdetr' | 'auto' (đoán theo tên file/đường dẫn: có 'rtdetr' -> RT-DETR)."""
    from ultralytics import RTDETR, YOLO

    if family == "auto":
        family = "rtdetr" if "rtdetr" in str(weights).lower() else "yolo"
    return RTDETR(str(weights)) if family == "rtdetr" else YOLO(str(weights))


def make_epoch_logger(csv_path: str | Path, logger: logging.Logger | None = None):
    """Callback ghi thời gian từng epoch + bộ nhớ GPU. Đăng ký: model.add_callback('on_fit_epoch_end', fn)."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"t_last": time.time(), "t0": time.time()}

    def _cb(trainer):
        if trainer.epoch >= trainer.epochs:   # bỏ qua lần gọi sau validate cuối
            return
        now = time.time()
        dt = now - state["t_last"]
        
        state["t_last"] = now
        mem = 0.0
        try:
            import torch

            if torch.cuda.is_available():
                mem = torch.cuda.max_memory_allocated() / 1e9
        except ImportError:
            pass
        m = getattr(trainer, "metrics", {}) or {}
        row = dict(epoch=int(trainer.epoch) + 1, epoch_time_s=round(dt, 1),
                   elapsed_min=round((now - state["t0"]) / 60, 2), gpu_mem_gb=round(mem, 2),
                   mAP50=round(float(m.get("metrics/mAP50(B)", float("nan"))), 5),
                   mAP50_95=round(float(m.get("metrics/mAP50-95(B)", float("nan"))), 5))
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        if logger:
            logger.info("epoch %(epoch)d | %(epoch_time_s).1fs | mAP50=%(mAP50).4f | mAP50-95=%(mAP50_95).4f | gpu=%(gpu_mem_gb).1fGB", row)

    return _cb


def train_model(model_name: str, data_yaml: str, project: str, name: str, imgsz: int = 640, epochs: int = 20,
                batch: int = 16, seed: int = 42, workers: int = 2, device=0, patience: int = 100,
                fraction: float = 1.0, save_period: int = 5, resume: bool = True, family: str = "auto",
                logger: logging.Logger | None = None, extra: dict | None = None):
    """
    Huấn luyện 1 mô hình. Tự động resume nếu đã có last.pt dở dang trong project/name;
    bỏ qua train nếu đã chạy đủ epoch. Trả về (best_pt, save_dir).
    Mọi baseline phải dùng cùng imgsz/epochs/batch/seed (cấu hình chung do A chốt).
    """
    save_dir = Path(project) / name
    last, best = save_dir / "weights" / "last.pt", save_dir / "weights" / "best.pt"
    res_csv = save_dir / "results.csv"
    done = res_csv.exists() and len(pd.read_csv(res_csv)) >= epochs and best.exists()
    if done:
        (logger.info if logger else print)(f"Run '{name}' đã đủ {epochs} epoch -> bỏ qua train.")
        return str(best), save_dir

    kw = dict(data=str(data_yaml), imgsz=imgsz, epochs=epochs, batch=batch, seed=seed, deterministic=True,
              workers=workers, device=device, patience=patience, fraction=fraction, save_period=save_period,
              project=str(project), name=name, exist_ok=True, plots=True)
    kw.update(extra or {})   # extra có thể ghi đè bất kỳ tham số nào ở trên (vd deterministic=False nếu CUDA báo lỗi)
    if resume and last.exists():
        (logger.info if logger else print)(f"Tìm thấy {last} -> resume.")
        model = load_model(last, family)
        model.add_callback("on_fit_epoch_end", make_epoch_logger(save_dir / "epoch_time.csv", logger))
        model.train(resume=True)
    else:
        model = load_model(model_name, family)
        model.add_callback("on_fit_epoch_end", make_epoch_logger(save_dir / "epoch_time.csv", logger))
        model.train(**kw)
    save_dir = Path(getattr(model.trainer, "save_dir", save_dir))
    return str(save_dir / "weights" / "best.pt"), save_dir


# ----------------------------------------------------------------------------
# 4. Đánh giá val/test, FPS
# ----------------------------------------------------------------------------
def count_params(model):
    """(params triệu, GFLOPs) - GFLOPs có thể None tùy phiên bản."""
    p = sum(x.numel() for x in model.model.parameters()) / 1e6
    try:
        g = float(model.info(verbose=False)[3])
    except Exception:
        g = None
    return round(p, 2), g


def measure_fps(model, img_paths, imgsz=640, n=100, warmup=10, device=None) -> float:
    """FPS đo thật (batch=1, gồm tiền xử lý + suy luận + NMS) trên n ảnh."""
    paths = list(img_paths)[: n + warmup]
    for p in paths[:warmup]:
        model.predict(p, imgsz=imgsz, verbose=False, device=device)
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass
    t = time.time()
    for p in paths[warmup:]:
        model.predict(p, imgsz=imgsz, verbose=False, device=device)
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass
    return round(len(paths[warmup:]) / (time.time() - t), 2)


def evaluate_split(weights, data_yaml, split="test", imgsz=640, batch=16, conf=0.001, iou=0.6, device=None,
                   project="runs/eval", name=None, family="auto", plots=True) -> dict:
    """
    Chạy model.val trên split (val/test) -> dict chỉ số. Chọn mô hình bằng 'val', 'test' chỉ để báo cáo.
    Đầu ra dùng được cho make_summary_table / plot_per_class_ap.
    """
    model = load_model(weights, family)
    name = name or f"{Path(weights).parent.parent.name}_{split}"
    r = model.val(data=str(data_yaml), split=split, imgsz=imgsz, batch=batch, conf=conf, iou=iou, device=device,
                  plots=plots, project=str(project), name=name, exist_ok=True, verbose=False)
    b = r.box
    P, R = float(b.mp), float(b.mr)
    cls_names = r.names
    per_class = {cls_names[i]: dict(AP50=float("nan"), AP50_95=float("nan"), P=float("nan"), R=float("nan"))
                 for i in sorted(cls_names)}
    for j, c in enumerate(b.ap_class_index):
        per_class[cls_names[int(c)]] = dict(AP50=float(b.ap50[j]), AP50_95=float(b.ap[j]),
                                            P=float(b.p[j]), R=float(b.r[j]))
    sp = r.speed
    ms = sp.get("preprocess", 0) + sp.get("inference", 0) + sp.get("postprocess", 0)
    params, gflops = count_params(model)
    return {"Model": name, "split": split, "P": P, "R": R, "F1": 2 * P * R / (P + R + 1e-16),
            "mAP50": float(b.map50), "mAP50-95": float(b.map), "FPS_val": round(1000 / ms, 1) if ms else None,
            "Params(M)": params, "GFLOPs": gflops, "per_class": per_class, "weights": str(weights),
            "save_dir": str(getattr(r, "save_dir", ""))}


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


# ----------------------------------------------------------------------------
# 5. Thu thập dự đoán -> khớp hộp -> PR / ROC / FROC / confusion matrix
# ----------------------------------------------------------------------------
def read_yolo_labels(txt_path: str, w: int, h: int) -> np.ndarray:
    """Nhãn YOLO -> mảng (n,5): cls, x1, y1, x2, y2 (pixel)."""
    if not os.path.exists(txt_path):
        return np.zeros((0, 5))
    a = np.array([[float(v) for v in ln.split()[:5]] for ln in Path(txt_path).read_text().splitlines()
                  if len(ln.split()) >= 5], dtype=float).reshape(-1, 5)
    out = np.zeros_like(a)
    out[:, 0] = a[:, 0]
    out[:, 1] = (a[:, 1] - a[:, 3] / 2) * w
    out[:, 2] = (a[:, 2] - a[:, 4] / 2) * h
    out[:, 3] = (a[:, 1] + a[:, 3] / 2) * w
    out[:, 4] = (a[:, 2] + a[:, 4] / 2) * h
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU từng cặp, a:(n,4) b:(m,4) xyxy -> (n,m)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def collect_predictions(model, img_paths, imgsz=640, conf=0.001, iou=0.6, batch=16, device=None,
                        max_det=300) -> list[dict]:
    """
    Dự đoán trên danh sách ảnh với ngưỡng conf rất thấp (để vẽ được toàn bộ đường cong), kèm nhãn GT.
    Trả về list record: {path, pred:(n,6)[x1,y1,x2,y2,conf,cls], gt:(m,5)[cls,x1,y1,x2,y2]}.
    """
    img_paths = list(img_paths)
    records = []
    for i in range(0, len(img_paths), batch):
        chunk = img_paths[i:i + batch]
        results = model.predict(chunk, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=device,
                                verbose=False)
        for p, r in zip(chunk, results):
            h, w = r.orig_shape
            bx = r.boxes
            pred = np.concatenate([bx.xyxy.cpu().numpy(), bx.conf.cpu().numpy()[:, None],
                                   bx.cls.cpu().numpy()[:, None]], axis=1) if len(bx) else np.zeros((0, 6))
            records.append(dict(path=p, pred=pred, gt=read_yolo_labels(label_path_for(p), w, h)))
    return records


def _compute_ap(recall, precision):
    """AP nội suy 101 điểm (cùng cách tính với Ultralytics/COCO)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    px = np.interp(x, mrec, mpre)
    return float(_trapz(px, x)), x, px


def match_per_class(records, nc: int, iou_thr: float = 0.5) -> dict:
    """Khớp tham lam theo confidence giảm dần, từng lớp, từng ảnh. Trả {cls: scores, tp, n_gt}."""
    out = {c: dict(scores=[], tp=[], n_gt=0) for c in range(nc)}
    for rec in records:
        pred, gt = rec["pred"], rec["gt"]
        for c in range(nc):
            g = gt[gt[:, 0] == c][:, 1:]
            p = pred[pred[:, 5] == c]
            out[c]["n_gt"] += len(g)
            if len(p) == 0:
                continue
            p = p[np.argsort(-p[:, 4], kind="stable")]
            ious = box_iou(p[:, :4], g)
            used = np.zeros(len(g), dtype=bool)
            for i in range(len(p)):
                tp = False
                if len(g):
                    cand = np.where(used, -1.0, ious[i])
                    j = int(cand.argmax())
                    if cand[j] >= iou_thr:
                        used[j] = True
                        tp = True
                out[c]["scores"].append(float(p[i, 4]))
                out[c]["tp"].append(tp)
    return out


def build_curves(records, nc: int, iou_thr: float = 0.5) -> dict:
    """
    Tính PR (AP@iou_thr), ROC mức phát hiện và FROC cho từng lớp.

    LƯU Ý về ROC trong bài toán detection: không có 'true negative' theo nghĩa phân loại, nên ROC ở đây là
    ROC MỨC PHÁT HIỆN:  trục y = TP tích lũy / số GT của lớp (recall thật, đã tính cả lỗi bỏ sót);
                        trục x = FP tích lũy / tổng FP của lớp (khi quét ngưỡng confidence từ cao xuống thấp).
    AUC đo khả năng confidence xếp TP trước FP. Cần nêu rõ định nghĩa này trong báo cáo.
    FROC (recall theo số FP trên mỗi ảnh) là thước đo thay thế dễ diễn giải hơn trong kiểm tra công nghiệp.
    """
    matches = match_per_class(records, nc, iou_thr)
    n_img = max(len(records), 1)
    pr, roc, froc = {}, {}, {}
    for c in range(nc):
        n_gt = matches[c]["n_gt"]
        if n_gt == 0:
            continue
        s = np.array(matches[c]["scores"]); tp = np.array(matches[c]["tp"], dtype=bool)
        if len(s) == 0:
            x = np.linspace(0, 1, 101)
            pr[c] = dict(x=x, precision=np.zeros(101), ap=0.0, n_gt=n_gt)
            roc[c] = dict(fpr=np.array([0, 1.0]), tpr=np.array([0, 0.0]), auc=0.0)
            froc[c] = dict(fp_per_img=np.array([0.0]), recall=np.array([0.0]))
            continue
        order = np.argsort(-s, kind="stable")
        tp = tp[order]
        tpc, fpc = np.cumsum(tp), np.cumsum(~tp)
        rec, prec = tpc / n_gt, tpc / (tpc + fpc)
        ap, x, px = _compute_ap(rec, prec)
        pr[c] = dict(x=x, precision=px, ap=ap, n_gt=n_gt)
        fpr = fpc / max(fpc[-1], 1)
        fpr_f = np.concatenate(([0.0], fpr, [1.0])); tpr_f = np.concatenate(([0.0], rec, [rec[-1]]))
        roc[c] = dict(fpr=fpr_f, tpr=tpr_f, auc=float(_trapz(tpr_f, fpr_f)))
        froc[c] = dict(fp_per_img=np.concatenate(([0.0], fpc / n_img)), recall=np.concatenate(([0.0], rec)))
    return dict(pr=pr, roc=roc, froc=froc, n_images=n_img, iou_thr=iou_thr)


def confusion_matrix_det(records, nc: int, conf_thr: float = 0.25, iou_thr: float = 0.45) -> np.ndarray:
    """
    Ma trận nhầm lẫn (nc+1)x(nc+1): HÀNG = nhãn thật, CỘT = dự đoán; chỉ số nc = 'background'.
      cm[i, j]   : GT lớp i được dự đoán thành lớp j (i==j là đúng)
      cm[i, nc]  : GT lớp i bị bỏ sót (FN)
      cm[nc, j]  : dự đoán lớp j không khớp GT nào (FP / dự đoán dư)
    Khớp 1-1 theo IoU giảm dần, không phân biệt lớp ở bước khớp (giống Ultralytics) để lộ lỗi nhầm lớp.
    """
    cm = np.zeros((nc + 1, nc + 1), dtype=int)
    for rec in records:
        pred = rec["pred"][rec["pred"][:, 4] >= conf_thr] if len(rec["pred"]) else rec["pred"]
        gt = rec["gt"]
        if len(gt) == 0:
            for c in pred[:, 5].astype(int):
                cm[nc, c] += 1
            continue
        if len(pred) == 0:
            for c in gt[:, 0].astype(int):
                cm[c, nc] += 1
            continue
        iou = box_iou(gt[:, 1:], pred[:, :4])
        ij = np.argwhere(iou >= iou_thr)
        ij = ij[np.argsort(-iou[ij[:, 0], ij[:, 1]])] if len(ij) else ij
        ug, up = set(), set()
        for gi, pi in ij:
            if gi in ug or pi in up:
                continue
            ug.add(gi); up.add(pi)
            cm[int(gt[gi, 0]), int(pred[pi, 5])] += 1
        for gi in range(len(gt)):
            if gi not in ug:
                cm[int(gt[gi, 0]), nc] += 1
        for pi in range(len(pred)):
            if pi not in up:
                cm[nc, int(pred[pi, 5])] += 1
    return cm


# ----------------------------------------------------------------------------
# 6. Hàm vẽ
# ----------------------------------------------------------------------------
def _finish(fig, save_path=None, show=True, dpi=200):
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _read_results(csv) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]
    return df


def plot_training_curves(results_csv, title="", save_path=None, show=True):
    """
    Biểu đồ huấn luyện từ results.csv của Ultralytics (YOLO và RT-DETR): mọi cặp train/val loss,
    Precision-Recall và mAP theo epoch; đánh dấu epoch tốt nhất theo mAP50-95.
    """
    df = _read_results(results_csv)
    x = df["epoch"] if "epoch" in df else np.arange(1, len(df) + 1)
    losses = {}
    for c in df.columns:
        if c.startswith("train/") and c.endswith("loss"):
            k = c.split("/", 1)[1]
            losses[k] = (c, f"val/{k}" if f"val/{k}" in df.columns else None)
    n = len(losses) + 2
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    best = None
    m = "metrics/mAP50-95(B)"
    if m in df:
        best = int(df[m].idxmax())
    for ax, (k, (tr, va)) in zip(axes, losses.items()):
        ax.plot(x, df[tr], label="train", lw=1.8)
        if va:
            ax.plot(x, df[va], label="val", lw=1.8)
        ax.set_title(k); ax.set_xlabel("epoch"); ax.grid(alpha=.3); ax.legend()
    i = len(losses)
    ax = axes[i]
    for col, lab in (("metrics/precision(B)", "Precision"), ("metrics/recall(B)", "Recall")):
        if col in df:
            ax.plot(x, df[col], label=lab, lw=1.8)
    ax.set_title("Precision / Recall (val)"); ax.set_xlabel("epoch"); ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend()
    ax = axes[i + 1]
    for col, lab in (("metrics/mAP50(B)", "mAP@0.5"), (m, "mAP@0.5:0.95")):
        if col in df:
            ax.plot(x, df[col], label=lab, lw=1.8)
    if best is not None:
        ax.axvline(x.iloc[best] if hasattr(x, "iloc") else x[best], color="gray", ls="--", lw=1)
        ax.set_title(f"mAP (val) - best epoch {int(x.iloc[best] if hasattr(x, 'iloc') else x[best])}: {df[m].iloc[best]:.4f}")
    else:
        ax.set_title("mAP (val)")
    ax.set_xlabel("epoch"); ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend()
    for ax in axes[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    _finish(fig, save_path, show)


def plot_compare_curves(csv_dict: dict, metric="metrics/mAP50-95(B)", title=None, save_path=None, show=True):
    """So sánh 1 chỉ số theo epoch giữa nhiều mô hình. csv_dict = {tên: đường_dẫn_results.csv}."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, p in csv_dict.items():
        df = _read_results(p)
        ax.plot(df["epoch"] if "epoch" in df else np.arange(1, len(df) + 1), df[metric], label=name, lw=1.8)
    ax.set_xlabel("epoch"); ax.set_ylabel(metric); ax.grid(alpha=.3); ax.legend()
    ax.set_title(title or f"{metric} theo epoch")
    _finish(fig, save_path, show)


def plot_pr_curves(curves: dict, names, title="Precision-Recall (per class)", save_path=None, show=True):
    """Đường PR từng lớp + đường trung bình (đậm). curves = build_curves(...)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10" if len(names) <= 10 else "tab20")
    aps, precs = [], []
    for c, d in curves["pr"].items():
        ax.plot(d["x"], d["precision"], color=cmap(c % cmap.N), lw=1.4, label=f"{names[c]}  {d['ap']:.3f}")
        aps.append(d["ap"]); precs.append(d["precision"])
    if precs:
        ax.plot(d["x"], np.mean(precs, axis=0), color="black", lw=3,
                label=f"all classes  mAP@{curves['iou_thr']:.2f} = {np.mean(aps):.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(alpha=.3); ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    _finish(fig, save_path, show)
    return {names[c]: d["ap"] for c, d in curves["pr"].items()}


def plot_roc_curves(curves: dict, names, title="ROC (detection-level, per class)", save_path=None, show=True):
    """ROC mức phát hiện từng lớp (định nghĩa xem build_curves). Trả {lớp: AUC}."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10" if len(names) <= 10 else "tab20")
    aucs = {}
    for c, d in curves["roc"].items():
        ax.plot(d["fpr"], d["tpr"], color=cmap(c % cmap.N), lw=1.6, label=f"{names[c]}  AUC={d['auc']:.3f}")
        aucs[names[c]] = d["auc"]
    ax.set_xlabel("False positive rate (FP / total FP)"); ax.set_ylabel("True positive rate (TP / #GT)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
    ax.set_title(f"{title}\nmean AUC = {np.mean(list(aucs.values())):.3f}" if aucs else title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    _finish(fig, save_path, show)
    return aucs


def plot_froc_curves(curves: dict, names, max_fp=2.0, title="FROC (recall vs FP per image)", save_path=None, show=True):
    """Recall theo số FP trung bình trên mỗi ảnh (thước đo thay thế cho ROC)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10" if len(names) <= 10 else "tab20")
    for c, d in curves["froc"].items():
        ax.plot(d["fp_per_img"], d["recall"], color=cmap(c % cmap.N), lw=1.6, label=names[c])
    ax.set_xlim(0, max_fp); ax.set_ylim(0, 1.02); ax.set_xlabel("FP per image"); ax.set_ylabel("Recall")
    ax.grid(alpha=.3); ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    _finish(fig, save_path, show)


def plot_confusion_matrix(cm: np.ndarray, names, normalize=True, title="Confusion matrix",
                          save_path=None, show=True):
    """cm từ confusion_matrix_det (hàng = thật, cột = dự đoán, nhãn cuối = background)."""
    labels = list(names) + ["background"]
    n = len(labels)
    mat = cm.astype(float)
    if normalize:
        mat = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * n + 2), max(5, 0.8 * n + 1.5)))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    thr = mat.max() / 2 if mat.max() > 0 else 1
    for i in range(n):
        for j in range(n):
            if cm[i, j] == 0 and normalize:
                continue
            txt = f"{mat[i, j]:.2f}" if normalize else f"{cm[i, j]}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="white" if mat[i, j] > thr else "black")
    ax.set_title(title + (" (row-normalized)" if normalize else " (counts)"))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _finish(fig, save_path, show)


SUMMARY_COLS = ["Model", "P", "R", "F1", "mAP50", "mAP50-95", "FPS", "Params(M)", "GFLOPs"]


def make_summary_table(results: dict | list, decimals=4, save_csv=None, save_md=None) -> pd.DataFrame:
    """
    Bảng tổng hợp nhiều mô hình. results = {tên: dict từ evaluate_split (+ 'FPS' nếu đo bằng measure_fps)}
    hoặc list các dict có khóa 'Model'. Hiển thị tô màu: display(df.style.highlight_max(...)) - xem style_summary.
    """
    items = [dict(v, Model=k) for k, v in results.items()] if isinstance(results, dict) else list(results)
    rows = []
    for r in items:
        row = {c: r.get(c) for c in SUMMARY_COLS}
        if row["FPS"] is None:
            row["FPS"] = r.get("FPS_val")
        rows.append(row)
    df = pd.DataFrame(rows).dropna(axis=1, how="all")
    num = df.select_dtypes("number").columns
    df[num] = df[num].round(decimals)
    if save_csv:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)
    if save_md:
        best = {c: df[c].max() for c in ("P", "R", "F1", "mAP50", "mAP50-95", "FPS") if c in df}
        lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
        for _, r in df.iterrows():
            cells = []
            for c in df.columns:
                v = r[c]
                s = "-" if pd.isna(v) else (f"{v:.{decimals}f}" if c in ("P", "R", "F1", "mAP50", "mAP50-95") else f"{v:g}" if isinstance(v, float) else str(v))
                cells.append(f"**{s}**" if c in best and not pd.isna(v) and v == best[c] else s)
            lines.append("| " + " | ".join(cells) + " |")
        Path(save_md).write_text("\n".join(lines), encoding="utf-8")
    return df


def style_summary(df: pd.DataFrame):
    """Styler tô xanh giá trị tốt nhất mỗi cột chỉ số (dùng trong Jupyter)."""
    cols = [c for c in ("P", "R", "F1", "mAP50", "mAP50-95", "FPS") if c in df]
    return df.style.highlight_max(subset=cols, color="#c6efce").format(precision=4)


def plot_model_comparison(df: pd.DataFrame, metrics=("P", "R", "mAP50", "mAP50-95"), title="Model comparison",
                          save_path=None, show=True):
    """Cột nhóm so sánh các chỉ số giữa mô hình; nếu có cột FPS thì thêm 1 panel FPS."""
    metrics = [m for m in metrics if m in df]
    has_fps = "FPS" in df and df["FPS"].notna().any()
    fig, axes = plt.subplots(1, 2 if has_fps else 1, figsize=(11 if has_fps else 8, 4.8),
                             gridspec_kw=dict(width_ratios=[3, 1]) if has_fps else None)
    axes = np.atleast_1d(axes)
    ax = axes[0]
    w = 0.8 / max(len(df), 1)
    xs = np.arange(len(metrics))
    for i, (_, r) in enumerate(df.iterrows()):
        vals = [r[m] for m in metrics]
        bars = ax.bar(xs + i * w - 0.4 + w / 2, vals, w, label=r["Model"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, f"{v:.3f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(xs); ax.set_xticklabels(metrics); ax.set_ylim(0, 1.1); ax.grid(axis="y", alpha=.3)
    ax.legend(fontsize=8); ax.set_title(title)
    if has_fps:
        axes[1].bar(df["Model"], df["FPS"], color=[plt.get_cmap("tab10")(i) for i in range(len(df))])
        axes[1].set_title("FPS"); axes[1].tick_params(axis="x", rotation=45); axes[1].grid(axis="y", alpha=.3)
    _finish(fig, save_path, show)


def per_class_table(results: dict, key="AP50") -> pd.DataFrame:
    """results={model: dict evaluate_split} -> DataFrame (lớp x mô hình) của AP50 hoặc AP50_95."""
    return pd.DataFrame({m: {c: v[key] for c, v in r["per_class"].items()} for m, r in results.items()})


def plot_per_class_ap(table: pd.DataFrame, title="AP per class", save_path=None, show=True):
    """Heatmap AP theo lớp (hàng) x mô hình (cột) để phân tích lớp mạnh/yếu."""
    fig, ax = plt.subplots(figsize=(1.6 * len(table.columns) + 3, 0.6 * len(table) + 2))
    im = ax.imshow(table.values.astype(float), cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(table.shape[1])); ax.set_xticklabels(table.columns, rotation=30, ha="right")
    ax.set_yticks(range(table.shape[0])); ax.set_yticklabels(table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            ax.text(j, i, "n/a" if pd.isna(v) else f"{v:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title(title); fig.colorbar(im, ax=ax, fraction=0.04)
    _finish(fig, save_path, show)


def show_predictions(model, img_paths, conf=0.25, imgsz=640, ncols=3, titles=None, save_path=None, show=True):
    """Lưới ảnh dự đoán (hộp + lớp + confidence) bằng bộ vẽ của Ultralytics."""
    img_paths = list(img_paths)
    res = model.predict(img_paths, imgsz=imgsz, conf=conf, verbose=False)
    nrows = math.ceil(len(res) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, (ax, r) in enumerate(zip(axes, res)):
        ax.imshow(r.plot()[..., ::-1]); ax.axis("off")
        ax.set_title(titles[i] if titles else Path(r.path).name, fontsize=9)
    for ax in axes[len(res):]:
        ax.axis("off")
    _finish(fig, save_path, show)


def plot_gt_vs_pred(model, img_path, names, conf=0.25, imgsz=640, save_path=None, show=True):
    """Hai khung cạnh nhau: nhãn thật (xanh lá) và dự đoán (đỏ). Dùng cho phân tích lỗi bỏ sót/dư/nhầm lớp."""
    import matplotlib.patches as mp
    from PIL import Image

    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    gt = read_yolo_labels(label_path_for(img_path), w, h)
    r = model.predict(img_path, imgsz=imgsz, conf=conf, verbose=False)[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, t in zip(axes, ("Ground truth", f"Prediction (conf >= {conf})")):
        ax.imshow(img); ax.axis("off"); ax.set_title(t)
    for c, x1, y1, x2, y2 in gt:
        axes[0].add_patch(mp.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="lime", lw=1.8))
        axes[0].text(x1, y1 - 3, names[int(c)], color="lime", fontsize=8, weight="bold")
    for (x1, y1, x2, y2), cf, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
        axes[1].add_patch(mp.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=1.8))
        axes[1].text(x1, y1 - 3, f"{names[int(c)]} {cf:.2f}", color="red", fontsize=8, weight="bold")
    _finish(fig, save_path, show)


def evaluate_curves_for_model(model, data_yaml, split="test", imgsz=640, device=None, out_dir="figures",
                              tag="model", conf_cm=0.25, show=True) -> dict:
    """
    Tiện ích 1 lệnh: thu dự đoán trên split -> vẽ & lưu PR, ROC, FROC, confusion matrix (đếm + chuẩn hóa).
    Trả về dict {'AP50_custom', 'AUC', 'cm'} để đối chiếu với số của Ultralytics.
    """
    cfg, names, root = load_data_yaml(data_yaml)
    imgs = list_images(root, cfg[split])
    recs = collect_predictions(model, imgs, imgsz=imgsz, device=device)
    curves = build_curves(recs, len(names), 0.5)
    out = Path(out_dir)
    ap = plot_pr_curves(curves, names, title=f"PR curves - {tag} ({split})", save_path=out / f"{tag}_{split}_pr.png", show=show)
    auc = plot_roc_curves(curves, names, title=f"ROC - {tag} ({split})", save_path=out / f"{tag}_{split}_roc.png", show=show)
    plot_froc_curves(curves, names, title=f"FROC - {tag} ({split})", save_path=out / f"{tag}_{split}_froc.png", show=show)
    cm = confusion_matrix_det(recs, len(names), conf_thr=conf_cm)
    plot_confusion_matrix(cm, names, True, f"Confusion matrix - {tag} ({split})", out / f"{tag}_{split}_cm_norm.png", show)
    plot_confusion_matrix(cm, names, False, f"Confusion matrix - {tag} ({split})", out / f"{tag}_{split}_cm_count.png", show)
    return dict(AP50_custom=ap, AUC=auc, cm=cm.tolist())


# ----------------------------------------------------------------------------
# 7. Tiện ích DeepPCB -> YOLO (chỉ cần khi kiểm tra chéo / gộp dữ liệu)
# ----------------------------------------------------------------------------
DEEPPCB_ID2NAME = {1: "open", 2: "short", 3: "mousebite", 4: "spur", 5: "copper", 6: "pin-hole"}
# Gợi ý ánh xạ sang 5 lớp tương đồng của DsPCBSD+ (SỬA theo tên lớp thực tế trong data.yaml của A).
# pin-hole KHÔNG ánh xạ (None): không có lớp tương đương rõ ràng, tránh nhãn sai.
DEEPPCB_TO_DSPCBSD = {"open": "OP", "short": "SH", "mousebite": "MB", "spur": "SP", "copper": "SC", "pin-hole": None}


def convert_deeppcb_annotation(txt_path, target_names: list[str], name_map: dict = DEEPPCB_TO_DSPCBSD,
                               w: int = 640, h: int = 640) -> list[str]:
    """
    Nhãn DeepPCB (mỗi dòng: x1,y1,x2,y2,type; type 1..6, 0 là nền) -> các dòng YOLO (cls cx cy w h chuẩn hóa).
    target_names = danh sách tên lớp trong data.yaml đích. Lớp ánh xạ None/không có trong đích sẽ bị bỏ.
    """
    out = []
    for ln in Path(txt_path).read_text().splitlines():
        p = [v for v in re.split(r"[,\s]+", ln.strip()) if v]
        if len(p) < 5:
            continue
        x1, y1, x2, y2, t = float(p[0]), float(p[1]), float(p[2]), float(p[3]), int(float(p[4]))
        tgt = name_map.get(DEEPPCB_ID2NAME.get(t, ""))
        if tgt is None or tgt not in target_names:
            continue
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        out.append(f"{target_names.index(tgt)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return out
