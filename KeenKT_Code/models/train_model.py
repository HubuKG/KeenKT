import os
import sys
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy
import numpy as np
from .evaluate_model import evaluate
from train.config import que_type_models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log_epoch_summary(
    epoch,
    train_loss,
    valid_auc,
    valid_acc,
    best_auc,
    best_acc,
    best_epoch,
    model_name,
    emb_type,
    save_dir,
    test_auc=-1,
    test_acc=-1,
    window_test_auc=-1,
    window_test_acc=-1
):
    print("=" * 80)
    print(f"Epoch: {epoch}")
    print(f"Train Loss: {train_loss:.6f}")
    print(f"Valid AUC: {valid_auc:.4f}, Valid ACC: {valid_acc:.4f}")
    print(f"Best AUC: {best_auc:.4f}, Best ACC: {best_acc:.4f} (Epoch {best_epoch})")
    print(f"Model: {model_name}, Embedding Type: {emb_type}")
    print(f"Save Directory: {save_dir}")
    print(f"Test AUC: {test_auc}, Test ACC: {test_acc}, Window Test AUC: {window_test_auc}, Window Test ACC: {window_test_acc}")
    print("=" * 80)

def cal_loss(model, ys, r, rshft, sm, preloss=None):
    model_name = model.model_name
    preloss = preloss if preloss is not None else []

    def _base_bce(y_pred, target, sm):
        y_valid = y_pred[sm]
        t_valid = target[sm]
        return binary_cross_entropy(y_valid.double(), t_valid.double())

    if model_name in ["simplekt", "dkt", "sakt", "saint"]:
        return _base_bce(ys[0], rshft, sm)

    elif model_name in ["KeenKT", "KEENKT"]:
        if not ys:
            raise ValueError("KEENKT模型未返回有效输出，请检查forward方法")

        bce_loss = _base_bce(ys[0], rshft, sm)
        cl_loss = 0.0
        if model.use_CL and len(ys) > 1 and isinstance(ys[1], torch.Tensor):
            cl_loss = ys[1].mean()
        return bce_loss + model.cl_weight * cl_loss

    elif model_name in ["akt"]:
        if len(ys) < 1 or len(preloss) < 1:
            raise ValueError("AKT模型输出结构不完整")
        return _base_bce(ys[0], rshft, sm) + preloss[0]

    else:
        raise ValueError(f"不支持的模型: {model_name}")

def model_forward(model, data):
    model_name = model.model_name
    dcur = {k: v.to(device) for k, v in data.items()}

    q, c, r = dcur["qseqs"], dcur["cseqs"], dcur["rseqs"]
    qshft, rshft, sm = dcur["shft_qseqs"], dcur["shft_rseqs"], dcur["smasks"]

    ys, preloss = [], []

    if model_name in ["simplekt"]:
        outputs = model(dcur, train=True)
        ys = [outputs[0][:, 1:]]

    elif model_name in ["KeenKT", "KEENKT"]:
        outputs = model(dcur, train=True)
        if model.use_CL:
            if len(outputs) < 3:
                raise ValueError("KEENKT对比学习模式需要返回至少3个值")
            y_main, cl_loss, temp = outputs[:3]
            ys = [y_main[:, 1:], cl_loss]
        else:
            ys = [outputs[0][:, 1:]]

    elif model_name in ["dkt"]:
        y = model(c.long(), r.long())
        y = (y * nn.functional.one_hot(qshft.long(), model.num_q)).sum(-1)[:, 1:]
        ys.append(y)

    elif model_name in ["akt"]:
        y, reg_loss = model(dcur["cseqs"].long(), dcur["rseqs"].long(), qshft.long())
        ys.append(y[:, 1:])
        preloss.append(reg_loss)

    loss = cal_loss(model, ys, r, rshft, sm, preloss)

    if model_name in ["KeenKT"] and model.use_CL:
        return loss, temp.item()
    return loss

def train_model(model, train_loader, valid_loader, num_epochs, optimizer,
                ckpt_path, test_loader=None, save_model=True, fold=None, 
                patience=10, log_interval=100):

    best_metrics = {
        "valid_auc": -1,
        "test_auc": -1,
        "best_epoch": 0,
        "best_temps": []
    }
    best_epoch = 0
    wait = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = []
        temps = []

        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            if model.model_name in ["KeenKT"] and model.use_CL:
                loss, temp = model_forward(model, data)
                temps.append(temp)
            else:
                loss = model_forward(model, data)

            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

            if batch_idx % log_interval == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}",
                      flush=True)

        model.eval()
        with torch.no_grad():
            valid_auc, valid_acc = evaluate(model, valid_loader, model.model_name)

        if valid_auc > best_metrics["valid_auc"] + 1e-4:
            best_metrics.update({
                "valid_auc": valid_auc,
                "best_epoch": epoch,
                "best_temps": temps.copy()
            })
            wait = 0

            if save_model:
                fold_str = f"_fold{fold}" if fold is not None else ""
                ckpt_name = f"{model.model_name}{fold_str}_epoch{epoch}.pth"
                torch.save(model.state_dict(), os.path.join(ckpt_path, ckpt_name))
                print(f"[SAVE] 保存最佳模型 epoch{epoch} (AUC={valid_auc:.4f})", flush=True)

            if test_loader:
                test_auc, test_acc = evaluate(model, test_loader, model.model_name)
                best_metrics["test_auc"] = test_auc
        else:
            wait += 1
            if wait >= patience:
                print(f"[STOP] 早停触发：连续{patience}轮未提升", flush=True)
                break

        log_epoch_summary(
            epoch=epoch,
            train_loss=np.mean(epoch_loss),
            valid_auc=valid_auc,
            valid_acc=valid_acc,
            best_auc=best_metrics["valid_auc"],
            best_acc=valid_acc,
            best_epoch=best_metrics["best_epoch"],
            model_name=model.model_name,
            emb_type=model.emb_type,
            save_dir=ckpt_path,
            test_auc=best_metrics.get("test_auc", -1),
            test_acc=-1
        )

    return best_metrics
