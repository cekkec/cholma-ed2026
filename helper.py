# helper.py
import numpy as np
import torch
from torch.cuda import amp
from tqdm import tqdm
import torch.nn.functional as F
from cam_runtime import batch_cam_to_nodes

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

def logsumexp_pool(edge_logits_list, tau=0.5, num_classes=10, device="cuda"):
    """List[Tensor(E_i,K)] -> (B,K)"""
    pooled = []
    for E in edge_logits_list:
        if E.numel() == 0:
            pooled.append(torch.full((num_classes,), -10.0, device=device))
        else:
            z = torch.log(torch.clamp(torch.mean(torch.exp(E / tau), dim=0), min=1e-8))
            pooled.append(tau * z)
    return torch.stack(pooled, dim=0)

def build_graph_ctx(batch_nodes, max_nodes=16, max_edges=64):
    """
    batch_nodes: List[dict] with {"boxes":[N,4], "class_ids":[N], "scores":[N], "spatial_scale":1.0}
    boxes are assumed to be in feature-map coordinate system (Hf x Wf).
    """
    rois, edge_index, edge_geom, spatial_list = [], [], [], []
    for b, nodes in enumerate(batch_nodes):
        boxes = torch.tensor(nodes.get("boxes", []), dtype=torch.float32)
        cids  = nodes.get("class_ids", [])
        if boxes.numel() > 0:
            boxes = boxes[:max_nodes]
            N = boxes.size(0)
            rois_b = boxes.clone()  # (N,4) in fmap coords
        else:
            rois_b = torch.empty(0, 4, dtype=torch.float32)  # ← (0,4)modified to
            N = 0
        rois.append(rois_b)

        tools = [i for i, c in enumerate(cids[:N]) if c < 6]
        tgts  = [i for i, c in enumerate(cids[:N]) if 6 <= c < 20]
        pairs = [[u, v] for u in tools for v in tgts]
        if len(pairs) == 0:
            edge_index.append(torch.empty(0, 2, dtype=torch.long))
            edge_geom.append(torch.empty(0, 6, dtype=torch.float32))
        else:
            pairs = pairs[:max_edges]
            ei = torch.tensor(pairs, dtype=torch.long)  # (E,2)

            def _geom(bb):
                x1, y1, x2, y2 = bb
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                w, h = (x2 - x1), (y2 - y1)
                area = max(w * h, 1.0)
                return cx, cy, w, h, area

            gi = []
            for u, v in pairs:
                bb_u = boxes[u].tolist() if N else [0, 0, 1, 1]
                bb_v = boxes[v].tolist() if N else [0, 0, 1, 1]
                cxu, cyu, wu, hu, au = _geom(bb_u)
                cxv, cyv, wv, hv, av = _geom(bb_v)
                dx = (cxv - cxu) / max(wu + wv, 1e-3)
                dy = (cyv - cyu) / max(hu + hv, 1e-3)
                dist = np.hypot(dx, dy)
                # IoU (fmap coords)
                xi1, yi1 = max(bb_u[0], bb_v[0]), max(bb_u[1], bb_v[1])
                xi2, yi2 = min(bb_u[2], bb_v[2]), min(bb_u[3], bb_v[3])  # ← duplicate/typo cleanup
                inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
                iou = inter / (au + av - inter + 1e-6)
                gi.append([dx, dy, dist, iou, au, av])

            edge_index.append(ei)
            edge_geom.append(torch.tensor(gi, dtype=torch.float32))

        spatial_list.append(nodes.get("spatial_scale", 1.0))
    return {"rois": rois, "edge_index": edge_index, "edge_geom": edge_geom, "spatial_scale": spatial_list}


# ...omitted (keep file top / existing code)

def train_fn(train_loader, model, CFG, criterion, optimizer, epoch, scheduler, device, scaler):
    losses = AverageMeter()
    model.train()
    accum = max(1, getattr(CFG, "gradient_accumulation_steps", 1))
    optimizer.zero_grad(set_to_none=True)

    verb_learning = getattr(CFG, "verb_learning", "graph")  # graph | img | both

    with torch.no_grad():
        cls_weights = torch.cat([model.tool_head.weight.data, model.target_head.weight.data], dim=0).detach().clone()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for step, data in enumerate(pbar):
        images, labels_tuple = data[:2] if isinstance(data, (list, tuple)) else data
        ivt, inst, verb, target, it, iv, vt = labels_tuple

        images    = images.to(device)
        inst_gt   = inst.to(device)
        verb_gt   = verb.to(device)
        target_gt = target.to(device)
        ivt_gt    = ivt.to(device)

        with amp.autocast():
            if CFG.stage == 1:
                y = model(images, stage=1)  # (B,21)
                loss = criterion(y[:, :6], inst_gt) + criterion(y[:, 6:21], target_gt)

            elif CFG.stage == 2:
                teacher = getattr(model, "_ema_teacher", None) or model
                nodes_list = batch_cam_to_nodes(images, teacher, cls_weights, CFG)
                gctx = build_graph_ctx(nodes_list,
                                       max_nodes=CFG.nodes_max_per_image,
                                       max_edges=CFG.edges_max_per_image)
                y, edge_logits = model(images, stage=2, graph_ctx=gctx)  # y: (B,31)
                tool_logit    = y[:, :6]
                verb_img_logit= y[:, 6:16]     # image-level verb
                target_logit  = y[:, 16:31]

                # MIL pooling for graph verb (image-levelpromotion to)
                verb_mil_logit = logsumexp_pool(edge_logits, tau=CFG.mil_tau, num_classes=10, device=device)  # (B,10)

                # base image heads (tool/target)always trained
                loss = criterion(tool_logit, inst_gt) + criterion(target_logit, target_gt)

                # verb source select
                if verb_learning in ["graph", "both"]:
                    loss = loss + CFG.lambda_verb * criterion(verb_mil_logit, verb_gt)
                if verb_learning in ["img", "both"]:
                    loss = loss + CFG.lambda_img  * criterion(verb_img_logit, verb_gt)

            else:  # Stage 3
                teacher = getattr(model, "_ema_teacher", None) or model
                nodes_list = batch_cam_to_nodes(images, teacher, cls_weights, CFG)
                gctx = build_graph_ctx(nodes_list,
                                       max_nodes=CFG.nodes_max_per_image,
                                       max_edges=CFG.edges_max_per_image)
                y, edge_logits = model(images, stage=3, graph_ctx=gctx)  # y: (B,131)
                tool_logit     = y[:, :6]
                verb_img_logit = y[:, 6:16]
                target_logit   = y[:, 16:31]
                triplet_logit  = y[:, 31:131]

                verb_mil_logit = logsumexp_pool(edge_logits, tau=CFG.mil_tau, num_classes=10, device=device)

                # base image heads + triplet
                loss = criterion(tool_logit, inst_gt) + criterion(target_logit, target_gt) + criterion(triplet_logit, ivt_gt)

                # verb source select
                if verb_learning in ["graph", "both"]:
                    loss = loss + CFG.lambda_verb * criterion(verb_mil_logit, verb_gt)
                if verb_learning in ["img", "both"]:
                    loss = loss + CFG.lambda_img  * criterion(verb_img_logit, verb_gt)

        loss = loss / accum
        scaler.scale(loss).backward()
        if (step + 1) % accum == 0:
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if hasattr(model, "_ema_obj") and model._ema_obj is not None:
                model._ema_obj.update(model)
                model._ema_teacher = model._ema_obj.teacher

        losses.update(loss.item() * accum, images.size(0))
        pbar.set_postfix(loss=f"{losses.avg:.4f}")

    scheduler.step()
    return losses.avg


def valid_fn(valid_loader, model, CFG, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []

    verb_learning = getattr(CFG, "verb_learning", "graph")  # graph | img | both

    with torch.no_grad():
        cls_weights = torch.cat([model.tool_head.weight.data, model.target_head.weight.data], dim=0)
        teacher = getattr(model, "_ema_teacher", None) or model

        for data in valid_loader:
            images, labels_tuple = data[:2] if isinstance(data, (list, tuple)) else data
            ivt, inst, verb, target, it, iv, vt = labels_tuple

            images    = images.to(device)
            inst_gt   = inst.to(device)
            verb_gt   = verb.to(device)
            target_gt = target.to(device)
            ivt_gt    = ivt.to(device)

            if CFG.stage == 1:
                y = model(images, stage=1)
                loss = criterion(y[:, :6], inst_gt) + criterion(y[:, 6:21], target_gt)
                preds.append(torch.sigmoid(y).cpu().numpy())

            elif CFG.stage == 2:
                nodes_list = batch_cam_to_nodes(images, teacher, cls_weights, CFG)
                gctx = build_graph_ctx(nodes_list,
                                       max_nodes=CFG.nodes_max_per_image,
                                       max_edges=CFG.edges_max_per_image)
                y, edge_logits = model(images, stage=2, graph_ctx=gctx)
                tool_logit     = y[:, :6]
                verb_img_logit = y[:, 6:16]
                target_logit   = y[:, 16:31]
                verb_mil_logit = logsumexp_pool(edge_logits, tau=CFG.mil_tau, num_classes=10, device=device)

                loss_img  = criterion(tool_logit, inst_gt) + criterion(target_logit, target_gt)
                loss = 0.0
                if verb_learning in ["graph", "both"]:
                    loss = loss + CFG.lambda_verb * criterion(verb_mil_logit, verb_gt)
                if verb_learning in ["img", "both"]:
                    loss = loss + CFG.lambda_img  * criterion(verb_img_logit, verb_gt)
                loss = loss + loss_img

                # for evaluation verb select
                if verb_learning == "img":
                    verb_eval = torch.sigmoid(verb_img_logit)
                else:  # graph or both -> evaluate based on MIL
                    verb_eval = torch.sigmoid(verb_mil_logit)

                y_eval = torch.cat([torch.sigmoid(tool_logit),
                                    verb_eval,
                                    torch.sigmoid(target_logit)], dim=1)
                preds.append(y_eval.cpu().numpy())

            else:  # Stage 3
                nodes_list = batch_cam_to_nodes(images, teacher, cls_weights, CFG)
                gctx = build_graph_ctx(nodes_list,
                                       max_nodes=CFG.nodes_max_per_image,
                                       max_edges=CFG.edges_max_per_image)
                y, edge_logits = model(images, stage=3, graph_ctx=gctx)
                tool_logit     = y[:, :6]
                verb_img_logit = y[:, 6:16]
                target_logit   = y[:, 16:31]
                triplet_logit  = y[:, 31:131]
                verb_mil_logit = logsumexp_pool(edge_logits, tau=CFG.mil_tau, num_classes=10, device=device)

                loss_img  = criterion(tool_logit, inst_gt) + criterion(target_logit, target_gt) + criterion(triplet_logit, ivt_gt)
                loss = loss_img
                if verb_learning in ["graph", "both"]:
                    loss = loss + CFG.lambda_verb * criterion(verb_mil_logit, verb_gt)
                if verb_learning in ["img", "both"]:
                    loss = loss + CFG.lambda_img  * criterion(verb_img_logit, verb_gt)

                if verb_learning == "img":
                    verb_eval = torch.sigmoid(verb_img_logit)
                else:
                    verb_eval = torch.sigmoid(verb_mil_logit)

                y_eval = torch.cat([torch.sigmoid(tool_logit),
                                    verb_eval,
                                    torch.sigmoid(target_logit),
                                    torch.sigmoid(triplet_logit)], dim=1)
                preds.append(y_eval.cpu().numpy())

            losses.update(loss.item(), images.size(0))

    predictions = np.concatenate(preds, axis=0)
    return losses.avg, predictions


