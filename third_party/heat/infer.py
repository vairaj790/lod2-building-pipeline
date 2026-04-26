import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import scipy.ndimage.filters as filters
from torch.utils.data import DataLoader

from datasets.outdoor_buildings import OutdoorBuildingDataset
from datasets.s3d_floorplans import S3DFloorplanDataset
from datasets.data_utils import collate_fn, get_pixel_features
from models.resnet import ResNetBackbone
from models.corner_models import HeatCorner
from models.edge_models import HeatEdge
from models.corner_to_edge import get_infer_edge_pairs
from utils.geometry_utils import corner_eval
from metrics.get_metric import compute_metrics, get_recall_and_precision


def visualize_cond_generation(
    positive_pixels,
    confs,
    image,
    save_path,
    gt_corners=None,
    prec=None,
    recall=None,
    image_masks=None,
    edges=None,
    edge_confs=None
):
    image = image.copy()
    viz_confs = confs if confs is not None else None

    if edges is not None and len(edges) > 0:
        preds = positive_pixels.astype(int)
        c_degrees = dict()
        for edge_i, edge_pair in enumerate(edges):
            conf = float(edge_confs[edge_i]) if edge_confs is not None and edge_i < len(edge_confs) else 0.5
            conf = max(0.0, min(1.0, conf))
            color_value = int(255 * ((conf * 2) - 1))
            color_value = max(0, min(255, color_value))

            p1 = tuple(map(int, preds[edge_pair[0]]))
            p2 = tuple(map(int, preds[edge_pair[1]]))
            cv2.line(image, p1, p2, (color_value, color_value, 0), 2)

            c_degrees[edge_pair[0]] = c_degrees.setdefault(edge_pair[0], 0) + 1
            c_degrees[edge_pair[1]] = c_degrees.setdefault(edge_pair[1], 0) + 1

    for idx, c in enumerate(positive_pixels):
        if edges is not None and len(edges) > 0 and idx not in c_degrees:
            continue

        if confs is None:
            cv2.circle(image, (int(c[0]), int(c[1])), 3, (0, 0, 255), -1)
        else:
            conf_val = float(viz_confs[idx]) if idx < len(viz_confs) else 0.0
            conf_val = max(0.0, min(1.0, conf_val))
            cv2.circle(image, (int(c[0]), int(c[1])), 3, (0, 0, int(255 * conf_val)), -1)

    if gt_corners is not None and len(gt_corners) > 0:
        for c in gt_corners:
            cv2.circle(image, (int(c[0]), int(c[1])), 3, (0, 255, 0), -1)

    if image_masks is not None:
        mask_ids = np.where(image_masks == 1)[0]
        for mask_id in mask_ids:
            y_idx = mask_id // 64
            x_idx = (mask_id - y_idx * 64)
            x_coord = x_idx * 4
            y_coord = y_idx * 4
            cv2.rectangle(image, (x_coord, y_coord), (x_coord + 3, y_coord + 3), (127, 127, 0), thickness=-1)

    if prec is not None:
        if isinstance(prec, tuple):
            cv2.putText(
                image,
                'edge p={:.2f}, edge r={:.2f}'.format(prec[0], recall[0]),
                (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                image,
                'region p={:.2f}, region r={:.2f}'.format(prec[1], recall[1]),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA
            )
        else:
            cv2.putText(
                image,
                'prec={:.2f}, recall={:.2f}'.format(prec, recall),
                (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA
            )

    cv2.imwrite(save_path, image)


def corner_nms(preds, confs, image_size):
    data = np.zeros([image_size, image_size], dtype=np.float32)
    neighborhood_size = 5
    threshold = 0

    for i in range(len(preds)):
        x = int(preds[i, 0])
        y = int(preds[i, 1])
        if 0 <= x < image_size and 0 <= y < image_size:
            data[y, x] = max(data[y, x], float(confs[i]))

    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    results = np.where(maxima > 0)
    filtered_preds = np.stack([results[1], results[0]], axis=-1)

    new_confs = []
    for pred in filtered_preds:
        new_confs.append(data[pred[1], pred[0]])
    new_confs = np.array(new_confs, dtype=np.float32)

    return filtered_preds, new_confs


def postprocess_preds(corners, confs, edges):
    if len(corners) == 0 or len(edges) == 0:
        return corners, confs, edges

    corner_degrees = dict()
    for edge_pair in edges:
        corner_degrees[edge_pair[0]] = corner_degrees.setdefault(edge_pair[0], 0) + 1
        corner_degrees[edge_pair[1]] = corner_degrees.setdefault(edge_pair[1], 0) + 1

    good_ids = [i for i in range(len(corners)) if i in corner_degrees]
    if len(good_ids) == len(corners):
        return corners, confs, edges

    good_corners = corners[good_ids]
    good_confs = confs[good_ids]
    id_mapping = {value: idx for idx, value in enumerate(good_ids)}
    new_edges = []
    for edge_pair in edges:
        new_edges.append((id_mapping[edge_pair[0]], id_mapping[edge_pair[1]]))
    return good_corners, good_confs, np.array(new_edges)


def convert_annot(annot):
    corners = np.array(list(annot.keys()))
    corners_mapping = {tuple(c): idx for idx, c in enumerate(corners)}
    edges = set()
    for corner, connections in annot.items():
        idx_c = corners_mapping[tuple(corner)]
        for other_c in connections:
            idx_other_c = corners_mapping[tuple(other_c)]
            if (idx_c, idx_other_c) not in edges and (idx_other_c, idx_c) not in edges:
                edges.add((idx_c, idx_other_c))
    edges = np.array(list(edges))
    return {'corners': corners, 'edges': edges}


def get_results(image, backbone, corner_model, edge_model, pixels, pixel_features, args, infer_times, corner_thresh=0.01, edge_thresh=0.70, image_size=256):
    image_feats, feat_mask, all_image_feats = backbone(image)
    pixel_features = pixel_features.unsqueeze(0).repeat(image.shape[0], 1, 1, 1)

    preds_s1 = corner_model(image_feats, feat_mask, pixel_features, pixels, all_image_feats)
    c_outputs_np = preds_s1[0].detach().cpu().numpy()

    pos_indices = np.where(c_outputs_np >= corner_thresh)
    pred_corners = pixels[pos_indices]
    pred_confs = c_outputs_np[pos_indices]
    pred_corners, pred_confs = corner_nms(pred_corners, pred_confs, image_size=image_size)

    if len(pred_corners) < 2:
        return pred_corners, pred_confs, np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32), c_outputs_np

    pred_corners, pred_confs, edge_coords, edge_mask, edge_ids = get_infer_edge_pairs(pred_corners, pred_confs)

    if edge_coords.shape[1] == 0:
        return pred_corners, pred_confs, np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32), c_outputs_np

    corner_nums = torch.tensor([len(pred_corners)], device=image.device)
    max_candidates = torch.stack([corner_nums.max() * args.corner_to_edge_multiplier] * len(corner_nums), dim=0)

    all_pos_ids = set()
    all_edge_confs = dict()

    gt_values = torch.zeros_like(edge_mask).long()
    gt_values[:, :] = 2

    for tt in range(infer_times):
        s1_logits, s2_logits_hb, s2_logits_rel, selected_ids, s2_mask, _ = edge_model(
            image_feats, feat_mask, pixel_features, edge_coords, edge_mask,
            gt_values, corner_nums, max_candidates, True
        )

        selected_ids = selected_ids.squeeze().detach().cpu().numpy()
        selected_ids = np.atleast_1d(selected_ids)

        s2_preds_hb = s2_logits_hb.squeeze().softmax(0)
        if s2_preds_hb.ndim == 1:
            s2_preds_hb = s2_preds_hb.unsqueeze(1)
        s2_preds_np = s2_preds_hb[1, :].detach().cpu().numpy()
        s2_preds_np = np.atleast_1d(s2_preds_np)

        if tt != infer_times - 1:
            pos_edge_ids = np.where(s2_preds_np >= 0.9)[0]
            neg_edge_ids = np.where(s2_preds_np <= 0.01)[0]

            for pos_id in pos_edge_ids:
                actual_id = int(selected_ids[pos_id])
                if gt_values[0, actual_id] != 2:
                    continue
                all_pos_ids.add(actual_id)
                all_edge_confs[actual_id] = float(s2_preds_np[pos_id])
                gt_values[0, actual_id] = 1

            for neg_id in neg_edge_ids:
                actual_id = int(selected_ids[neg_id])
                if gt_values[0, actual_id] != 2:
                    continue
                gt_values[0, actual_id] = 0

            num_total = s1_logits.shape[2]
            num_selected = selected_ids.shape[0]
            num_filtered = num_total - num_selected
            num_to_pred = int((gt_values == 2).sum().item())
            if num_to_pred <= num_filtered:
                break
        else:
            pos_edge_ids = np.where(s2_preds_np >= edge_thresh)[0]
            for pos_id in pos_edge_ids:
                actual_id = int(selected_ids[pos_id])
                if bool(s2_mask[0][pos_id]) is True or gt_values[0, actual_id] != 2:
                    continue
                all_pos_ids.add(actual_id)
                all_edge_confs[actual_id] = float(s2_preds_np[pos_id])

    pos_edge_ids = sorted(list(all_pos_ids))
    if len(pos_edge_ids) == 0:
        return pred_corners, pred_confs, np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32), c_outputs_np

    edge_confs = np.array([all_edge_confs[idx] for idx in pos_edge_ids], dtype=np.float32)
    pos_edges = edge_ids[pos_edge_ids].cpu().numpy()

    return pred_corners, pred_confs, pos_edges, edge_confs, c_outputs_np


def main(args):
    ckpt = torch.load(args.checkpoint_path)
    print('Load from ckpts of epoch {}'.format(ckpt['epoch']))
    ckpt_args = ckpt['args']

    if args.dataset == 'outdoor':
        data_path = './data/outdoor/cities_dataset'

        # The key change:
        # In inference, do not require det_final.
        dataset = OutdoorBuildingDataset(
            data_path=data_path,
            det_path=None,
            phase='all',
            image_size=args.image_size,
            rand_aug=False,
            inference=True
        )

    elif args.dataset == 's3d_floorplan':
        data_path = './data/s3d_floorplan'
        dataset = S3DFloorplanDataset(data_path, phase='test', rand_aug=False, inference=True)
    else:
        raise ValueError('Unknown dataset type: {}'.format(args.dataset))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    backbone = ResNetBackbone()
    strides = backbone.strides
    num_channels = backbone.num_channels
    backbone = nn.DataParallel(backbone).cuda().eval()

    corner_model = HeatCorner(
        input_dim=128,
        hidden_dim=256,
        num_feature_levels=4,
        backbone_strides=strides,
        backbone_num_channels=num_channels
    )
    corner_model = nn.DataParallel(corner_model).cuda().eval()

    edge_model = HeatEdge(
        input_dim=128,
        hidden_dim=256,
        num_feature_levels=4,
        backbone_strides=strides,
        backbone_num_channels=num_channels
    )
    edge_model = nn.DataParallel(edge_model).cuda().eval()

    backbone.load_state_dict(ckpt['backbone'])
    corner_model.load_state_dict(ckpt['corner_model'])
    edge_model.load_state_dict(ckpt['edge_model'])

    print('Loaded saved model from {}'.format(args.checkpoint_path))

    os.makedirs(args.viz_base, exist_ok=True)
    os.makedirs(args.save_base, exist_ok=True)

    all_prec = []
    all_recall = []

    corner_tp = 0.0
    corner_fp = 0.0
    corner_length = 0.0
    edge_tp = 0.0
    edge_fp = 0.0
    edge_length = 0.0
    region_tp = 0.0
    region_fp = 0.0
    region_length = 0.0
    num_eval_samples = 0

    pixels, pixel_features = get_pixel_features(image_size=args.image_size)

    processed_count = 0

    for data_i, data in enumerate(dataloader):
        if data_i < args.start_index:
            continue

        if args.max_samples != -1 and processed_count >= args.max_samples:
            print(f"Reached max_samples={args.max_samples}. Stopping inference.")
            break

        image = data['img'].cuda()
        img_path = data['img_path'][0]
        annot_path = data['annot_path'][0] if len(data['annot_path']) > 0 else ''
        has_gt = bool(annot_path) and os.path.exists(annot_path)

        annot = None
        gt_corners = None
        if has_gt:
            annot = np.load(annot_path, allow_pickle=True, encoding='latin1').tolist()
            gt_corners = np.array(list(annot.keys())).round()

        with torch.no_grad():
            pred_corners, pred_confs, pos_edges, edge_confs, _ = get_results(
                image,
                backbone,
                corner_model,
                edge_model,
                pixels,
                pixel_features,
                ckpt_args,
                args.infer_times,
                corner_thresh=args.corner_thresh,
                edge_thresh=args.edge_thresh,
                image_size=args.image_size
            )

        viz_image = data['raw_img'][0].cpu().numpy().transpose(1, 2, 0)
        viz_image = (viz_image * 255).astype(np.uint8)

        img_basename = os.path.splitext(os.path.basename(img_path))[0]

        if has_gt:
            gt_path = os.path.join(args.viz_base, f'{img_basename}_gt.png')
            visualize_cond_generation(gt_corners, None, viz_image, gt_path)

        if has_gt and len(pred_corners) > 0:
            prec, recall = corner_eval(gt_corners, pred_corners)
        elif has_gt:
            prec, recall = 0, 0
        else:
            prec, recall = None, None

        if has_gt:
            all_prec.append(prec)
            all_recall.append(recall)

        if pred_confs.shape[0] == 0:
            pred_confs = None

        pred_corner_path = os.path.join(args.viz_base, f'{img_basename}_pred_corner.png')
        visualize_cond_generation(
            pred_corners,
            pred_confs,
            viz_image,
            pred_corner_path,
            gt_corners=gt_corners if has_gt else None,
            prec=prec,
            recall=recall
        )

        pred_corners, pred_confs, pos_edges = postprocess_preds(pred_corners, pred_confs, pos_edges)

        save_results = {
            'corners': pred_corners,
            'edges': pos_edges,
            'image_path': img_path,
        }
        save_path = os.path.join(args.save_base, f'{img_basename}_results.npy')
        np.save(save_path, save_results)

        if has_gt:
            pred_data = {'corners': pred_corners, 'edges': pos_edges}
            gt_data = convert_annot(annot)

            try:
                score = compute_metrics(gt_data, pred_data)

                edge_recall, edge_prec = get_recall_and_precision(score['edge_tp'], score['edge_fp'], score['edge_length'])
                region_recall, region_prec = get_recall_and_precision(score['region_tp'], score['region_fp'], score['region_length'])
                er_recall = (edge_recall, region_recall)
                er_prec = (edge_prec, region_prec)

                corner_tp += score['corner_tp']
                corner_fp += score['corner_fp']
                corner_length += score['corner_length']
                edge_tp += score['edge_tp']
                edge_fp += score['edge_fp']
                edge_length += score['edge_length']
                region_tp += score['region_tp']
                region_fp += score['region_fp']
                region_length += score['region_length']
                num_eval_samples += 1

            except ZeroDivisionError:
                print(f"⚠️ Skipping GT metrics for {img_basename}: no valid GT region could be formed.")
                er_prec = None
                er_recall = None
        else:
            er_prec = None
            er_recall = None

        pred_edge_path = os.path.join(args.viz_base, f'{img_basename}_pred_edge.png')
        visualize_cond_generation(
            pred_corners,
            pred_confs,
            viz_image,
            pred_edge_path,
            gt_corners=gt_corners if has_gt else None,
            prec=er_prec,
            recall=er_recall,
            edges=pos_edges,
            edge_confs=edge_confs
        )

        print(f'Finish inference for sample No.{data_i} | {img_basename} | GT={has_gt}')
        processed_count += 1

    if num_eval_samples > 0:
        avg_prec = np.array(all_prec).mean() if len(all_prec) > 0 else 0.0
        avg_recall = np.array(all_recall).mean() if len(all_recall) > 0 else 0.0

        recall, precision = get_recall_and_precision(corner_tp, corner_fp, corner_length)
        f_score = 2.0 * precision * recall / (recall + precision + 1e-8)
        print('corners - precision: %.3f recall: %.3f f_score: %.3f' % (precision, recall, f_score))

        recall, precision = get_recall_and_precision(edge_tp, edge_fp, edge_length)
        f_score = 2.0 * precision * recall / (recall + precision + 1e-8)
        print('edges - precision: %.3f recall: %.3f f_score: %.3f' % (precision, recall, f_score))

        recall, precision = get_recall_and_precision(region_tp, region_fp, region_length)
        f_score = 2.0 * precision * recall / (recall + precision + 1e-8)
        print('regions - precision: %.3f recall: %.3f f_score: %.3f' % (precision, recall, f_score))

        print('Avg prec: {}, Avg recall: {}'.format(avg_prec, avg_recall))
    else:
        print('No GT available in the selected inference set. Saved prediction-only outputs.')


def get_args_parser():
    parser = argparse.ArgumentParser('HEAT inference', add_help=False)
    parser.add_argument('--dataset', default='outdoor', help='outdoor/s3d_floorplan')
    parser.add_argument('--checkpoint_path', default='', help='path to checkpoint')
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--viz_base', default='./results/viz', help='path to save visualizations')
    parser.add_argument('--save_base', default='./results/npy', help='path to save prediction npy files')
    parser.add_argument('--infer_times', default=3, type=int)
    parser.add_argument('--start_index', default=0, type=int, help='first sample index to start from')
    parser.add_argument('--corner_thresh', default=0.03, type=float, help='corner threshold before NMS')
    parser.add_argument('--edge_thresh', default=0.70, type=float, help='final edge threshold')
    parser.add_argument('--max_samples', default=-1, type=int,
                        help='maximum number of samples to process after start_index; -1 means all')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HEAT inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)