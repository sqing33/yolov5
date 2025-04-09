from pathlib import Path
import cv2
import numpy as np
import os
import time
import torch
import warnings

warnings.filterwarnings("ignore")

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_boxes, cv2
from utils.torch_utils import select_device, smart_inference_mode


class YOLOv5Evaluator:
    """
    使用PyTorch进行YOLOv5模型评估的类
    """

    def __init__(self, model, classes, conf_threshold=0.5, iou_threshold=0.45):
        # 加载PyTorch模型
        self.device = select_device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = DetectMultiBackend(model,
                                        device=self.device,
                                        dnn=False,
                                        data=None,
                                        fp16=False)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size((640, 640), s=self.stride)

        # 预热模型
        self.model.warmup(imgsz=(1, 3, *self.imgsz))

        # 加载类别信息
        self.classes = self._load_classes(classes)
        self.model_name = Path(model).stem

        # 阈值
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def _load_classes(self, input):
        if isinstance(input, str):
            with open(input, "r") as f:
                classes = [line.strip() for line in f.readlines()]
        else:
            classes = input
        return classes

    def calculate_iou(self, box1, box2):
        """
        计算两个边界框的IoU
        box1, box2: [x1, y1, x2, y2] 格式
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    @smart_inference_mode()
    def evaluate(self, image_dir, label_dir, save=True, output_dir="output"):
        """
        执行模型评估
        """

        total_images = 0
        total_correct = 0
        total_gt_boxes = 0
        total_pred_boxes = 0
        total_infer_time = 0

        # 创建输出目录
        if save:
            if '.' in model:
                parts = model.split('.')
                target_dir = os.path.join(output_dir, f"{parts[0]}-{parts[1]}")
            else:
                parts = model.split('_')
                target_dir = os.path.join(output_dir, f"{parts[0]}-{parts[1]}")

            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)

        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                total_images += 1
                image_path = os.path.join(image_dir, filename)
                label_path = os.path.join(
                    label_dir,
                    os.path.splitext(filename)[0] + '.txt')

                # 读取图像 (使用 LoadImages 的预处理部分)
                dataset = LoadImages(image_path,
                                     img_size=self.imgsz,
                                     stride=self.stride,
                                     auto=self.pt)
                _, im, im0s, _, _ = next(
                    iter(dataset))  # im 是预处理后的, im0s 是原始图像
                img = im0s

                # 预处理
                im = torch.from_numpy(im).to(self.model.device)
                im = im.float() / 255.0
                if len(im.shape) == 3:
                    im = im[None]

                # 使用 DetectMultiBackend 进行推理
                start_time = time.time()
                pred = self.model(im, augment=False, visualize=False)
                infer_time = time.time() - start_time
                total_infer_time += infer_time

                # NMS
                pred = non_max_suppression(pred,
                                           self.conf_threshold,
                                           self.iou_threshold,
                                           classes=None,
                                           agnostic=False,
                                           max_det=1000)

                # 解析预测结果
                pred_boxes = []
                pred_scores = []
                pred_classes = []

                for i, det in enumerate(pred):
                    if len(det):
                        # 将坐标从 img_size 缩放到 原始图像尺寸
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4],
                                                 im0s.shape).round()

                        for *xyxy, conf, cls in det:
                            x1, y1, x2, y2 = map(int, xyxy)
                            pred_boxes.append([x1, y1, x2, y2])
                            pred_scores.append(conf.item())
                            pred_classes.append(int(cls))

                # 读取真实标签
                gt_boxes = []
                gt_classes = []
                orig_h, orig_w = img.shape[:2]
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        for line in f.readlines():
                            parts = line.strip().split()
                            class_id = int(parts[0])
                            cx = float(parts[1]) * orig_w
                            cy = float(parts[2]) * orig_h
                            w = float(parts[3]) * orig_w
                            h = float(parts[4]) * orig_h

                            x1 = int(cx - w / 2)
                            y1 = int(cy - h / 2)
                            x2 = int(cx + w / 2)
                            y2 = int(cy + h / 2)
                            gt_boxes.append([x1, y1, x2, y2])
                            gt_classes.append(class_id)

                # 统计指标
                total_gt_boxes += len(gt_boxes)
                total_pred_boxes += len(pred_boxes)

                # 可视化结果
                vis_img = img.copy()
                for box, score, cls in zip(pred_boxes, pred_scores,
                                           pred_classes):
                    x1, y1, x2, y2 = box
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"{self.classes[cls]}: {score:.2f}"
                    cv2.putText(vis_img, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                for box, cls in zip(gt_boxes, gt_classes):
                    x1, y1, x2, y2 = box
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_img, self.classes[cls], (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 保存可视化结果
                if save:
                    save_path = os.path.join(target_dir, filename)
                    cv2.imwrite(save_path, vis_img)

                # 计算匹配结果
                matched = [False] * len(gt_boxes)
                correct = 0
                for p_idx, (p_box,
                            p_cls) in enumerate(zip(pred_boxes, pred_classes)):
                    for g_idx, (g_box,
                                g_cls) in enumerate(zip(gt_boxes, gt_classes)):
                        if p_cls == g_cls and not matched[g_idx]:
                            iou = self.calculate_iou(p_box, g_box)
                            if iou > 0.5:
                                correct += 1
                                matched[g_idx] = True
                                break

                total_correct += correct
                print(f"----- 图像: {filename} -----")
                print(f"  推理时间: {infer_time*1000:.2f} ms")
                print(f"  预测框数量: {len(pred_boxes)}, 真实框数量: {len(gt_boxes)}")
                print(f"  此图像中正确检测的数量: {correct}")
                print("\n")

        # 计算评估指标
        precision = total_correct / total_pred_boxes if total_pred_boxes > 0 else 0
        recall = total_correct / total_gt_boxes if total_gt_boxes > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (
            precision + recall) > 0 else 0

        # 计算每张图片平均耗时
        avg_infer_time = total_infer_time / total_images if total_images > 0 else 0
        """
        精确率 (Precision)：模型预测为正类的样本里，真正是正类样本的比例，体现预测正类结果的准确性。
        召回率 (Recall)：实际正类样本中，被模型正确预测为正类的比例，反映模型找出正类样本的能力。
        F1 分数 (F1-score)：精确率和召回率的调和平均数，用于衡量模型在两者间平衡表现的综合指标。
        """
        print(f"----- 总体评估结果 -----")
        print(f"总图像数量: {total_images}")
        print(f"数据集框数: {total_gt_boxes}")
        print(f"总预测框数: {total_pred_boxes}")
        print(f"正确检测数: {total_correct}")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1 分数 (F1-score): {f1_score:.4f}")
        print(f"每张图片平均推理耗时: {avg_infer_time*1000:.2f}ms")


if __name__ == "__main__":
    # 输入模型 (pt / onnx / openvino)
    # model = "yolov5n.pt"
    model = "yolov5n.onnx"
    # model = "yolov5n_openvino_model"

    # 输入类别（类名文件 / 类名列表）
    classes = "coco.names"
    # classes = ["role", "monster", "door", "drop"]

    # 置信度， IOU 阈值
    conf_threshold = 0.5
    iou_threshold = 0.45

    # 用于评估的数据集
    image_dir = "coco/images"
    label_dir = "coco/labels"

    # 是否保存可视化结果
    # (保存路径 = output_dir / 模型名称-模型类型 / filename)
    save = True
    output_dir = "coco/output"

    evaluator = YOLOv5Evaluator(model, classes, conf_threshold, iou_threshold)
    evaluator.evaluate(image_dir, label_dir, save, output_dir)
