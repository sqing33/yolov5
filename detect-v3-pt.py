from pathlib import Path
import time
import cv2
import torch
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class YOLOv5Detector:

    def __init__(self, model, classes, conf_threshold=0.5, nms_threshold=0.45):
        # 获取置信度和 NMS 阈值
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        # 加载类别标签
        self.classes = self._load_classes(classes)

        # 加载PyTorch模型
        self.model = self._load_model(model)
        self.model_name = Path(model).stem

        # 获取模型输入尺寸
        self.input_shape = self.model.img_size if hasattr(
            self.model, 'img_size') else 640

    def _load_classes(self, input):
        if isinstance(input, str):
            with open(input, "r") as f:
                classes = [line.strip() for line in f.readlines()]
        else:
            classes = input
        return classes

    def _load_model(self, model):
        # 使用YOLOv5官方方式加载模型
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model)
        model.conf = self.conf_threshold  # 置信度阈值
        model.iou = self.nms_threshold  # NMS阈值
        return model

    def preprocess(self, frame):
        # 使用YOLOv5自带的预处理
        return frame

    def postprocess(self, results, frame):
        # 直接使用YOLOv5的结果解析
        num_objects = 0
        for *box, conf, cls in results.xyxy[0]:
            if conf >= self.conf_threshold:
                num_objects += 1
                self.draw_boxes(frame, int(cls), conf.item(), int(box[0]),
                                int(box[1]), int(box[2]), int(box[3]))
        return frame, num_objects

    def draw_boxes(self, frame, class_id, confidence, left, top, right,
                   bottom):
        # 绘制边界框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # 文本标签
        label = f"{self.classes[class_id]} {confidence*100:.1f}%"

        # 计算文本尺寸
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                              1, 2)

        # 绘制文本背景
        cv2.rectangle(frame, (left, top - text_h - 5),
                      (left + text_w, top - 2), (0, 0, 255), -1)

        # 绘制文本
        cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

        return frame

    def detect(self, frame):
        # YOLOv5自动处理预处理
        results = self.model(frame)
        # 后处理
        return self.postprocess(results, frame)

    def process_resource(self, resource, save=True, show=True, scale=1):
        full_path_str = Path(resource).resolve()
        resource_name = Path(resource).stem
        save_name = f"{resource_name}-{self.model_name}-pytorch"

        img = cv2.imread(resource)

        total_objects_detected = 0

        detect_start_time = time.time()
        if img is not None:
            height, width = img.shape[:2]
            result, num_objects = self.detect(img)
            total_objects_detected += num_objects

            print(
                f"image {full_path_str}: {(time.time() - detect_start_time) * 1000:.1f}ms"
            )

            if save:
                cv2.imwrite(f"{save_name}.jpg", result)
                print(f"检测结果已保存到 '{save_name}.jpg'")
            if show:
                if scale != 1:
                    result = cv2.resize(
                        result, (int(width * scale), int(height * scale)))
                cv2.imshow("result", result)
                cv2.waitKey(0)
        else:
            cap = cv2.VideoCapture(resource)

            if not cap.isOpened():
                print("无法打开视频文件")
                return

            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if save:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(f"{save_name}.mp4", fourcc, 20.0,
                                      (original_width, original_height))

            frame_count = 0
            total_time = 0
            fps = 0.0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                start_time = time.time()
                result, num_objects = self.detect(frame)
                elapsed_time = time.time() - start_time
                total_objects_detected += num_objects

                frame_count += 1
                total_time += elapsed_time

                print(
                    f"video ({frame_count}/{total_frames}) {full_path_str}: {elapsed_time * 1000:.1f}ms"
                )

                # 显示FPS
                fps = (fps * 0.9) + (0.1 / elapsed_time)
                cv2.putText(result, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if scale != 1:
                    result = cv2.resize(result, (int(
                        original_width * scale), int(original_height * scale)))

                if show:
                    cv2.imshow("result", result)

                if save:
                    out.write(result)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            average_fps = frame_count / total_time
            print(f"平均 {average_fps:.2f}FPS")
            print(f"检测耗时: {time.time() - detect_start_time:.2f}s")
            if save:
                print(f"保存结果到 '{save_name}.mp4'")

            cap.release()
            if save:
                out.release()
            cv2.destroyAllWindows()

        print(f"所有帧共检测到: {total_objects_detected} 个物体")


if __name__ == "__main__":
    # 模型路径（YOLOv5 PyTorch模型）
    model = "yolov5n.pt"

    # 类别文件
    classes = "coco.names"

    # 要检测的资源
    # resource = "bus.jpg"
    resource = "car.mp4"

    conf_threshold = 0.5
    nms_threshold = 0.45

    # 创建检测器实例
    detector = YOLOv5Detector(model, classes, conf_threshold, nms_threshold)
    detector.process_resource(resource, save=True, show=True, scale=0.6)
