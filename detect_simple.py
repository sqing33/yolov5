from pathlib import Path
import time
import torch
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import LOGGER, Profile, check_img_size, non_max_suppression, scale_boxes, cv2
from ultralytics.utils.plotting import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(weights,
        source,
        imgsz=(640, 640),
        conf_thres=0.5,
        iou_thres=0.45,
        device="cpu",
        save=True,
        show=False,
        scale=1,
        save_name=""):

    if save_name == "":
        source_name = Path(source).stem
        file = weights.rfind('.')
        if file == -1:
            weights_name = weights.split('_')[0]
            weights_suffix = weights.split('_')[1]
        else:
            weights_name = Path(weights).stem
            weights_suffix = Path(weights).suffix.lstrip('.')

        save_name = f"{source_name}-{weights_name}-{weights_suffix}.mp4"

    # 1. 加载模型
    device = select_device(device)
    model = DetectMultiBackend(weights,
                               device=device,
                               dnn=False,
                               data=None,
                               fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # 2. 加载视频数据
    dataset = LoadImages(source,
                         img_size=imgsz,
                         stride=stride,
                         auto=pt,
                         vid_stride=1)
    batch_size = 1
    vid_writer = [None] * batch_size

    # 3. 模型预热（可选，但建议执行）
    model.warmup(imgsz=(1 if pt else batch_size, 3, *imgsz))

    # 4. 循环遍历视频帧
    seen, dt = 0, (Profile(), Profile(), Profile())
    total_fps = 0  # 用于计算平均帧率
    frame_count = 0
    total_detections = 0  # 用于统计总检测数

    for path, im, im0s, vid_cap, s in dataset:
        frame_count += 1
        # 4.1 图像预处理
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.float() / 255.0
            if len(im.shape) == 3:
                im = im[None]

        # 4.2 模型推理
        with dt[1]:
            pred = model(im, augment=False, visualize=False)

        # 4.3 非极大值抑制 (NMS)
        with dt[2]:
            pred = non_max_suppression(pred,
                                       conf_thres,
                                       iou_thres,
                                       classes=None,
                                       agnostic=False,
                                       max_det=1000)

        # 计算当前帧率
        current_fps = 1.0 / (dt[1].dt + dt[2].dt)  # 假设主要时间消耗在推理和NMS
        total_fps += current_fps

        # 4.4 处理检测结果
        for i, det in enumerate(pred):
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)
            p = Path(p)
            s += "%gx%g " % im.shape[2:]
            annotator = Annotator(im0, line_width=3, example=str(names))
            detections_in_frame = 0  # 用于统计当前帧检测数

            if len(det):

                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4],
                                         im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f"{names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    detections_in_frame += 1

            # 添加帧率显示
            fps_text = f"FPS: {current_fps:.2f}"
            annotator.text((10, 30), fps_text, txt_color=(0, 0, 255))

            # 4.5 获取带有标注的图像
            im0 = annotator.result()

            # 4.6 保存/显示 输出
            if save:
                if vid_writer[i] is None:
                    fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
                    w = int(vid_cap.get(
                        cv2.CAP_PROP_FRAME_WIDTH)) if vid_cap else im0.shape[1]
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                            ) if vid_cap else im0.shape[0]
                    vid_writer[i] = cv2.VideoWriter(
                        str(save_name), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                        (w, h))
                vid_writer[i].write(im0)

            if show:
                if scale != 1:
                    im0 = cv2.resize(
                        im0, (int(w * scale), int(h * scale)))  # 调整图像大小
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)
        total_detections += detections_in_frame
        # 打印时间（仅推理）
        LOGGER.info(
            f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms"
        )

    # 5. 释放资源 (VideoWriter)
    for vw in vid_writer:
        if vw is not None:
            vw.release()

    # 6. 打印总体处理速度 和 平均帧率
    t = tuple(x.t / seen * 1e3 for x in dt)  # 每张图片的速度
    LOGGER.info(
        f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}"
        % t)
    if frame_count > 0:
        average_fps = total_fps / frame_count
        print(f"平均FPS: {average_fps:.2f}")
    print(f"所有帧共检测到: {total_detections} 个物体")
    if save:
        print(f"保存结果到 '{save_name}'")


if __name__ == "__main__":
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = "yolov5n.pt"
    # weights = "yolov5n.onnx"
    weights = "yolov5n_openvino_model"
    source = "ikun.mp4"
    imgsz = (640, 640)
    conf_thres = 0.5
    iou_thres = 0.45
    save = True
    show = True
    scale = 0.6

    run(weights, source, imgsz, conf_thres, iou_thres, device, save, show,
        scale)

    print(f"耗时: {time.time() - start_time:.2f}s")
