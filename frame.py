import os
import cv2
import random


def process_video(video_path, output_dir, skip_frames=0, method="裁剪"):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frame_count = 0

    # 获取总帧数（可能出现无法精确获取的情况）
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0

    while True:
        # 直接跳转到目标帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()

        # 检查是否读取成功以及是否超出范围
        if not ret or frame_count >= total_frames:
            break

        # 处理当前帧
        if method == "缩放":
            frame = resize_and_pad(frame, (640, 640))
        elif method == "裁剪":
            frame = resize_and_crop(frame, (640, 640))

        print(f"生成第{saved_frame_count+190}帧")
        frame_path = os.path.join(output_dir, f"{saved_frame_count+190}.jpg")
        cv2.imwrite(frame_path, frame)
        saved_frame_count += 1

        # 计算下一目标帧
        frame_count += skip_frames + 1

    cap.release()
    print(f"Processed video {os.path.basename(video_path)}")


def resize_and_pad(image, size):
    h, w = image.shape[:2]
    # 使用更快的插值方法
    scale = min(size[0] / w, size[1] / h) if w != 0 and h != 0 else 1.0
    new_w, new_h = int(w * scale), int(h * scale)

    # 使用线性插值代替默认的INTER_LINEAR（其实默认就是这个，可注释）
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 计算填充区域
    top = (size[1] - new_h) // 2
    bottom = size[1] - new_h - top
    left = (size[0] - new_w) // 2
    right = size[0] - new_w - left

    # 使用更快的边框填充方式（这里已经是较快的常量填充）
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded_image


def resize_and_crop(image, size):
    h, w = image.shape[:2]
    scale = size[0] / min(w, h) if w != 0 and h != 0 else 1.0
    new_w, new_h = int(w * scale), int(h * scale)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if new_w > size[0]:
        left = random.randint(0, new_w - size[0])
        cropped_image = resized_image[:, left : left + size[0]]
    else:
        top = random.randint(0, new_h - size[1])
        cropped_image = resized_image[top : top + size[1], :]

    return cropped_image


if __name__ == "__main__":
    process_video("0.mkv", "./datasets/frames", skip_frames=50, method="裁剪")
