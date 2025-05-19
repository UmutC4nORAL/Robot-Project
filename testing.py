import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage
import numpy as np
from simple_pid import PID
from image_processing_2025.utils import (
    letterbox, lane_line_mask, select_device
)
import time


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(1, 29, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(3, stride=2 , ceil_mode=True)
        self.conv3 = nn.Conv2d(29, 59, kernel_size=3)
        self.maxpool3 = nn.MaxPool2d(3, stride=2 , ceil_mode=True)
        self.conv4 = nn.Conv2d(59, 74, kernel_size=3)
        self.maxpool4 = nn.MaxPool2d(3, stride=2 , ceil_mode=True)
        self.conv2_drop = nn.Dropout2d()
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1184, 300)
        self.fc2 = nn.Linear(300, 43)
        self.conv0_bn = nn.BatchNorm2d(3)
        self.conv1_bn = nn.BatchNorm2d(1)
        self.conv2_bn = nn.BatchNorm2d(29)
        self.conv3_bn = nn.BatchNorm2d(59)
        self.conv4_bn = nn.BatchNorm2d(74)
        self.dense1_bn = nn.BatchNorm1d(300)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(self.conv0_bn(x))))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(self.maxpool2(x))))
        x = F.relu(self.conv4_bn(self.conv4(self.maxpool3(x))))
        x = self.maxpool4(x)
        x = x.view(-1, 1184)
        x = F.relu(self.fc1(x))
        x = self.dense1_bn(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, training=self.training)
        return F.log_softmax(x, dim=1)


def main():
    # Modellerin yüklenmesi
    yolo_model = YOLO("/home/otagg/Desktop/bestv11.pt")
    yolo_model.to('cuda:0' if torch.cuda.is_available() else 'cpu')

    class_model = Net()
    class_model.load_state_dict(torch.load("/home/otagg/Desktop/balanced_micron.pth"))
    class_model.to('cuda:0' if torch.cuda.is_available() else 'cpu')
    class_model.eval()

    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    classes = ["20 km/h hiz siniri",
            "30 km/h hiz siniri",
            "50 km/h hiz siniri",
            "60 km/h hiz siniri",
            "70 km/h hiz siniri",
            "80 km/h hiz siniri",
            "mecburi sag",
            "100 km/h hiz siniri",
            "120 km/h hiz siniri",
            "sollamak yasak",
            "kamyonlar icin sollamak yasak",
            "ana yol tali yol kavsagi",
            "park etmek yasak",
            "yol ver",
            "dur",
            "tasit trafigine kapali yol",
            "kamyon giremez",
            "girisi olmayan yol",
            "dikkat",
            "sola donus yasak",
            "saga donus yasak",
            "sola tehlikeli devamli virajlar",
            "sola mecburi yon",
            "yol calismasi",
            "kaygan yol",
            "donel kavsak",
            "trafik isareti",
            "yaya geciti",
            "park",
            "bisiklet giremez",
            "gizli buzlanma",
            "durak",
            "kirmizi isik",
            "ileriden saga mecburi yon",
            "ileriden sola mecburi yon",
            "ileri mecburi yon",
            "ileri ve saga mecburi yon",
            "ileri ve sola mecburi yon",
            "sagdan gidiniz",
            "soldan gidiniz",
            "sari isik",
            "yesil isik",
            "sagdan daralan yol"]  # Aynı sınıf isimleri burada yer almalı (gerekirse yukarıdan kopyalanabilir)

    # YOLOPv2 şerit tespit modeli
    weights = '/home/otagg/Desktop/yolopv2.pt'
    img_size = 640
    stride = 32
    half = True
    device = select_device('0')
    model = torch.jit.load(weights).to(device)
    if half:
        model.half()
    model.eval()

    pid = PID(0.005, 0, 0.001)

    # Kamera başlat
    cap = cv2.VideoCapture('/dev/video2')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        is_accelerated = False

        # === YOLOv8 ile nesne tespiti ===
        try:
            results = yolo_model(frame)
            for r in results:
                boxes = r.boxes
                cls_ids = r.boxes.cls.cpu().numpy()
                for i, box in enumerate(boxes):
                    b = box.xyxy[0].to('cpu').detach().numpy().copy()
                    cropped_img = frame[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
                    if cropped_img.size == 0:
                        continue
                    cropped_img_pil = PILImage.fromarray(cropped_img)
                    cropped_tensor = transform(cropped_img_pil).unsqueeze(0).to('cuda:1' if torch.cuda.is_available() else 'cpu')
                    with torch.no_grad():
                        output = class_model(cropped_tensor)
                        _, predicted = torch.max(output, 1)
                        class_id = predicted.item()
                    area = (int(b[3] - b[1])) * (int(b[2] - b[0]))
                    print(f"Class ID: {class_id} - Area: {area}")
                    
                    if class_id == 32 and area > 100:
                        print("Red light detected - Stopping")
                        is_accelerated = True
                    elif class_id == 34 and area > 2000:
                        print("Left turn detected - Turning")

        except Exception as e:
            print(f"YOLO error: {e}")

        # === Şerit Algılama ===
        try:
            img = letterbox(frame, img_size, stride=stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            [pred, anchor_grid], seg, ll = model(img)
            ll_seg_mask = lane_line_mask(ll)

            contours, _ = cv2.findContours((ll_seg_mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            lanes, close_lanes = [[0, 0], [0, 0]], [999, 999]
            height, width, _ = frame.shape
            mask_width = ll_seg_mask.shape[1]
            mean = mask_width / 2

            for contour in contours:
                if cv2.contourArea(contour) < 50:
                    continue
                points = contour[:, 0, :]
                if len(contour) < 200:
                    continue
                min_x = np.mean(points[:, 0])
                min_y = np.mean(points[:, 1])
                max_y = np.max(points[:, 1])
                dist = abs(min_x - mean) + abs(max_y - ll_seg_mask.shape[0])
                if dist < close_lanes[0]:
                    close_lanes[1], lanes[1] = close_lanes[0], lanes[0]
                    close_lanes[0], lanes[0] = dist, [min_x, min_y]
                elif dist < close_lanes[1]:
                    close_lanes[1], lanes[1] = dist, [min_x, min_y]

            midpoint_x = int((lanes[0][0] + lanes[1][0]) / 2 * (width / mask_width))
            midpoint_y = int((lanes[0][1] + lanes[1][1]) / 2 * (height / ll_seg_mask.shape[0]))
            cv2.circle(frame, (midpoint_x, midpoint_y), 7, (255, 255, 255), -1)

            control = pid((width // 2) - midpoint_x)
            print(f"Control (angular offset): {control:.2f}")
        except Exception as e:
            print(f"Lane detection error: {e}")

        # Görüntüyü göster
        cv2.imshow("Result", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
