import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/PL-YOLO.yaml')
    # model.load('yolo11n.pt') # loading pretrain weights
    model.train(data='ultralytics-yolo11-pomegranate/ultralytics-yolo11-main/dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=16,
                close_mosaic=0,
                workers=4,
                device='0',
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                # resume=True,
                amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )