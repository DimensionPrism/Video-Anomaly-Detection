from src.multi_models.yolov5.train import parse_opt
from src.multi_models.yolov5.train import main as train_yolov5

if __name__ == "__main__":
    opt = parse_opt()
    train_yolov5(opt)