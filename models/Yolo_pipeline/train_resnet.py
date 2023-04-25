import argparse
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from src.utils.preprocess_utils import preprocess_fer2013
from src.multi_models.resnet.model import resnet, EmotionClassifier

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./datasets/FER2013/fer2013_1.csv', help='fer2013 csv path')
    parser.add_argument('--weights', type=str, default='./model_results/resnet_weights/resnet50_scratch_weight.pkl', help='pretrained resnet50 scratch weight pkl path')
    parser.add_argument('--project', type=str, default='./model_results/resnet_runs/', help='save to project')

    return parser.parse_args()


if __name__ == "__main__":
    cfg = config()
    processed_fer2013 = preprocess_fer2013(cfg.data)
    resnet50 = resnet(cfg.weights)

    emotion_classifier = EmotionClassifier(model=resnet50, processed_fer2013=processed_fer2013, batch_size=64)
    trainer = pl.Trainer(
        logger=pl_loggers.CSVLogger(
            save_dir=cfg.project
        ),
        callbacks=[
            ModelCheckpoint(
                dirpath=cfg.project,
                monitor='val_accuracy',
                filename='fer2013_{val_accuracy:.3f}',
                mode='max'
            ),
        ],
        gpus=1,
        max_epochs=50,
    )
    trainer.fit(emotion_classifier)
    trainer.test(emotion_classifier)