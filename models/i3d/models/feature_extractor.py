import torch
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pdb
from resnet import i3_res50_nl

class VideoDataset(Dataset):
    def __init__(self, video_path, num_crops):
        self.video_path = video_path
        self.clip_length = 16
        self.num_clips = int(np.round(cv2.VideoCapture(self.video_path).get(cv2.CAP_PROP_FRAME_COUNT) / self.clip_length))
        self.num_crops = num_crops

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.video_path)
        start_frame = idx * self.clip_length
        end_frame = start_frame + self.clip_length
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for _ in range(self.clip_length):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                break
        cap.release()
        return np.stack(frames)
    
def extract_crops(clip, num_crops=10, crop_size=(224, 224)):
    _, H, W, _ = clip[0].shape
    crops = []
    for _ in range(num_crops):
        y = np.random.randint(0, H - crop_size[0])
        x = np.random.randint(0, W - crop_size[1])
        crop_frames = [frame[y:y + crop_size[0], x:x + crop_size[1]] for frame in clip]
        crops.append(crop_frames)
    return crops

def extract_features(video_path, num_crops=10, feature_dim=2048):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained I3D model
    weights = torchvision.models.video.Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
    model = torchvision.models.video.swin3d_b(weights=weights)
    model.head = torch.nn.Linear(1024, 2048, bias=True)
    model.to(device)
    model.eval()

    # Define video transformation
    transform = weights.transforms()
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(224),
    ])

    # Create video dataset
    dataset = VideoDataset(video_path, num_crops)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Extract features
    clip_features = []
    with torch.no_grad():
        for idx, video_clip in enumerate(dataloader):
            # video_clip = torch.FloatTensor(video_clip)
            video_clip = video_clip.permute(0, 1, 4, 2, 3)
            video_clip = video_clip.to(device)
            crop_features = []
            for _ in range(10):
                video_crop = torchvision.transforms.RandomCrop(224)(video_clip.float()).permute(0, 2, 1, 3, 4)
                crop_feature = model(video_crop)
                crop_features.append(crop_feature.squeeze(0).cpu().numpy())
            clip_feature = np.stack(crop_features)
            clip_features.append(clip_feature)

    features = np.stack(clip_features)
    return features

if __name__ == "__main__":
    video_path = "/home/stark/Video-Anomaly-Detection/models/i3d/data/custom_dataset/training/videos/Normal_Videos_704_x264.mp4"
    features = extract_features(video_path)
    print("Feature shape:", features.shape)
