import os
import cv2
import glob
import torch
import torchvision

import numpy as np
import scipy.io as scio

from collections import OrderedDict
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from utils.img_utils import get_foreground, normalize

transform = transforms.Compose([
    transforms.ToTensor(),
])

class VideoDataset(Dataset):
    def __init__(self, dataset_name, root, train=True, sequence_length=0, mode="last",
                 all_bboxes=None, patch_size=224, normalize=False, bboxes_extractions=False):
        super(VideoDataset, self).__init__()
        self.dataset_name = dataset_name
        self.root = os.path.join(root, dataset_name)
        self.train = train
        self.videos = OrderedDict()
        self.frame_addresses = []
        self.frame_video_idx = []
        self.sequence_length = sequence_length
        self.mode = mode
        self.all_bboxes = all_bboxes
        self.patch_size = patch_size
        self.normalize = normalize
        self.bboxes_extractions = bboxes_extractions
        self.__initialize()
        self.num_of_frames = len(self.frame_addresses)

    def __len__(self):
        return self.num_of_frames

    def __initialize(self):
        if self.train:
            self.root_frames = os.path.join(self.root, 'training', 'frames/*')
        else:
            self.root_frames = os.path.join(self.root, 'testing', 'frames/*')

        videos = glob.glob(self.root_frames)
        for i, video_path in enumerate(sorted(videos)):
            self.videos[i] = {}
            self.videos[i]['path'] = video_path
            self.videos[i]['frames'] = glob.glob(os.path.join(video_path, '*.jpg'))
            self.videos[i]['frames'].sort()
            self.frame_addresses += self.videos[i]['frames']
            self.videos[i]['length'] = len(self.videos[i]['frames'])
            self.frame_video_idx += [i] * self.videos[i]['length']

        if not self.train:
            self.get_gt()

    def get_gt(self):
        if self.dataset_name == 'ped2':
            mat_name = self.dataset_name + '.mat'
            mat_path = os.path.join(self.root, mat_name)
            abnormal_mat = scio.loadmat(mat_path, squeeze_me=True)['gt']
            self.all_gt = []
            for i, (_, video) in enumerate(self.videos.items()):
                length = video['length']
                sub_video_gt = np.zeros((length,), dtype=np.int8)
                one_abnormal = abnormal_mat[i]
                if one_abnormal.ndim == 1:
                    one_abnormal = one_abnormal.reshape((one_abnormal.shape[0], -1))
                for j in range(one_abnormal.shape[1]):
                    start = one_abnormal[0, j] - 1
                    end = one_abnormal[1, j]
                    sub_video_gt[start: end] = 1
                sub_video_gt = torch.from_numpy(sub_video_gt)
                self.all_gt.append(sub_video_gt)
            self.all_gt = torch.cat(self.all_gt, 0)

        elif self.dataset_name == 'avenue':
            ids = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
                   '18', '19', '20', '21']
            mat_name = self.dataset_name + '.mat'
            mat_path = os.path.join(self.root, mat_name)
            abnormal_mat = scio.loadmat(mat_path)
            self.all_gt = []
            for id in ids:
                self.all_gt.append(abnormal_mat[id][0])
            self.all_gt = np.concatenate(self.all_gt, axis=0)
            self.all_gt = torch.from_numpy(self.all_gt)

        elif self.dataset_name == 'shanghaitech':
            frame_mask = os.path.join(self.root, 'testing/test_frame_mask/*')
            frame_mask = glob.glob(frame_mask)
            frame_mask.sort()
            self.all_gt = []
            for i, (_, video) in enumerate(self.videos.items()):
                gt = torch.from_numpy(np.load(frame_mask[i]))
                self.all_gt.append(gt)
            self.all_gt = torch.cat(self.all_gt, 0)

    def get_sequence(self, index):
        """
        This function gets an index and returns a clip (of size self.sequence_length).
        """
        if index - self.sequence_length < 0:
            start_idx = 0
        else:
            start_idx = index - self.sequence_length

        if self.mode == "last":
            end_idx = index
            clip_length = self.sequence_length + 1  # future frame prediction
        else:   # Means that self.mode == "middle"
            if index + self.sequence_length > self.num_of_frames - 1:
                end_idx = self.num_of_frames - 1
            else:
                end_idx = index + self.sequence_length
            clip_length = 2 * self.sequence_length + 1

        main_frame_video_idx = self.frame_video_idx[index]
        clip_frames_video_idx = self.frame_video_idx[start_idx:end_idx + 1]
        need_border_padding = clip_length - len(clip_frames_video_idx)

        if need_border_padding > 0:
            if start_idx == 0:
                clip_frames_video_idx = [clip_frames_video_idx[0]] * need_border_padding + clip_frames_video_idx
            else:
                clip_frames_video_idx = clip_frames_video_idx + [clip_frames_video_idx[-1]] * need_border_padding

        all_frames_same_video = np.array(clip_frames_video_idx) - main_frame_video_idx
        offset = np.sum(all_frames_same_video)

        if all_frames_same_video[0] != 0 and all_frames_same_video[-1] != 0:    # Extreme condition that is not likely to happen
            print('The video is too short or the context frame number is too large!')
            raise NotImplementedError

        if need_border_padding == 0 and offset == 0:
            idx = [x for x in range(start_idx, end_idx + 1)]
            return idx
        else:
            if self.mode == 'last':
                if need_border_padding > 0 and np.abs(offset) > 0:
                    print('The video is too short or the context frame number is too large!')
                    raise NotImplementedError
                idx = [x for x in range(start_idx - offset, end_idx + 1)]
                idx = [idx[0]] * np.maximum(np.abs(offset), need_border_padding) + idx
                return idx
            else:   # Means that self.mode == "middle"
                if need_border_padding > 0 and np.abs(offset) > 0:
                    print('The video is too short or the context frame number is too large!')
                    raise NotImplementedError
                if offset > 0:
                    idx = [x for x in range(start_idx, end_idx - offset + 1)]
                    idx = idx + [idx[-1]] * np.abs(offset)
                    return idx
                elif offset < 0:
                    idx = [x for x in range(start_idx - offset, end_idx + 1)]
                    idx = [idx[0]] * np.abs(offset) + idx
                    return idx
                if need_border_padding > 0:
                    if start_idx == 0:
                        idx = [x for x in range(start_idx, end_idx + 1)]
                        idx = [idx[0]] * need_border_padding + idx
                        return idx
                    else:
                        idx = [x for x in range(start_idx, end_idx + 1)]
                        idx = idx + [idx[-1]] * need_border_padding
                        return idx

    def resize_batch(self, frame_idx_range, img_batch):
        img_batch_resized = []
        for i in range(len(frame_idx_range)):
            cur_img = img_batch[i]
            cur_img = cv2.resize(np.transpose(cur_img, [1, 2, 0]), (self.patch_size, self.patch_size))
            img_batch_resized.append(np.transpose(cur_img, [2, 0, 1]))
        img_batch = np.array(img_batch_resized)
        return img_batch


    def __getitem__(self, index):
        frame_idx_range = self.get_sequence(index=index)
        img_batch = []
        for i in frame_idx_range:
            # [h,w,c] -> [c,h,w] BGR
            cur_img = np.transpose(cv2.imread(self.frame_addresses[i]), [2, 0, 1])
            # cur_img = transform(Image.open(self.frame_addresses[i]).convert('RGB'))
            img_batch.append(cur_img)
        img_batch = np.array(img_batch)

        if self.all_bboxes is not None:
            # cropping
            if len(self.all_bboxes[index]) > 0:
                img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[index], patch_size=self.patch_size)
            else:
                img_batch, flows_batch = self.resize_batch(frame_idx_range, img_batch)
                img_batch = img_batch[None]

        elif not self.bboxes_extractions:
            img_batch = self.resize_batch(frame_idx_range, img_batch)
        img_batch = torch.from_numpy(img_batch)  # [num_bboxes, sequence_length, C, patch_size, patch_size]
        if self.normalize:
            img_batch = img_batch.to(dtype=torch.get_default_dtype()).div(255)  # scaling to [0, 1]
            img_batch = normalize(img_batch) if self.normalize else img_batch

        if self.train:
            return img_batch, torch.zeros(1)
        return img_batch, self.all_gt[index]

class VideoDatasetWithFlows(Dataset):
    def __init__(self, dataset_name, root, train=True, sequence_length=0, mode="last",
                 all_bboxes=None, patch_size=224, normalize=False, bboxes_extractions=False):
        super(VideoDatasetWithFlows, self).__init__()
        self.dataset_name = dataset_name
        self.root = os.path.join(root, dataset_name)
        self.train = train
        self.videos = OrderedDict()
        self.flows = OrderedDict()
        self.frame_addresses = []
        self.frame_addresses_flows = []
        self.frame_video_idx = []
        self.sequence_length = sequence_length
        self.mode = mode
        self.all_bboxes = all_bboxes
        self.patch_size = patch_size
        self.normalize = normalize
        self.bboxes_extractions = bboxes_extractions
        self.__initialize()
        self.num_of_frames = len(self.frame_addresses)


    def __len__(self):
        return self.num_of_frames

    def __initialize(self):
        if self.train:
            self.root_frames = os.path.join(self.root, 'training', 'frames/*')
            self.root_frames_flows = os.path.join(self.root, 'training', 'flows/*')
        else:
            self.root_frames = os.path.join(self.root, 'testing', 'frames/*')
            self.root_frames_flows = os.path.join(self.root, 'testing', 'flows/*')

        videos = glob.glob(self.root_frames_flows)
        for i, video_path in enumerate(sorted(videos)):
            self.flows[i] = {}
            self.flows[i]['path'] = video_path
            self.flows[i]['frames'] = glob.glob(os.path.join(video_path, '*.npy'))
            self.flows[i]['frames'].sort()
            self.frame_addresses_flows += self.flows[i]['frames']
            self.flows[i]['length'] = len(self.flows[i]['frames'])

        videos = glob.glob(self.root_frames)
        for i, video_path in enumerate(sorted(videos)):
            self.videos[i] = {}
            self.videos[i]['path'] = video_path
            self.videos[i]['frames'] = glob.glob(os.path.join(video_path, '*.jpg'))
            self.videos[i]['frames'].sort()
            self.frame_addresses += self.videos[i]['frames']
            self.videos[i]['length'] = len(self.videos[i]['frames'])
            self.frame_video_idx += [i] * self.videos[i]['length']

        if not self.train:
            self.get_gt()

    def get_gt(self):
        if self.dataset_name == 'ped2':
            mat_name = self.dataset_name + '.mat'
            mat_path = os.path.join(self.root, mat_name)
            abnormal_mat = scio.loadmat(mat_path, squeeze_me=True)['gt']
            self.all_gt = []
            for i, (_, video) in enumerate(self.videos.items()):
                length = video['length']
                sub_video_gt = np.zeros((length,), dtype=np.int8)
                one_abnormal = abnormal_mat[i]
                if one_abnormal.ndim == 1:
                    one_abnormal = one_abnormal.reshape((one_abnormal.shape[0], -1))
                for j in range(one_abnormal.shape[1]):
                    start = one_abnormal[0, j] - 1
                    end = one_abnormal[1, j]
                    sub_video_gt[start: end] = 1
                sub_video_gt = torch.from_numpy(sub_video_gt)
                self.all_gt.append(sub_video_gt)
            self.all_gt = torch.cat(self.all_gt, 0)

        elif self.dataset_name == 'avenue':
            ids = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
                   '18', '19', '20', '21']
            mat_name = self.dataset_name + '.mat'
            mat_path = os.path.join(self.root, mat_name)
            abnormal_mat = scio.loadmat(mat_path)
            self.all_gt = []
            for id in ids:
                self.all_gt.append(abnormal_mat[id][0])
            self.all_gt = np.concatenate(self.all_gt, axis=0)
            self.all_gt = torch.from_numpy(self.all_gt)

        elif self.dataset_name == 'shanghaitech':
            frame_mask = os.path.join(self.root, 'testing/test_frame_mask/*')
            frame_mask = glob.glob(frame_mask)
            frame_mask.sort()
            self.all_gt = []
            for i, (_, video) in enumerate(self.videos.items()):
                gt = torch.from_numpy(np.load(frame_mask[i]))
                self.all_gt.append(gt)
            self.all_gt = torch.cat(self.all_gt, 0)

    def get_sequence(self, index):
        """
        This function gets an index and returns a clip (of size self.sequence_length).
        """
        if index - self.sequence_length < 0:
            start_idx = 0
        else:
            start_idx = index - self.sequence_length

        if self.mode == "last":
            end_idx = index
            clip_length = self.sequence_length + 1  # future frame prediction
        else:   # Means that self.mode == "middle"
            if index + self.sequence_length > self.num_of_frames - 1:
                end_idx = self.num_of_frames - 1
            else:
                end_idx = index + self.sequence_length
            clip_length = 2 * self.sequence_length + 1

        main_frame_video_idx = self.frame_video_idx[index]
        clip_frames_video_idx = self.frame_video_idx[start_idx:end_idx + 1]
        need_border_padding = clip_length - len(clip_frames_video_idx)

        if need_border_padding > 0:
            if start_idx == 0:
                clip_frames_video_idx = [clip_frames_video_idx[0]] * need_border_padding + clip_frames_video_idx
            else:
                clip_frames_video_idx = clip_frames_video_idx + [clip_frames_video_idx[-1]] * need_border_padding

        all_frames_same_video = np.array(clip_frames_video_idx) - main_frame_video_idx
        offset = np.sum(all_frames_same_video)

        if all_frames_same_video[0] != 0 and all_frames_same_video[-1] != 0:    # Extreme condition that is not likely to happen
            print('The video is too short or the context frame number is too large!')
            raise NotImplementedError

        if need_border_padding == 0 and offset == 0:
            idx = [x for x in range(start_idx, end_idx + 1)]
            return idx
        else:
            if self.mode == 'last':
                if need_border_padding > 0 and np.abs(offset) > 0:
                    print('The video is too short or the context frame number is too large!')
                    raise NotImplementedError
                idx = [x for x in range(start_idx - offset, end_idx + 1)]
                idx = [idx[0]] * np.maximum(np.abs(offset), need_border_padding) + idx
                return idx
            else:   # Means that self.mode == "middle"
                if need_border_padding > 0 and np.abs(offset) > 0:
                    print('The video is too short or the context frame number is too large!')
                    raise NotImplementedError
                if offset > 0:
                    idx = [x for x in range(start_idx, end_idx - offset + 1)]
                    idx = idx + [idx[-1]] * np.abs(offset)
                    return idx
                elif offset < 0:
                    idx = [x for x in range(start_idx - offset, end_idx + 1)]
                    idx = [idx[0]] * np.abs(offset) + idx
                    return idx
                if need_border_padding > 0:
                    if start_idx == 0:
                        idx = [x for x in range(start_idx, end_idx + 1)]
                        idx = [idx[0]] * need_border_padding + idx
                        return idx
                    else:
                        idx = [x for x in range(start_idx, end_idx + 1)]
                        idx = idx + [idx[-1]] * need_border_padding
                        return idx

    def resize_batch(self, frame_idx_range, img_batch, flows_batch):
        img_batch_resized = []
        flows_batch_resized = []
        for i in range(len(frame_idx_range)):
            cur_img = img_batch[i]
            cur_flow = flows_batch[i]

            cur_img = cv2.resize(np.transpose(cur_img, [1, 2, 0]), (self.patch_size, self.patch_size))
            cur_flow = cv2.resize(np.transpose(cur_flow, [1, 2, 0]), (self.patch_size, self.patch_size))

            img_batch_resized.append(np.transpose(cur_img, [2, 0, 1]))
            flows_batch_resized.append(np.transpose(cur_flow, [2, 0, 1]))

        img_batch = np.array(img_batch_resized)
        flows_batch = np.array(flows_batch_resized)
        return img_batch, flows_batch

    def __getitem__(self, index):
        frame_idx_range = self.get_sequence(index=index)
        img_batch = []
        flows_batch = []
        for i in frame_idx_range:
            # [h,w,c] -> [c,h,w] BGR
            cur_img = np.transpose(cv2.imread(self.frame_addresses[i]), [2, 0, 1])
            cur_flow = np.transpose(np.load(self.frame_addresses_flows[i]), [2, 0, 1])
            img_batch.append(cur_img)
            flows_batch.append(cur_flow)
        img_batch = np.array(img_batch)
        flows_batch = np.array(flows_batch)

        if self.all_bboxes is not None:
            # cropping
            if len(self.all_bboxes[index]) > 0:
                img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[index], patch_size=self.patch_size)
                flows_batch = get_foreground(img=flows_batch, bboxes=self.all_bboxes[index], patch_size=self.patch_size)
            else:
                img_batch, flows_batch = self.resize_batch(frame_idx_range, img_batch, flows_batch)
                img_batch = img_batch[None]
                flows_batch = flows_batch[None]

        elif not self.bboxes_extractions:
            img_batch, flows_batch = self.resize_batch(frame_idx_range, img_batch, flows_batch)
        img_batch = torch.from_numpy(img_batch)  # [num_bboxes, sequence_length, C, patch_size, patch_size]
        flows_batch = torch.from_numpy(flows_batch)  # [num_bboxes, sequence_length, C, patch_size, patch_size]

        if self.normalize:
            img_batch = img_batch.to(dtype=torch.get_default_dtype()).div(255)  # scaling to [0, 1]
            img_batch = normalize(img_batch) if self.normalize else img_batch

        if self.train:
            return img_batch, flows_batch, torch.zeros(1)
        return img_batch, flows_batch, self.all_gt[index]

def get_video_dataset(args, root):
    all_bboxes_train = np.load(os.path.join(root, args.dataset_name, '%s_bboxes_train.npy' % args.dataset_name),
                               allow_pickle=True)
    all_bboxes_test = np.load(os.path.join(root, args.dataset_name, '%s_bboxes_test.npy' % args.dataset_name),
                              allow_pickle=True)

    if args.dataset_name == 'shanghaitech': # ShanghaiTech normalization
        all_bboxes_train_classes = np.load(os.path.join(root, args.dataset_name, f'{args.dataset_name}_bboxes_train_classes.npy'),
                                   allow_pickle=True)
        all_bboxes_test_classes = np.load(os.path.join(root, args.dataset_name, f'{args.dataset_name}_bboxes_test_classes.npy'),
                                  allow_pickle=True)
    else:
        all_bboxes_train_classes = None
        all_bboxes_test_classes = None

    train_dataset = VideoDatasetWithFlows(dataset_name=args.dataset_name, root=root,
                                          train=True, sequence_length=0, all_bboxes=all_bboxes_train, normalize=True)
    test_dataset = VideoDatasetWithFlows(dataset_name=args.dataset_name, root=root,
                                         train=False, sequence_length=0, all_bboxes=all_bboxes_test, normalize=True)
    
    return all_bboxes_train_classes, all_bboxes_test_classes,\
           train_dataset, test_dataset