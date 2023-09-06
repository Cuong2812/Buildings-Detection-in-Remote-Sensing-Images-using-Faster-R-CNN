import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Compose, Normalize, PILToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from PIL import ImageFilter, Image
import albumentations as A
import torchvision.transforms.functional as F
from skimage import io, transform
import cv2
import os
import random
import shutil
import json
import geopandas as gpd
from torchvision.ops import box_convert
from gbbox import LineString






class BuildingDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.images_dir = os.path.join(data_dir, 'images')
        self.masks_dir = os.path.join(data_dir, 'masks')
        self.image_filenames = sorted(os.listdir(self.images_dir))
        self.mask_filenames = sorted(os.listdir(self.masks_dir))
        self.transforms = transforms

        
    def __getitem__(self, idx):
        # Load the image and the mask
        image_path = os.path.join(self.images_dir, self.image_filenames[idx])
        # Check image 0 
        mask_path = os.path.join(self.masks_dir, self.mask_filenames[idx])
        
        mask = default_loader(mask_path)
        image = default_loader(image_path)

        image = F.resize(image, (406, 438), interpolation=Image.BILINEAR)
        mask = F.resize(mask, (406, 438), interpolation=Image.BILINEAR)
        # Apply the transforms
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return image, mask

    def __len__(self):
        return len(self.image_filenames)

def collate_fn(batch):
    return tuple(zip(*batch))


class MyDataset(Dataset):
    def __init__(self, dataframe_dir, image_dir,  transforms=None): 
        self.df = dataframe_dir
        self.image_dir = image_dir
        self.transforms = transforms
        self.imageList = sorted(os.listdir(self.image_dir), key = lambda x: int(x[19:-4]))
        
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.imageList[idx])
        image_id = self.imageList[idx][19:-4]
        # image = cv2.imread(self.image_dir+'/'+image_path)
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image = np.transpose(image,(2,1,0))
        image = image.astype(np.float32)
        image = torch.tensor(image)
        # print(image.shape.reshape(-1,2,0))
        # image = image.astype(np.float32)

        # image = torchvision.transforms.functional.pil_to_tensor(image)

        dataframe_path = os.path.join(self.df, self.imageList[idx].replace('3band','Geo').replace('.tif','.geojson'))
        df = gpd.read_file(dataframe_path)

        # image = F.resize(image, (406, 438), interpolation= Image.BILINEAR)

        polygons = df['geometry'].tolist()
        labels = []


        def degenerate_to_bbox(bbox):
            """Converts a degenerate box to a regular bounding box by expanding the width and height."""
            if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
                epsilon = 0.0001
                x_min, y_min, x_max, y_max = bbox
                if x_min > x_max:
                    x_min -= epsilon
                    x_max += epsilon
                if y_min < y_max:
                    y_min -= epsilon
                    y_max += epsilon
                bbox = [x_min, y_min, x_max, y_max]
            return bbox

        def calculate_bbox(coords):
            """
            Calculate the bounding box of a Polygon item using np.min and np.max
            
            Args:
            poly (list): A list of linear rings, where each linear ring is a list of (lon, lat) pairs
            
            Returns:
            tuple: A tuple of four values (min_lon, min_lat, max_lon, max_lat)
            """
            lons = [p[0] for p in coords]
            lats = [p[1] for p in coords]
            min_lon, min_lat = np.min(lons), np.min(lats)
            max_lon, max_lat = np.max(lons), np.max(lats)
            return (min_lon, min_lat, max_lon, max_lat)

        def checkbbox(bbox):
            min_x, min_y, max_x, max_y = bbox
            width = max_x - min_x
            height = max_y - min_y
            box_area = width * height
            # Ignore smaller bounding boxes
            if box_area > 0:
                return False
                # Here you can retain these bounding boxes 
            else:
                return True

        def convert_to_negative(tup):
            return tuple(-1 * x for x in tup)


        



        for polygon in polygons:
            new = polygon.bounds
            if(checkbbox(new)):
                labels.append(new)
            else:
                labels.append(convert_to_negative(new))
        boxes = np.array(labels)

        areas = []
        for bbox in boxes:
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            areas.append(area)
        area = areas

        # boxes = normalize_bbox(boxes)

        area = torch.as_tensor(area)

                # there is only one class
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int8)
        target = {}
        thing = torchvision.ops.box_convert((torch.FloatTensor(boxes)), in_fmt='xywh',out_fmt='xyxy')
        target['boxes'] = thing
        target['labels'] = labels
        target['image_id'] = torch.tensor([int(image_id)], dtype=torch.int64)
        target['area'] = area
        target['iscrowd'] = iscrowd
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            image = sample['image']
            
            target['boxes'] =  torch.tensor(sample['bboxes']).float()
            return image, target, image_id

        return image, target, image_id


    def __len__(self):
        return len(self.imageList)


def medianFilter(x):
    return x.filter(ImageFilter.MedianFilter(size=3))
def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_train_transform():
    return A.Compose(
        [
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                           contrast_limit=0.2, p=0.9),
            ],p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.1),
            A.VerticalFlip(p=0.1),
            A.Resize(height=256, width=256, p=1),
            PILToTensor(),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )


def get_valid_transform():
    return A.Compose(
        [
            A.Resize(height=256, width=256, p=1.0),
            PILToTensor(),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )


if __name__ == '__main__':
    transforms = Compose([
        # lambda x: x.filter(ImageFilter.MedianFilter(size=3)),  # apply median filter
        # transforms.ToTensor(),
        # transforms.Resize((406, 438)),  # add this line to resize images
            transforms.ToTensor(),
            transforms.Resize((406,438)),
    ])

    # Define the dataset and dataloader for the training set
    
    train_dataset = MyDataset(image_dir = 'buildings/3bandData/splitData/train/images', dataframe_dir = 'buildings/3bandData/splitData/train/masks', transforms=get_train_transform())
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
    # Define the dataset and dataloader for the validation set
    val_dataset = MyDataset(image_dir = 'buildings/3bandData/splitData/val/images', dataframe_dir = 'buildings/3bandData/splitData/val/masks', transforms=get_valid_transform())
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)
    # Define the dataset and dataloader for the test set
    test_dataset = MyDataset(image_dir = 'buildings/3bandData/splitData/test/images', dataframe_dir = 'buildings/3bandData/splitData/test/masks', transforms=transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)

    class Averager:
        def __init__(self):
            self.current_total = 0.0
            self.iterations = 0.0

        def send(self, value):
            self.current_total += value
            self.iterations += 1

        @property
        def value(self):
            if self.iterations == 0:
                return 0
            else:
                return 1.0 * self.current_total / self.iterations

        def reset(self):
            self.current_total = 0.0
            self.iterations = 0.0

    def calculate_iou(gt, pr, form='pascal_voc') -> float:
        """Calculates the Intersection over Union.

        Args:
            gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
            pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box
            form: (str) gt/pred coordinates format
                - pascal_voc: [xmin, ymin, xmax, ymax]
                - coco: [xmin, ymin, w, h]
        Returns:
            (float) Intersection over union (0.0 <= iou <= 1.0)
        """
        if form == 'coco':
            gt = gt.copy()
            pr = pr.copy()

            gt[2] = gt[0] + gt[2]
            gt[3] = gt[1] + gt[3]
            pr[2] = pr[0] + pr[2]
            pr[3] = pr[1] + pr[3]

        # Calculate overlap area
        dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1
        
        if dx < 0:
            return 0.0
        dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1

        if dy < 0:
            return 0.0

        overlap_area = dx * dy

        # Calculate union area
        union_area = (
                (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +
                (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -
                overlap_area
        )

        return overlap_area / union_area


    def find_best_match(gts, pred, pred_idx, threshold = 0.5, form = 'pascal_voc', ious=None) -> int:
        """Returns the index of the 'best match' between the
        ground-truth boxes and the prediction. The 'best match'
        is the highest IoU. (0.0 IoUs are ignored).

        Args:
            gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
            pred: (List[Union[int, float]]) Coordinates of the predicted box
            pred_idx: (int) Index of the current predicted box
            threshold: (float) Threshold
            form: (str) Format of the coordinates
            ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

        Return:
            (int) Index of the best match GT box (-1 if no match above threshold)
        """
        best_match_iou = -np.inf
        best_match_idx = -1
        for gt_idx in range(len(gts)):
            
            if gts[gt_idx][0] < 0:
                # Already matched GT-box
                continue
            
            iou = -1 if ious is None else ious[gt_idx][pred_idx]

            if iou < 0:
                iou = calculate_iou(gts[gt_idx], pred, form=form)
                
                if ious is not None:
                    ious[gt_idx][pred_idx] = iou

            if iou < threshold:
                continue

            if iou > best_match_iou:
                best_match_iou = iou
                best_match_idx = gt_idx

        return best_match_idx


    def calculate_image_precision(gts, preds, thresholds = (0.5, ), form = 'coco') -> float:
        """Calculates image precision.

        Args:
            gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
            preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
                sorted by confidence value (descending)
            thresholds: (float) Different thresholds
            form: (str) Format of the coordinates

        Return:
            (float) Precision
        """
        n_threshold = len(thresholds)
        image_precision = 0.0
        
        ious = np.ones((len(gts), len(preds))) * -1
        # ious = None

        for threshold in thresholds:
            precision_at_threshold = calculate_precision(gts.copy(), preds, threshold=threshold,
                                                        form=form, ious=ious)
            image_precision += precision_at_threshold / n_threshold

        return image_precision

    def calculate_precision(gts, preds, threshold = 0.5, form = 'coco', ious=None) -> float:
        """Calculates precision for GT - prediction pairs at one threshold.

        Args:
            gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
            preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
                sorted by confidence value (descending)
            threshold: (float) Threshold
            form: (str) Format of the coordinates
            ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

        Return:
            (float) Precision
        """
        n = len(preds)
        tp = 0
        fp = 0
        
        for pred_idx in range(n):

            best_match_gt_idx = find_best_match(gts, preds[pred_idx], pred_idx,
                                                threshold=threshold, form=form, ious=ious)

            if best_match_gt_idx >= 0:
                # True positive: The predicted box matches a gt box with an IoU above the threshold.
                tp += 1
                # Remove the matched GT box
                gts[best_match_gt_idx] = -1
            else:
                # No match
                # False positive: indicates a predicted box had no associated gt box.
                fp += 1

        # False negative: indicates a gt box had no associated predicted box.
        fn = (gts.sum(axis=1) > 0).sum()
        return tp / (tp + fp + fn)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Create the Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(num_classes=2)
    model.to(device)

    # Define the optimizer and learning rate scheduler
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10


    # Define the loss function
    # loss_fn = torch.nn.CrossEntropyLoss()

    # Train the model
    loss_hist = Averager()
    t = 1
    valid_pred_min = 0.65 
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    itr = 1
    for epoch in range(num_epochs):
        loss_hist.reset()
        
        for images, targets, image_ids in train_dataloader:
            images = list(image for image in images)
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            
            loss_hist.send(loss_value)
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            if itr % 50 == 0:
                print(f"Iteration #{itr} loss: {loss_value}")
                
            itr += 1
        print(model.eval())
        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()
            
        print(f"Epoch #{epoch} loss: {loss_hist.value}")
        # Evaluate on the validation set
    # validation_image_precisions = []
    # iou_thresholds = [x for x in np.arange(0.5, 0.76, 0.05)]
    # for images, targets, image_ids in val_dataloader:       
    #     images = list(image.to(device) for image in images)
    #     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
    #     with torch.no_grad():
    #         outputs = model(images)
            
        
        
        
    #     for i, image in enumerate(images):
    #         boxes = outputs[i]['boxes'].data.cpu().numpy()
    #         scores = outputs[i]['scores'].data.cpu().numpy()
    #         gt_boxes = targets[i]['boxes'].cpu().numpy()
    #         preds_sorted_idx = np.argsort(scores)[::-1]
    #         preds_sorted = boxes[preds_sorted_idx]
    #         image_precision = calculate_image_precision(preds_sorted,
    #                                                     gt_boxes,
    #                                                     thresholds=iou_thresholds,
    #                                                     form='coco')
    #         validation_image_precisions.append(image_precision)

    # valid_prec = np.mean(validation_image_precisions)
    # print("Validation IOU: {0:.4f}".format(valid_prec))
              
       
    
    # #print training/validation statistics 
    # print('Epoch: {} \tTraining Loss: {:.6f}'.format(
    #     epoch, 
    #     train_loss
    # ))
        
    # ## TODO: save the model if validation precision has decreased
    # if valid_prec >= valid_pred_min:
    #     print('Validation precision increased({:.6f} --> {:.6f}).  Saving model ...'.format(
    #         valid_pred_min,
    #         valid_prec))
    #     torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')
    #     valid_pred_min = valid_prec



