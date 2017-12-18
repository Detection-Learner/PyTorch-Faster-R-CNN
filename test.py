from lib.layers.roi_data_layer.image_loader import ImageLoader, detection_collate
import torch

loader = ImageLoader('img_list.txt', 'ann_list.txt')

train_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=10, shuffle=True,
        num_workers=3, collate_fn=detection_collate, pin_memory=True)

for i, (imgs, im_infos, gt_boxes) in enumerate(train_loader):
    print imgs.shape
    print len(im_infos), [len(im_info) for im_info in im_infos]
    print len(gt_boxes), [len(gt_box) for gt_box in gt_boxes]

