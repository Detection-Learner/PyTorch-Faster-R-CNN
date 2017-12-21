import os
import sys
import time
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_PATH, '..'))
import torch
import torch.backends.cudnn as cudnn
import argparse
from lib.networks.Faster_RCNN import FasterRCNN
from lib.layers.roi_data_layer.image_loader import ImageLoader, detection_collate
from lib.utils.config import cfg, cfg_from_file
from utils import *
np.set_printoptions(threshold=np.inf)

def parse_args():
    """
    """
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN')
    parser.add_argument('--cfg', dest='cfg_file',
            help='optional config file', default=None, type=str)
    parser.add_argument('--workers', dest='workers',
            help='the number of workers to read data', default=6, type=int)
    parser.add_argument('--batch_size', dest='batch_size',
            help='the batch size of input', default=1, type=int)
    parser.add_argument('--train_img_list', dest='train_img_list',
            help='the path of file, saved the path of training img', type=str)
    parser.add_argument('--train_ann_list', dest='train_ann_list',
            help='the path of file, saved the path of training annotations', type=str)
    parser.add_argument('--val_img_list', dest='val_img_list',
            help='the path of file, saved the path of validation img', default=None, type=str)
    parser.add_argument('--val_ann_list', dest='val_ann_list',
            help='the path of file, saved the path of validation annotations', default=None, type=str)
    parser.add_argument('--gpu', dest='gpu', nargs='+',
            help='optional config file', default=[0], type=int)

    parser.add_argument('--lr', dest='base_lr',
            help='the initial learning rate', default=0.01, type=float)
    parser.add_argument('--momentum', dest='momentum',
            help='the value of momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', dest='weight_decay',
            help='the value of weight decay', default=0.0005, type=float)
    parser.add_argument('--start_iter', dest='start_iter',
            help='the start iteration', default=0, type=int)
    parser.add_argument('--max_iter', dest='max_iter',
            help='the max iterations', default=100000, type=int)
    parser.add_argument('--display', dest='display',
            help='the value of display', default=1, type=int)
    parser.add_argument('--test_interval', dest='test_interval',
            help='the value of test interval', default=1000, type=int)

    args = parser.parse_args()

    return args

def constructe_model():

    model = FasterRCNN(pretrained=True)
    print model
    model = model.cuda()

    return model

def train_val(model, args):

    cudnn.benchmark = True

    train_loader = torch.utils.data.DataLoader(
            ImageLoader(args.train_img_list, args.train_ann_list),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, collate_fn=detection_collate, pin_memory=True)

    if args.val_img_list is not None:
        val_loader = torch.utils.data.DataLoader(
                ImageLoader(args.val_img_list, args.val_ann_list),
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, collate_fn=detection_collate, pin_memory=True)

    params, multiple = get_parameters(model, args.base_lr)

    optimizer = torch.optim.SGD(model.parameters(), args.base_lr, momentum=args.momentum,
            weight_decay=args.weight_decay)

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_list = [AverageMeter() for i in range((len(cfg.FEATURE_LAYERS) + 1) * 2)]

    end = time.time()

    iters = args.start_iter
    learning_rate = args.base_lr
    while iters < args.max_iter:

        for i, (imgs, im_infos, gt_boxes) in enumerate(train_loader):
            print imgs.size()

            # learning_rate = adjust_learning_rate(optimizer, iters, args.base_lr, multiple=multiple)
            data_time.update(time.time() - end)

            input_var = torch.autograd.Variable(imgs.cuda())
            model(input_var, im_infos, gt_boxes)

            if cfg.USE_FPN == False:
                loss = model.loss + model.rpn.loss
                losses.update(loss.data[0], imgs.size(0))
                losses_list[0].update(model.rpn.cross_entropy.data[0], imgs.size(0))
                losses_list[1].update(model.rpn.loss_box.data[0], imgs.size(0))
                losses_list[2].update(model.cross_entropy.data[0], imgs.size(0))
                losses_list[3].update(model.bbox_loss.data[0], imgs.size(0))
            else:
                loss = model.loss + model.fpn.loss
                losses.update(loss.data[0], imgs.size(0))
                for i, rpn in enumerate(model.fpn.RPN_Units):
                    losses_list[i * 2].update(rpn.cross_entropy, imgs.size[0])
                    losses_list[i * 2 + 1].update(rpn.loss_box, imgs.size(0))
                losses_list[-2].update(model.cross_entropy, imgs.size(0))
                losses_list[-1].update(model.bbox_loss, imgs.size(0))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            iters += 1
            if iters % args.display == 0:
                print('Train Iteration: {0}\t'
                    'Time {batch_time.sum:.3f}s / {1} iters, {batch_time.avg:.3f}s / 1 iter\t'
                    'Data load {data_time.sum:.3f}s / {1} iters, {data_time.avg:3f}s / 1 iter\n'
                    'Learning rate = {2}\n'
                    'Loss = {loss.val:.8f} ave = {loss.avg:.8f}\n'.format(
                    iters, args.display, learning_rate, batch_time=batch_time, data_time=data_time, loss=losses))
                for i in range((len(cfg.FEATURE_LAYERS))):
                    print('rpn{0}_cross_entyopy = {loss1.avg:.8f} rpn{0}_loss_box = {loss2.avg:.8f}\n'.format(i + 1, loss1=losses_list[i * 2], loss2=losses_list[i * 2 + 1]))
                print('cross_entropy = {loss1.avg:.8f} loss_box = {loss2.avg:.8f}\n'.format(loss1=losses_list[-2], loss2=losses_list[-1]))
                batch_time.reset()
                data_time.reset()
                losses.reset()
                for i in range((len(cfg.FEATURE_LAYERS) + 1)):
                    losses_list[i].reset()
            
            if args.val_img_list is not None and iters % args.test_interval == 0:
                model.eval()
                for j, (imgs, im_infos, gt_boxes) in enumerate(val_loader):
                    input_var = torch.autograd.Variable(imgs.cuda(), volatile=True)
                    model(input_var, im_infos, gt_boxes)

                model.train()

            if iters == args.max_iter:
                break

if __name__ == '__main__':

    args = parse_args()
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg.GPU_ID = 0#args.gpu[0]
    print cfg

    model = constructe_model()

    train_val(model, args)
