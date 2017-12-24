import os
import sys
import time
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_PATH, '..'))
import torch
import torch.backends.cudnn as cudnn
import argparse
from lib.networks.Faster_RCNN import FasterRCNN
from lib.datasets import PascalData, detection_collate
from lib.utils.config import cfg, cfg_from_file
from lib.utils.transform_bbox import bbox_transform_inv
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
    parser.add_argument('--vocroot', dest='vocroot',
                        help='the path of Pascal VOC 2007 & 2012 data', type=str)
    parser.add_argument('--resume', dest='resume',
                        help='the path of resumed weights path', type=str, default=None)
    parser.add_argument('--weight_dir', dest='weight_dir',
                        help='the path to save model weights', type=str, default='../weights')
    parser.add_argument('--gpu', dest='gpu', nargs='+',
                        help='which gpu you want to use', default='0', type=str)

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
                        help='the value of display', default=100, type=int)
    parser.add_argument('--test_interval', dest='test_interval',
                        help='the value of test interval', default=1000, type=int)

    args = parser.parse_args()

    return args


def constructe_model(args):

    gpus = list(range(len(args.gpu.split(','))))
    if args.resume is not None:
        model = FasterRCNN(pretrained=False)
        print('Load weight from {}'.format(args.resume))
        model.load_state_dict(torch.load(args.resume))
        print('Done !')
    else:
        model = FasterRCNN(pretrained=True)
    # print model
    # FIXME: when use multi-GPUs with DataParallel, there will be a `module` front of model.
    #       Then the loss will not be used as `model.loss` but `model.module.loss` replace
    if len(gpus) == 1:
        model = model.cuda()
    else:
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    return model


def train_val(model, args):
    policy_parameter = {'gamma': 0.1, 'step_size': 50000}
    cudnn.benchmark = True

    train_loader = torch.utils.data.DataLoader(
        PascalData(VOCdevkitRoot=args.vocroot),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=detection_collate, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        PascalData(VOCdevkitRoot=args.vocroot, trainval=False),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=detection_collate, pin_memory=True)

    params, multiple = get_parameters(model, args.base_lr)

    optimizer = torch.optim.SGD(model.parameters(), args.base_lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_list = [AverageMeter()
                   for i in range((len(cfg.FEATURE_LAYERS) + 1) * 2)]

    end = time.time()

    iters = args.start_iter
    learning_rate = args.base_lr

    progress = ProgressBar()
    while iters < args.max_iter:

        for i, (imgs, im_infos, gt_boxes) in enumerate(train_loader):
            # print imgs.size()

            learning_rate = adjust_learning_rate(
                optimizer, iters, args.base_lr, policy_parameter, multiple=multiple)
            data_time.update(time.time() - end)

            input_var = torch.autograd.Variable(imgs.cuda())
            model(input_var, im_infos, gt_boxes)

            if cfg.USE_FPN == False:
                loss = model.loss + model.rpn.loss
                losses.update(loss.data[0], imgs.size(0))
                losses_list[0].update(
                    model.rpn.cross_entropy.data[0], imgs.size(0))
                losses_list[1].update(model.rpn.loss_box.data[0], imgs.size(0))
                losses_list[2].update(
                    model.cross_entropy.data[0], imgs.size(0))
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
                    print('rpn{0}_cross_entyopy = {loss1.avg:.8f} rpn{0}_loss_box = {loss2.avg:.8f}\n'.format(
                        i + 1, loss1=losses_list[i * 2], loss2=losses_list[i * 2 + 1]))
                print('cross_entropy = {loss1.avg:.8f} loss_box = {loss2.avg:.8f}\n'.format(
                    loss1=losses_list[-2], loss2=losses_list[-1]))
                batch_time.reset()
                data_time.reset()
                losses.reset()
                for i in range((len(cfg.FEATURE_LAYERS) + 1)):
                    losses_list[i].reset()

            if iters % args.test_interval == 0:
                print('Start Eval:')
                model.eval()
                val_losses = AverageMeter()
                for j, (imgs, im_infos, gt_boxes) in enumerate(val_loader):
                    input_var = torch.autograd.Variable(
                        imgs.cuda(), volatile=True)
                    cls_pred, bbox_delta, boxes = model(
                        input_var, im_infos, gt_boxes)

                    if cfg.USE_FPN == False:
                        loss = model.loss + model.rpn.loss
                    else:
                        loss = model.loss + model.fpn.loss

                    val_losses.update(loss.data[0], imgs.size(0))
                    progress.show_info(
                        index=j + 1, max_length=len(iter(val_loader)), loss=val_losses.avg)
                    # TODO: Compute the box, mAP and so on.
                    #       It will be very good if it could be visualized real-time.
                    #       Such as `crayon`: https://github.com/torrvision/crayon
                    # pred_boxes = bbox_transform_inv(
                    #     boxes[:, 1:5], bbox_delta.data.cpu().numpy().reshape(1, -1, 4 * cfg.NCLASSES))
                print('Eval Loss: {:.4f}'.format(val_losses.avg))
                if not os.path.exists(args.weight_dir):
                    os.system('mkdir {}'.format(args.weight_dir))
                torch.save(
                    model.state_dict(), '{0}/faster_rcnn_{1}.pth'.format(args.weight_dir, iters))

                model.train()

            if iters == args.max_iter:
                break


if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg.GPU_ID = 0  # args.gpu[0]
    print(cfg)
    print(args)

    model = constructe_model(args)

    train_val(model, args)
