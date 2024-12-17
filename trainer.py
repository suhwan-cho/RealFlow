from utils import AverageMeter, get_iou
import torch


class Trainer(object):
    def __init__(self, model, optimizer, train_loader, val_set, save_name, save_step, val_step):
        self.model = model.cuda()
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_set = val_set
        self.save_name = save_name
        self.save_step = save_step
        self.val_step = val_step
        self.epoch = 1
        self.best_score = 1
        self.score = 1
        self.stats = {'loss': AverageMeter(), 'iou': AverageMeter()}

    def train(self, max_epochs):
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch
            self.train_epoch()
            if self.epoch % self.save_step == 0:
                print('saving checkpoint\n')
                self.save_checkpoint()
            if self.score < self.best_score:
                print('new best checkpoint, after epoch {}\n'.format(self.epoch))
                self.save_checkpoint(alt_name='best')
                self.best_score = self.score
        print('finished training!\n', flush=True)

    def train_epoch(self):

        # train
        self.model.train()
        self.cycle_dataset(mode='train')

        # val
        self.model.eval()
        if self.epoch % self.val_step == 0:
            if self.val_set is not None:
                with torch.no_grad():
                    self.score = self.cycle_dataset(mode='val')

        # update stats
        for stat_value in self.stats.values():
            stat_value.new_epoch()

    def cycle_dataset(self, mode):
        if mode == 'train':
            for vsod_data in self.train_loader:
                imgs = vsod_data['imgs'].cuda()
                flows = vsod_data['flows'].cuda()
                masks = vsod_data['masks'].cuda()
                B, L, _, H, W = imgs.size()

                # model run
                vsod_out = self.model(imgs, flows)
                loss = torch.nn.CrossEntropyLoss()(vsod_out.view(B * L, 2, H, W), masks.reshape(B * L, H, W))

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # loss, iou
                self.stats['loss'].update(loss.detach().cpu().item(), B)
                iou = torch.mean(get_iou(vsod_out.view(B * L, 2, H, W), masks.reshape(B * L, H, W))[:, 1:])
                self.stats['iou'].update(iou.detach().cpu().item(), B)

            print('[ep{:04d}] loss: {:.5f}, iou: {:.5f}'.format(self.epoch, self.stats['loss'].avg, self.stats['iou'].avg))

        if mode == 'val':
            maes = []
            test_loader = torch.utils.data.DataLoader(self.val_set, batch_size=1, num_workers=4)
            for vsod_data in test_loader:
                imgs = vsod_data['imgs'].cuda()
                flows = vsod_data['flows'].cuda()
                masks = vsod_data['masks'].cuda()
                vsod_out = self.model(imgs, flows)

                # calculate scores
                score = torch.softmax(vsod_out, dim=2)[:, :, 1:]
                mae = torch.mean(abs(score - masks))
                maes.append(mae)

            # gather scores
            final_mean = sum(maes) / len(maes)
            print('[ep{:04d}] MAE: {:.5f}\n'.format(self.epoch, final_mean))
            return final_mean

    def save_checkpoint(self, alt_name=None):
        if alt_name is not None:
            file_path = 'weights/{}_{}.pth'.format(self.save_name, alt_name)
        else:
            file_path = 'weights/{}_{:04d}.pth'.format(self.save_name, self.epoch)
        torch.save(self.model.module.state_dict(), file_path)
