import time
from dataset import *
from realflow import RealFlow
from trainer import Trainer
from optparse import OptionParser
import warnings
warnings.filterwarnings('ignore')


parser = OptionParser()
parser.add_option('--train', action='store_true', default=None)
parser.add_option('--test', action='store_true', default=None)
options = parser.parse_args()[0]


def train_mixed(model):
    dutsv3_set = TrainDUTSv3('../DB/DUTSv3', clip_n=256)
    davis_set = TrainDAVIS('../DB/DAVIS16/TrainSet', clip_n=128)
    davsod_set = TrainDAVSOD('../DB/DAVSOD/TrainSet', clip_n=128)
    train_set = torch.utils.data.ConcatDataset([dutsv3_set, davis_set, davsod_set])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_set = TestDAVIS('../DB/DAVIS16/TestSet')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    trainer = Trainer(model, optimizer, train_loader, val_set, save_name='mixed', save_step=1000, val_step=100)
    trainer.train(4000)


def test(model, root):

    # define dataset
    if 'DAVIS16' in root:
        test_set = TestDAVIS(root)
        dataset = 'DAVIS16_val'
    if 'FBMS' in root:
        test_set = TestFBMS(root)
        dataset = 'FBMS_test'
    if 'DAVSOD' in root:
        test_set = TestDAVSOD(root)
        dataset = 'DAVSOD_test'
    if 'ViSal' in root:
        test_set = TestViSal(root)
        dataset = 'ViSal'
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=4)
    model.cuda()

    # inference
    for vsod_data in test_loader:
        imgs = vsod_data['imgs'].cuda()
        flows = vsod_data['flows'].cuda()
        video_name = vsod_data['video_name'][0]
        files = vsod_data['files']
        os.makedirs('outputs/{}/{}'.format(dataset, video_name), exist_ok=True)

        # inference
        t0 = time.time()
        vsod_out = model(imgs, flows)
        t1 = time.time()
        seconds = t1 - t0
        frames = imgs.size(1)

        # save results of each sequence
        score = torch.softmax(vsod_out, dim=2)[:, :, 1:]
        for i in range(frames):
            tv.utils.save_image(score[0, i], 'outputs/{}/{}/{}'.format(dataset, video_name, files[i][0].split('/')[-1]))
        print('seq: {}, {:.1f} fps'.format(video_name, frames / seconds))

    # dataset finished
    print('--- {} done ---\n'.format(dataset))


if __name__ == '__main__':

    # set device
    torch.cuda.set_device(0)

    # define model
    ver = 'mitb2'
    model = RealFlow(ver).eval()

    # training stage
    if options.train:
        model = torch.nn.DataParallel(model)
        train_mixed(model)

    # testing stage
    if options.test:
        model.load_state_dict(torch.load('weights/RealFlow_{}.pth'.format(ver), map_location='cpu'))
        with torch.no_grad():
            test(model, '../DB/DAVIS16/TestSet')
            test(model, '../DB/FBMS/TestSet')
            test(model, '../DB/DAVSOD/TestSet')
            test(model, '../DB/ViSal')
