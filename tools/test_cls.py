import os
from tqdm import tqdm
import time

import torch
from torch.utils.data import DataLoader

from FMF.models import build_model
from FMF.datasets import build_dataset

from FMF.utils.parser import parse_args, load_config
from FMF.utils.others import get_time
from FMF.utils.metrics import ClassificationMetric


def main():
    args = parse_args()
    cfg = load_config(args)
    assert os.path.exists(cfg.MODEL.PRETRAINED), f'{cfg.MODEL.PRETRAINED}不存在！'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('[{}] Loading test set...'.format(get_time()))
    val_set = build_dataset(name=cfg.TEST.DATASET, cfg=cfg, split='test')
    val_loader = DataLoader(val_set, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.DATA_LOADER.NUM_WORKERS, pin_memory=cfg.DATA_LOADER.PIN_MEMORY)

    print('[{}] Constructing model...'.format(get_time()))
    model = build_model(cfg)
    model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED))
    model.to(device)

    test_metric = ClassificationMetric(numClass=cfg.MODEL.NUM_CLASSES)

    model.eval()
    print('[{}] Testing...'.format(get_time()))
    with torch.no_grad():
        start_time = time.time()
        for i,  xy, in enumerate(tqdm(val_loader, ncols=100)):
            y = xy[-1]
            x = [ipt.to(device) for ipt in xy[:-1]]
            y_hat = model(*x)

            test_metric.addBatch(torch.argmax(y_hat.detach().cpu(), dim=1).numpy(), y.cpu().numpy())
    end_time = time.time()
    fps = len(val_set) / (end_time - start_time)

    print('[{}] Tests complete!'.format(get_time()))
    print('ACC: {}'.format(test_metric.Accuracy()))
    print('F1 : {}'.format(test_metric.F1Score()))
    print('FDR: {}'.format(test_metric.FalsePositiveRate()))
    print('MDR: {}'.format(test_metric.FalseNegativeRate()))
    print('FPS: {}'.format(fps))


if __name__ == '__main__':
    main()
