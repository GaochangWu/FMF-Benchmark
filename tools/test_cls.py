import os  # 导入os库，用于文件和目录操作
from tqdm import tqdm  # 导入tqdm库，用于显示进度条
import time  # 导入time库，用于时间相关的操作

import torch  # 导入PyTorch库，用于深度学习模型的定义和训练
from torch.utils.data import DataLoader  # 从PyTorch中导入DataLoader，用于加载数据

from FMF.models import build_model  # 从FMF.models模块导入build_model函数，用于构建模型
from FMF.datasets import build_dataset  # 从FMF.datasets模块导入build_dataset函数，用于构建数据集

from FMF.utils.parser import parse_args, load_config  # 从FMF.utils.parser模块导入parse_args和load_config函数，用于解析命令行参数和加载配置
from FMF.utils.others import get_time  # 从FMF.utils.others模块导入get_time函数，用于获取当前时间
from FMF.utils.metrics import ClassificationMetric  # 从FMF.utils.metrics模块导入ClassificationMetric类，用于计算分类指标


def main():
    """
    主函数，执行测试流程。
    """
    # 解析命令行参数
    args = parse_args()
    # 根据命令行参数加载配置
    cfg = load_config(args)
    # 检查预训练模型是否存在，如果不存在则抛出异常
    assert os.path.exists(cfg.MODEL.PRETRAINED), f'{cfg.MODEL.PRETRAINED}不存在！'
    # 设置设备（GPU或CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载测试集
    print('[{}] Loading test set...'.format(get_time()))
    val_set = build_dataset(name=cfg.TEST.DATASET, cfg=cfg, split='test')
    val_loader = DataLoader(val_set, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.DATA_LOADER.NUM_WORKERS, pin_memory=cfg.DATA_LOADER.PIN_MEMORY)
    # 构建模型
    print('[{}] Constructing model...'.format(get_time()))

    model = build_model(cfg)
    # 加载预训练模型的权重

    # model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED))
    checkpoint = torch.load(cfg.MODEL.PRETRAINED)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # 将模型移动到指定设备（GPU或CPU）
    model.to(device)
    numClass = cfg.MODEL.NUM_CLASSES
    # 初始化分类指标计算工具
    test_metric = ClassificationMetric(numClass)

    # 将模型设置为评估模式
    model.eval()
    # 开始测试
    print('[{}] Testing...'.format(get_time()))
    with torch.no_grad():  # 禁用梯度计算，以节省内存和计算资源
        start_time = time.time()  # 记录测试开始时间
        for i, xy in enumerate(tqdm(val_loader, ncols=100)):  # 遍历测试集，并显示进度条
            y = xy[-1]  # 获取标签
            x = [ipt.to(device) for ipt in xy[:-1]]  # 将输入数据移动到设备
            y_hat = model(*x)  # 前向传播，获取模型的预测结果

            # 将预测结果和真实标签添加到分类指标计算工具中
            test_metric.addBatch(torch.argmax(y_hat.detach().cpu(), dim=1).numpy(), y.cpu().numpy())
    end_time = time.time()  # 记录测试结束时间
    fps = len(val_set) / (end_time - start_time)  # 计算每秒帧数（FPS）

    # 打印测试结果
    print('[{}] Tests complete!'.format(get_time()))
    print('ACC: {}'.format(test_metric.Accuracy()))  # 打印准确率
    print('F1 : {}'.format(test_metric.F1Score()))  # 打印F1分数
    print('FDR: {}'.format(test_metric.FalsePositiveRate()))  # 打印假阳性率
    print('MDR: {}'.format(test_metric.FalseNegativeRate()))  # 打印假阴性率
    print('FPS: {}'.format(fps))  # 打印每秒帧数


if __name__ == '__main__':
    main()  # 执行主函数
