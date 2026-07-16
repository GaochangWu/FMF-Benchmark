import os
import torch
import time  # 导入time库，用于时间相关的操作
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # 导入TensorBoard
from FMF.utils.parser import parse_args, load_config  # 从FMF.utils.parser模块导入parse_args和load_config函数，用于解析命令行参数和加载配置
from FMF.models import build_model
from FMF.datasets.build import build_dataset
from FMF.utils.checkpoint import save_checkpoint, load_checkpoint
from FMF.utils.others import get_time
from FMF.utils.metrics import ClassificationMetric  # 从FMF.utils.metrics模块导入ClassificationMetric类，用于计算分类指标


def setup_logger(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'train.log'))
            # logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class Trainer:
    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        self.output_dir = output_dir
        self.logger = setup_logger(output_dir)
        
        # 创建TensorBoard的SummaryWriter
        self.writer = SummaryWriter('/root/tf-logs')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 构建数据集和数据加载器
        print('[{}] Loading test and train set...'.format(get_time()))
        self.train_dataset = build_dataset(name=cfg.TRAIN.DATASET,cfg=cfg, split='train')
        self.val_dataset = build_dataset(name=cfg.TEST.DATASET, cfg=cfg, split='test')
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY
        )
        # 构建模型

        print('[{}] Constructing model...'.format(get_time()))
        self.model = build_model(cfg)

        # 输出模型结构
        print(self.model)
        
        # 随机初始化模型参数
        print('[{}] Initializing model parameters...'.format(get_time()))
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                # 使用xavier初始化线性层
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                # 层归一化层初始化
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.MultiheadAttention):
                # 注意力层的初始化
                nn.init.xavier_uniform_(m.in_proj_weight)
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0.0)
                if m.out_proj.bias is not None:
                    nn.init.constant_(m.out_proj.bias, 0.0)
            elif isinstance(m, nn.Conv2d):
                # 如果有卷积层，使用kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

                
        
        # 将模型移动到指定设备（GPU或CPU）
        self.model = self.model.to(self.device)

        self.numClass = cfg.MODEL.NUM_CLASSES
        # 初始化分类指标计算工具
        self.test_metric = ClassificationMetric(self.numClass)
        self.train_metric = ClassificationMetric(self.numClass)

        # 定义损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg.SOLVER.BASE_LR
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.SOLVER.MAX_EPOCH,
            eta_min=cfg.SOLVER.COSINE_END_LR
        )
        
        self.start_epoch = 0
        self.best_acc = 0.0

    
    def train_epoch(self, epoch):
        self.model.train()
        self.train_metric.reset()
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.cfg.SOLVER.MAX_EPOCH}', ncols=100)
        total_loss = 0.0  # 添加损失累计变量
        for batch_idx, xy in enumerate(pbar):
            y = xy[-1].to(self.device)  # 获取标签
            x = [ipt.to(self.device) for ipt in xy[:-1]]

            # 前向传播
            outputs = self.model(*x)
            loss = self.criterion(outputs, y)
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # 累计损失
            total_loss += loss.item()
            # 计算准确率
            self.train_metric.addBatch(torch.argmax(outputs.detach().cpu(), dim=1).numpy(), y.cpu().numpy())
            acc = self.train_metric.Accuracy()
            f1 = self.train_metric.F1Score()
            recall = self.train_metric.Recall()
            self.avg_loss = total_loss / (batch_idx + 1)
            
            # 记录训练指标到TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Loss/train', self.avg_loss, global_step)
            self.writer.add_scalar('Accuracy/train', acc, global_step)
            self.writer.add_scalar('F1/train', f1, global_step)
            self.writer.add_scalar('Recall/train', recall, global_step)
            self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], global_step)
            
            # 更新进度条
            if batch_idx % 200 == 0:
                pbar.set_postfix({
                    'loss': f'{self.avg_loss:.4f}'
                })
            # 记录日志
            if batch_idx % self.cfg.PRINT_FREQ == 0:
                self.logger.info(
                    f'Epoch: [{epoch}][{batch_idx}/{len(self.train_loader)}] '
                    f'loss: {self.avg_loss:.4f} '
                    f'acc: {acc:.4f} '
                    f'F1: {f1:.4f} '
                    f'recall: {recall:.4f}'
                )

        print('[{}] Train test complete! loss: {: .4f}, acc : {: .4f}, F1: {: .4f}, recall: {: .4f}'.format(get_time(),
                                                                                    self.avg_loss,
                                                                                    self.train_metric.Accuracy(),
                                                                                    self.train_metric.F1Score(),
                                                                                    self.train_metric.Recall()
                                                                                    ))
        self.acc = self.train_metric.Accuracy()

    def validate(self):
        self.model.eval()
        self.test_metric.reset()

        # 开始测试
        print('[{}] Testing...'.format(get_time()))
        with torch.no_grad():  # 禁用梯度计算，以节省内存和计算资源
            start_time = time.time()  # 记录测试开始时间
            for i, xy in enumerate(tqdm(self.val_loader, ncols=100)):  # 遍历测试集，并显示进度条
                y = xy[-1]  # 获取标签
                x = [ipt.to(self.device) for ipt in xy[:-1]]  # 将输入数据移动到设备
                y_hat = self.model(*x)  # 前向传播，获取模型的预测结果

                # 将预测结果和真实标签添加到分类指标计算工具中
                self.test_metric.addBatch(torch.argmax(y_hat.detach().cpu(), dim=1).numpy(), y.cpu().numpy())
        end_time = time.time()  # 记录测试结束时间
        fps = len(self.val_loader) / (end_time - start_time)  # 计算每秒帧数（FPS）

        # 记录验证指标到TensorBoard
        val_acc = self.test_metric.Accuracy()
        val_f1 = self.test_metric.F1Score()
        val_fdr = self.test_metric.FalsePositiveRate()
        val_mdr = self.test_metric.FalseNegativeRate()
        
        self.writer.add_scalar('Accuracy/val', val_acc, self.current_epoch)
        self.writer.add_scalar('F1/val', val_f1, self.current_epoch)
        self.writer.add_scalar('FDR/val', val_fdr, self.current_epoch)
        self.writer.add_scalar('MDR/val', val_mdr, self.current_epoch)
        self.writer.add_scalar('FPS/val', fps, self.current_epoch)

        print('[{}] Tests complete! ACC: {}, F1 : {}, FDR: {}, MDR: {}, FPS: {}'.format(get_time(),
                                                                                    val_acc,
                                                                                    val_f1,
                                                                                    val_fdr,
                                                                                    val_mdr,
                                                                                    fps))

        return val_acc, val_f1
    
    def train(self):
        for epoch in range(self.start_epoch, self.cfg.SOLVER.MAX_EPOCH):
            self.current_epoch = epoch  # 添加当前epoch的记录
            # 训练一个epoch
            self.train_epoch(epoch)
            # 验证
            val_acc, val_f1 = self.validate()
            # 更新学习率
            self.scheduler.step()
            # 记录日志
            self.logger.info(
                f'Epoch: {epoch}, '
                f'Train Loss: {self.avg_loss:.4f}, '
                f'Train Acc: {self.acc:.4f}, '
                f'Val F1: {val_f1:.4f}, '
                f'Val Acc: {val_acc:.4f}'
            )
            
            # 保存最佳模型
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                save_checkpoint(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'best_acc': self.best_acc,
                    },
                    is_best=True,
                    output_dir=self.output_dir
                )

            # 定期保存检查点
            if (epoch + 1) % self.cfg.TRAIN.CHECKPOINT_PERIOD == 0:
                save_checkpoint(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'best_acc': self.best_acc,
                    },
                    is_best=False,
                    output_dir=self.output_dir
                )
        
        # 关闭TensorBoard的SummaryWriter
        self.writer.close()
    
    def resume_from_checkpoint(self, checkpoint_path):
        checkpoint = load_checkpoint(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']
        self.logger.info(f'Resumed from epoch {self.start_epoch} with best acc {self.best_acc}')

def main():
    # 解析命令行参数
    args = parse_args()
    # 根据命令行参数加载配置
    cfg = load_config(args)

    # 设置输出目录
    output_dir = os.path.join(
        cfg.OUTPUT_DIR,
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    )

    os.makedirs(output_dir, exist_ok=True)
    
    # 创建训练器
    trainer = Trainer(cfg, output_dir)
    
    # 如果指定了检查点，则从检查点恢复
    if args.resume:
        trainer.resume_from_checkpoint(args.resume)
    
    # 开始训练
    trainer.train()

if __name__ == '__main__':
    main() 