import os
import torch
import logging
import shutil
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)

def save_checkpoint(
    state: Dict,
    is_best: bool,
    output_dir: str,
    filename: str = 'checkpoint.pth'
) -> None:
    """
    保存模型检查点
    
    Args:
        state: 包含模型状态、优化器状态等的字典
        is_best: 是否为最佳模型
        output_dir: 输出目录
        filename: 检查点文件名
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存检查点
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)
    logger.info(f'Saved checkpoint to {filepath}')
    
    # 如果是最佳模型，保存一个副本
    if is_best:
        best_filepath = os.path.join(output_dir, 'model_best.pth')
        shutil.copyfile(filepath, best_filepath)
        logger.info(f'Saved best model to {best_filepath}')

def load_checkpoint(
    checkpoint_path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Dict:
    """
    加载模型检查点
    
    Args:
        checkpoint_path: 检查点文件路径
        model: 要加载权重的模型
        optimizer: 要加载状态的优化器
        scheduler: 要加载状态的学习率调度器
    
    Returns:
        包含检查点内容的字典
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found at {checkpoint_path}')
    
    logger.info(f'Loading checkpoint from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载模型权重
    if model is not None and 'model_state_dict' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info('Successfully loaded model weights')
        except Exception as e:
            logger.error(f'Error loading model weights: {e}')
            raise
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info('Successfully loaded optimizer state')
        except Exception as e:
            logger.error(f'Error loading optimizer state: {e}')
            raise
    
    # 加载学习率调度器状态
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info('Successfully loaded scheduler state')
        except Exception as e:
            logger.error(f'Error loading scheduler state: {e}')
            raise
    
    return checkpoint

def get_last_checkpoint(output_dir: str) -> Optional[str]:
    """
    获取最新的检查点文件路径
    
    Args:
        output_dir: 输出目录
    
    Returns:
        最新的检查点文件路径，如果没有找到则返回None
    """
    if not os.path.exists(output_dir):
        return None
    
    checkpoints = [f for f in os.listdir(output_dir) if f.endswith('.pth')]
    if not checkpoints:
        return None
    
    # 按修改时间排序，返回最新的检查点
    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))
    return os.path.join(output_dir, checkpoints[-1])

def load_pretrained_weights(
    model: torch.nn.Module,
    pretrained_path: str,
    strict: bool = True
) -> None:
    """
    加载预训练权重
    
    Args:
        model: 要加载权重的模型
        pretrained_path: 预训练权重文件路径
        strict: 是否严格匹配权重
    """
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f'Pretrained weights not found at {pretrained_path}')
    
    logger.info(f'Loading pretrained weights from {pretrained_path}')
    state_dict = torch.load(pretrained_path, map_location='cpu')
    
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    try:
        model.load_state_dict(state_dict, strict=strict)
        logger.info('Successfully loaded pretrained weights')
    except Exception as e:
        logger.error(f'Error loading pretrained weights: {e}')
        raise

def save_training_state(
    state: Dict,
    output_dir: str,
    filename: str = 'training_state.pth'
) -> None:
    """
    保存训练状态（不包含模型权重）
    
    Args:
        state: 包含训练状态的字典
        output_dir: 输出目录
        filename: 文件名
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)
    logger.info(f'Saved training state to {filepath}')

def cleanup_checkpoints(output_dir: str, keep_last_n: int = 5) -> None:
    """
    清理旧的检查点文件，只保留最新的N个
    
    Args:
        output_dir: 输出目录
        keep_last_n: 保留最新的检查点数量
    """
    if not os.path.exists(output_dir):
        return
    
    checkpoints = [f for f in os.listdir(output_dir) if f.endswith('.pth')]
    if len(checkpoints) <= keep_last_n:
        return
    
    # 按修改时间排序
    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))
    
    # 删除旧的检查点
    for checkpoint in checkpoints[:-keep_last_n]:
        filepath = os.path.join(output_dir, checkpoint)
        os.remove(filepath)
        logger.info(f'Removed old checkpoint: {filepath}') 