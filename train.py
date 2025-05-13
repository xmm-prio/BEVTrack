import argparse
from mmengine.config import Config
from mmengine.runner import Runner
import os
from datasets.metrics import TrackAccuracy
from mmengine.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Train a 3D detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--resume', default=None, help='train config file path')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

if __name__ == '__main__':

    args = parse_args()
    cfg = Config.fromfile(args.config)

    metric = TrackAccuracy()
    evaluator = Evaluator(metric)

    runner = Runner(model=cfg.model,
                    resume=args.resume,
                    visualizer=cfg.visualizer,
                    default_hooks=cfg.default_hooks,
                    env_cfg=cfg.env_cfg,
                    work_dir='./work_dir',
                    train_cfg=cfg.train_cfg,
                    train_dataloader=cfg.train_dataloader,
                    val_dataloader=cfg.val_dataloader,
                    val_evaluator=evaluator,
                    val_cfg=cfg.val_cfg,
                    optim_wrapper=cfg.optim_wrapper,
                    launcher=args.launcher,
                    cfg=dict())

    runner.train()
