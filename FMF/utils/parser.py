import argparse
import os

from FMF.config.defaults import get_cfg


def parse_args():
    """
    Args:
        cfg (str): path to the config file.
    """
    parser = argparse.ArgumentParser(
        description="Provide FMF-Trans training and testing pipeline."
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/CfgForViCu_Cls.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See FMF/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def load_config(args):
    """
    Load config from cfg
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.set_new_allowed(True)
        cfg.merge_from_file(args.cfg_file, )
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    # Create the checkpoint dir.
    if not os.path.exists(cfg.OUTPUT_DIR) and cfg.OUTPUT_DIR != '':
        os.makedirs(cfg.OUTPUT_DIR)

    return cfg
