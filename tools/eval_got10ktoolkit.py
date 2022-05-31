
import torch
import argparse
import sys
sys.path.append('../')

from got10k.trackers import Tracker
from got10k.experiments import ExperimentGOT10k

from pysot.core.config import cfg
from pysot.models.utile.model_builder import ModelBuilder
from pysot.tracker.tctrack_tracker import TCTrackTracker_ToolKitEval
from pysot.utils.model_load import load_pretrain

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='TCTrack demo')
parser.add_argument('--config', type=str, default='experiments/config.yaml', help='config file')
parser.add_argument('--snapshot', type=str, default='./tools/snapshot/checkpoint00_e84.pth', help='model name')
args = parser.parse_args()

if __name__ == '__main__':
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder('test')

    # load model
    model = load_pretrain(model, args.snapshot).eval().to(device)

    # build tracker
    tracker = TCTrackTracker_ToolKitEval(model)

    # run experiments on GOT-10k (validation subset)
    experiment = ExperimentGOT10k('data/GOT-10k',
                                  subset='test',
                                  result_dir='results',       # where to store tracking results
                                  report_dir='reports'        # where to store evaluation reports
                                )
    experiment.run(tracker, visualize=True)

    # report performance
    experiment.report([tracker.name])