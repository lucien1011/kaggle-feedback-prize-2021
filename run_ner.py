from pipeline import BasePipeline
from ner import Preprocess,PrepareData,Train
from utils import read_attr_conf

mod_map = dict(
        Preprocess=Preprocess(),
        PrepareData=PrepareData(),
        Train=Train(),
        )

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('conf',action='store')
    parser.add_argument('steps',action='store',default='Preprocess,PrepareData,Train')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()
    mod_params = read_attr_conf(args.conf,'conf')
    step_names = args.steps.split(',')
    pp = BasePipeline(
            [(step_name,mod_map[step_name]) for step_name in args.steps.split(',')],
            base_dir = mod_params['base_dir'],
            )
    pp.run(mod_params)
