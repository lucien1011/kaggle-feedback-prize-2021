from pipeline import BasePipeline
from ner import * 
from utils import read_attr_conf

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
            [(step_name,eval(step_name+"()")) for step_name in args.steps.split(',')],
            base_dir = mod_params['base_dir'],
            )
    pp.run(mod_params)
