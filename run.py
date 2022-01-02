from pipeline import BasePipeline
from sentclass import Preprocess 
from utils import read_attr_conf

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('conf',action='store')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()
    mod_params = read_attr_conf(args.conf,'conf')
    pp = BasePipeline(
            [
                ('Preprocess',Preprocess()),
            ],
            base_dir = mod_params['base_dir'],
            )
    pp.run(mod_params)
