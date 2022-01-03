import argparse
from utils import read_attr_conf

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('conf')
    parser.add_argument('jobs')
    return parser.parse_args()

def submit(conf,jobs,params):
    import os
    from slurm import SLURMWorker

    mkdir_p(params['base_dir'])
    script_file_name = os.path.join(params['base_dir'],params['slurm']['fname'])
    
    worker = SLURMWorker()
    slurm_commands = """
cd {base_path}
source setup_hpg.sh
python3 {pyscript} {cfg_path} {mode}
""".format(
            pyscript=params['slurm']['fname'],
            cfg_path=conf,
            mode=jobs,
            base_path=os.environ['BASE_PATH'],
            )
    worker.make_sbatch_script(
            script_file_name,
            conf['name'],
            conf['email'],
            "1",
            conf['memory'],
            conf['time'],
            conf['base_dir']
            slurm_commands,
            conf['gpu'],
            )
    worker.sbatch_submit(script_file_name)

if __name__ == "__main__":
    
    args = parse_arguments()
    params = read_attr_conf(args.conf,'conf')
    submit(args.conf,args.jobs,arams)
