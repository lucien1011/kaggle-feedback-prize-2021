import argparse
import sys

from utils import read_attr_conf,mkdir_p

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('conf')
    parser.add_argument('--jobs',action='store',default='')
    return parser.parse_args()

def submit(conf,jobs,params):
    import os
    from slurm import SLURMWorker

    mkdir_p(params['base_dir'])
    script_file_name = os.path.join(params['base_dir'],params['slurm']['fname'])
    
    worker = SLURMWorker()
    run_commands = params['slurm'].get('commands',"python3 {pyscript} {cfg_path} {mode}".format(pyscript=params['slurm']['pyscript'],mode=jobs,cfg_path=conf))

    slurm_commands = """
cd {base_path}
source setup_hpg.sh
{commands}
""".format(commands=run_commands,base_path=os.environ['BASE_PATH'])
    worker.make_sbatch_script(
            script_file_name,
            params['slurm']['name'],
            params['slurm']['email'],
            params['slurm']['ntasks'],
            params['slurm']['memory'],
            params['slurm']['time'],
            params['base_dir'],
            slurm_commands,
            params['slurm']['ncore'],
            params['slurm']['gpu'],
            )
    worker.sbatch_submit(script_file_name)

if __name__ == "__main__":
    
    args = parse_arguments()
    params = read_attr_conf(args.conf,'conf')
    submit(args.conf,args.jobs,params)
