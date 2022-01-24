import os
import pandas as pd
import pickle
import torch

from utils import mkdir_p

fname_ext_map = {
        'df_csv': '.csv',
        'pickle': '.p',
        'torch_model': '.pt',
        'matplotlib_fig+ax': '.png',
        }

class Container(object):
    def __init__(self,base_dir):
        self.base_dir = base_dir
        self.sub_dir = None
        self.items_to_write = {}
        self.items_to_read = {}

    def add_item(self,name,item,ftype,mode='read'):
        if mode == 'read':
            self.items_to_read[name] = (item,ftype)
        elif mode == 'write':
            self.items_to_write[name] = (item,ftype)
        else:
            raise RuntimeError

    def read_item_from_path(self,name,path,ftype,args={}):
        if ftype == 'df_csv':
            obj = pd.read_csv(path,**args)
        elif ftype == 'pickle':
            obj = pickle.load(open(path),'wb')
        else:
            raise RuntimeError
        self.add_item(name,obj,ftype,mode='read')

    def read_item_from_dir(self,name,ftype,args={},mod_name=None,newkey=None):
        if ftype == 'df_csv':
            obj = pd.read_csv(self.fname_ext(self.mod_path(name,mod_name),ftype),**args)
        elif ftype == 'pickle':
            obj = pickle.load(open(self.fname_ext(self.mod_path(name,mod_name),ftype),'wb'))
        else:
            raise RuntimeError
        self.add_item(name if newkey is None else newkey,obj,ftype,mode='read')

    def __getattr__(self,name):
        if name in self.items_to_read:
            return self.items_to_read[name][0]
        elif name in self.items_to_write:
            return self.items_to_write[name][0]
        else:
            raise AttributeError

    def get(self,name):
        return self.__getattr__(name)

    def save(self,clearcache=False):
        mkdir_p(self.mod_dir)
        for name,(item,ftype) in self.items_to_write.items():
            self.save_one_item(name,item,ftype)
        if clearcache: self.items_to_write = {}

    def save_one_item(self,fname,obj,ftype='df_csv',check_dir=False):
        path = self.fname_ext(self.mod_path(fname),ftype) 
        if check_dir:
            mkdir_p(os.path.dirname(path))
        if ftype == 'df_csv':
            obj.to_csv(path,index=False)
        elif ftype == 'pickle':
            pickle.dump(obj,open(path,'wb'))
        elif ftype == 'torch_model':
            torch.save(obj,path)
        elif ftype == 'matplotlib_fig+ax':
            fig,ax = obj
            fig.savefig(path)
        else:
            raise RuntimeError
        print('Object {} has been saved as type {}'.format(fname,ftype))

    def file_exists(self,fname):
        return os.path.exists(os.path.join(self.base_dir,self.sub_dir,fname))

    def fname_ext(self,fname,ftype):
        if ftype in fname_ext_map:
            return fname+fname_ext_map[ftype]
        else:
            raise KeyError

    def mod_path(self,fname,mod_name=None):
        return os.path.join(self.mod_dir if mod_name is None else os.path.join(self.base_dir,mod_name),fname)

    @property
    def mod_dir(self):
        if self.sub_dir is not None:
            return os.path.join(self.base_dir,self.sub_dir)
        else:
            raise AttributeError

    def set_subdir(self,name):
        self.sub_dir = name+'/'
