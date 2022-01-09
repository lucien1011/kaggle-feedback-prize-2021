class Module(object):

    _required_params = []
    _type = "Module"

    def __init__(self):
        pass

    def prepare(self,container,params):
        pass

    def fit(self,container,params):
        pass

    def wrapup(self,container,params):
        pass
 
    @property
    def type(self):
        return self._type

    def check_required_params(self,params):
        for p in self._required_params:
            if not p in params:
                raise AttributeError("{:s} not in {:s}".format(p,str(params)))
