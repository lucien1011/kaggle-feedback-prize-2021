from itertools import islice
from .Container import Container
from .Module import Module

class BasePipeline(object):
    _required_parameters = ["steps"]

    def __init__(self, steps, base_dir=None, verbose=False):
        self.steps = steps
        self.container = Container(base_dir)
        self.verbose = verbose

    def _validate_steps(self):
        names, steps = zip(*self.steps)

        self._validate_names(names)

        for s in steps:
            if not (
                    (hasattr(s, "prepare") and hasattr(s, "fit") and hasattr(s, "wrapup")) 
                    or hasattr(s,'run')
                    ):
                raise TypeError(
                    "All estimators should implement (prepare, fit and wrapup) or run."
                    "'%s' (type %s) doesn't" % (s, type(s))
                )

    def _validate_names(self,names):
        for n in names:
            if not type(n) == str:
                raise TypeError(
                    "Estimator names should be of type str. {s} (type %s) doesn't" % (n, type(n))
                )

    def __len__(self):
        """
        Returns the length of the Pipeline
        """
        return len(self.steps)

    def __getitem__(self, ind):
        """Returns a sub-pipeline or a single estimator in the pipeline
        Indexing with an integer will return an estimator; using a slice
        returns another Pipeline instance which copies a slice of this
        Pipeline. This copy is shallow: modifying (or fitting) estimators in
        the sub-pipeline will affect the larger pipeline and vice-versa.
        However, replacing a value in `step` will not affect a copy.
        """
        if isinstance(ind, slice):
            if ind.step not in (1, None):
                raise ValueError("Pipeline slicing only supports a step of 1")
            return self.__class__(
                self.steps[ind], verbose=self.verbose
            )
        try:
            name, est = self.steps[ind]
        except TypeError:
            # Not an int, try get step by name
            return self.named_steps[ind]
        return est

    @property
    def named_steps(self):
        """Access the steps by name.
        Read-only attribute to access any step by given name.
        Keys are steps names and values are the steps objects."""
        # Use Bunch object to improve autocomplete
        return Bunch(**dict(self.steps))

    def _iter(self):
        """
        Generate (idx, (name, mod)) tuples from self.steps
        """
        stop = len(self.steps)
        for idx, (name, step) in enumerate(islice(self.steps, 0, stop)):
            yield idx, name, step

    def _run(self,mod_params):
        self.steps = list(self.steps)
        self._validate_steps() 

        self.container.create_base_dir()
        for (step_idx, name, mods) in self._iter():
            if isinstance(mods,Module):
                mods = [mods]
            self.container.set_subdir(name)
            for mod in mods:
                mod.check_required_params(mod_params[name])
                mod.prepare(self.container,mod_params[name])
                mod.fit(self.container,mod_params[name])
                mod.wrapup(self.container,mod_params[name])

    def run(self,mod_params):
        self._run(mod_params)
