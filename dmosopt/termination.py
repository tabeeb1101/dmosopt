#
# Optimization termination conditions.
#
# Based on termination clasess from PyMOO: 
# https://github.com/anyoptimization/pymoo
#

import numpy as np
from abc import abstractmethod
from dmosopt.normalization import normalize
from dmosopt.indicators import IGD

class SlidingWindow(list):

    def __init__(self, size=None) -> None:
        super().__init__()
        self.size = size

    def append(self, entry):
        super().append(entry)

        if self.size is not None:
            while len(self) > self.size:
                self.pop(0)

    def is_full(self):
        return self.size == len(self)

    def to_numpy(self):
        return np.array(self)



class Termination:

    def __init__(self) -> None:
        """
        Base class for the implementation of a termination criterion for an optimization.
        """
        super().__init__()

        # the optimization can be forced to terminate by setting this attribute to true
        self.force_termination = False

    def do_continue(self, opt):
        """

        Whenever the optimization objects wants to know whether it should continue or not it simply
        asks the termination criterion for it.

        Parameters
        ----------
        opt : class
            The optimization object that is asking if it has terminated or not.

        Returns
        -------
        do_continue : bool
            Whether the optimization has terminated or not.

        """

        if self.force_termination:
            return False
        else:
            return self._do_continue(opt)

    # the concrete implementation of the optimization
    def _do_continue(self, opt, **kwargs):
        pass

    def has_terminated(self, opt):
        """
        Instead of asking if the optimization should continue it can also ask if it has terminated.
        (just negates the continue method.)
        """
        return not self.do_continue(optimization)



class TerminationCollection(Termination):

    def __init__(self, *args) -> None:
        super().__init__()
        self.terminations = args

    def _do_continue(self, opt):
        for term in self.terminations:
            if not term.do_continue(opt):
                return False
        return True


class MaximumFunctionCallTermination(Termination):

    def __init__(self, n_max_evals) -> None:
        super().__init__()
        self.n_max_evals = n_max_evals

        if self.n_max_evals is None:
            self.n_max_evals = float("inf")

    def _do_continue(self, opt):
        return opt.n_eval < self.n_max_evals



class MaximumGenerationTermination(Termination):

    def __init__(self, n_max_gen) -> None:
        super().__init__()
        self.n_max_gen = n_max_gen

        if self.n_max_gen is None:
            self.n_max_gen = float("inf")

    def _do_continue(self, opt):
        return opt.n_gen < self.n_max_gen

    
    
class SlidingWindowTermination(TerminationCollection):

    def __init__(self,
                 metric_window_size=None,
                 data_window_size=None,
                 min_data_for_metric=1,
                 nth_gen=1,
                 n_max_gen=None,
                 n_max_evals=None,
                 truncate_metrics=True,
                 truncate_data=True,
                 ):
        """

        Parameters
        ----------

        metric_window_size : int
            The last generations that should be considering during the calculations

        data_window_size : int
            How much of the history should be kept in memory based on a sliding window.

        nth_gen : int
            Each n-th generation the termination should be checked for

        """

        super().__init__(MaximumGenerationTermination(n_max_gen=n_max_gen),
                         MaximumFunctionCallTermination(n_max_evals=n_max_evals))

        # the window sizes stored in objects
        self.data_window_size = data_window_size
        self.metric_window_size = metric_window_size

        # the obtained data at each iteration
        self.data = SlidingWindow(data_window_size) if truncate_data else []

        # the metrics calculated also in a sliding window
        self.metrics = SlidingWindow(metric_window_size) if truncate_metrics else []

        # each n-th generation the termination decides whether to terminate or not
        self.nth_gen = nth_gen

        # number of entries of data need to be stored to calculate the metric at all
        self.min_data_for_metric = min_data_for_metric

        
    def _do_continue(self, opt):

        # if the maximum generation or maximum evaluations say terminated -> do so
        if not super()._do_continue(opt):
            return False

        # store the data decided to be used by the implementation
        obj = self._store(opt)
        if obj is not None:
            self.data.append(obj)

        # if enough data has be stored to calculate the metric
        if len(self.data) >= self.min_data_for_metric:
            metric = self._metric(self.data[-self.data_window_size:])
            if metric is not None:
                self.metrics.append(metric)

        # if its the n-th generation and enough metrics have been calculated make the decision
        if opt.n_gen % self.nth_gen == 0 and len(self.metrics) >= self.metric_window_size:

            # ask the implementation whether to terminate or not
            return self._decide(self.metrics[-self.metric_window_size:])

        # otherwise by default just continue
        else:
            return True

    # given an optimization object decide what should be stored as historical information - by default just opt
    def _store(self, opt):
        return opt

    @abstractmethod
    def _decide(self, metrics):
        pass

    @abstractmethod
    def _metric(self, data):
        pass

    def get_metric(self):
        if len(self.metrics) > 0:
            return self.metrics[-1]
        else:
            return None


class ParameterToleranceTermination(SlidingWindowTermination):

    def __init__(self,
                 n_last=20,
                 tol=1e-6,
                 nth_gen=1,
                 n_max_gen=None,
                 n_max_evals=None,
                 **kwargs):

        super().__init__(metric_window_size=n_last,
                         data_window_size=2,
                         min_data_for_metric=2,
                         nth_gen=nth_gen,
                         n_max_gen=n_max_gen,
                         n_max_evals=n_max_evals,
                         **kwargs)
        self.tol = tol

    def _store(self, opt):
        problem = opt.problem
        X = opt.x

        if X.dtype != object:
            if problem.xl is not None and problem.xu is not None:
                X = normalize(X, xl=problem.xl, xu=problem.xu)
            return X

    def _metric(self, data):
        last, current = data[-2], data[-1]
        return IGD(current).do(last)

    def _decide(self, metrics):
        return to_numpy(metrics).mean() > self.tol


def calc_delta_norm(a, b, norm):
    return np.max(np.abs((a - b) / norm))


class MultiObjectiveToleranceTermination(SlidingWindowTermination):

    def __init__(self,
                 tol=0.0025,
                 n_last=30,
                 nth_gen=5,
                 n_max_gen=None,
                 n_max_evals=None,
                 **kwargs) -> None:
        super().__init__(metric_window_size=n_last,
                         data_window_size=2,
                         min_data_for_metric=2,
                         nth_gen=nth_gen,
                         n_max_gen=n_max_gen,
                         n_max_evals=n_max_evals,
                         **kwargs)
        self.tol = tol

    def _store(self, opt):
        F = opt.y
        return {
            "ideal": F.min(axis=0),
            "nadir": F.max(axis=0),
            "F": F
        }

    def _metric(self, data):
        last, current = data[-2], data[-1]

        # this is the range between the nadir and the ideal point
        norm = current["nadir"] - current["ideal"]

        # if the range is degenerated (very close to zero) - disable normalization by dividing by one
        norm[norm < 1e-32] = 1

        # calculate the change from last to current in ideal and nadir point
        delta_ideal = calc_delta_norm(current["ideal"], last["ideal"], norm)
        delta_nadir = calc_delta_norm(current["nadir"], last["nadir"], norm)

        # get necessary data from the current population
        c_F, c_ideal, c_nadir = current["F"], current["ideal"], current["nadir"]

        # normalize last and current with respect to most recent ideal and nadir
        c_N = normalize(c_F, c_ideal, c_nadir)
        l_N = normalize(last["F"], c_ideal, c_nadir)

        # calculate IGD from one to another
        delta_f = IGD(c_N).do(l_N)

        return {
            "delta_ideal": delta_ideal,
            "delta_nadir": delta_nadir,
            "delta_f": delta_f
        }

    def _decide(self, metrics):
        delta_ideal = [e["delta_ideal"] for e in metrics]
        delta_nadir = [e["delta_nadir"] for e in metrics]
        delta_f = [e["delta_f"] for e in metrics]
        return max(max(delta_ideal), max(delta_nadir), max(delta_f)) > self.tol


class ConstraintViolationToleranceTermination(SlidingWindowTermination):

    def __init__(self,
                 n_last=20,
                 tol=1e-6,
                 nth_gen=1,
                 n_max_gen=None,
                 n_max_evals=None,
                 **kwargs):

        super().__init__(metric_window_size=n_last,
                         data_window_size=2,
                         min_data_for_metric=2,
                         nth_gen=nth_gen,
                         n_max_gen=n_max_gen,
                         n_max_evals=n_max_evals,
                         **kwargs)
        self.tol = tol

    def _store(self, opt):
        return opt.c

    def _metric(self, data):
        last, current = data[-2], data[-1]
        return {"cv": current,
                "delta_cv": abs(last - current)
                }

    def _decide(self, metrics):
        cv = to_numpy([e["cv"] for e in metrics])
        delta_cv = to_numpy([e["delta_cv"] for e in metrics])
        n_feasible = (cv > 0).sum()

        # if the whole window had only feasible solutions
        if n_feasible == len(metrics):
            return False
        # transition period - some were feasible some were not
        elif 0 < n_feasible < len(metrics):
            return True
        # all solutions are infeasible
        else:
            return delta_cv.max() > self.tol

        
class StdTermination(SlidingWindowTermination):

    def __init__(self,
                 x_tol,
                 cv_tol,
                 f_tol,
                 n_max_gen=1000,
                 n_max_evals=100000,
                 **kwargs):
        
        super().__init__(metric_window_size=1,
                         data_window_size=1,
                         min_data_for_metric=1,
                         n_max_gen=n_max_gen,
                         n_max_evals=n_max_evals,
                         **kwargs)

        self.x_tol = x_tol
        self.cv_tol = cv_tol
        self.f_tol = f_tol

    def metric(self, data):
        opt = data[-1]
        return {
            "x_tol": self.x_tol.do_continue(opt),
            "cv_tol": self.cv_tol.do_continue(opt),
            "f_tol": self.f_tol.do_continue(opt)
        }

    def _decide(self, metrics):
        decisions = metrics[-1]
        return decisions["x_tol"] and (decisions["cv_tol"] or decisions["f_tol"])


    
class MultiObjectiveStdTermination(DefaultTermination):
    def __init__(self,
                 x_tol=1e-8,
                 cv_tol=1e-6,
                 f_tol=0.01,
                 nth_gen=5,
                 n_last=30,
                 **kwargs) -> None:
        super().__init__(ParameterToleranceTermination(tol=x_tol, n_last=n_last),
                         ConstraintViolationToleranceTermination(tol=cv_tol, n_last=n_last),
                         MultiObjectiveToleranceTermination(tol=f_tol, n_last=n_last, nth_gen=nth_gen),
                         **kwargs)
