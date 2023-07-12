import sys, logging
import numpy as np
from dmosopt import dmosopt
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import os
from mpi4py import MPI
from aim import Run, Figure
from dmosopt.hv import *

if MPI.COMM_WORLD.Get_rank() == 0:
    run = Run(
        repo="/home/tabeeb/scratch/results"
    )


def zdt1(x):
    ''' This is the Zitzler-Deb-Thiele Function - type A
        Bound: XUB = [1,1,...]; XLB = [0,0,...]
        dim = 30
    '''
    num_variables = len(x)
    f = np.zeros(2)
    f[0] = x[0]
    g = 1. + 9./float(num_variables-1)*np.sum(x[1:])
    h = 1. - np.sqrt(f[0]/g)
    f[1] = g*h
    return f


def obj_fun(pp):
    """ Objective function to be minimized. """
    param_values = np.asarray([pp[k] for k in sorted(pp)])
    res = zdt1(param_values)
    logger.info(f"Iter: \t pp:{pp}, result:{res}")
    return res


def zdt1_pareto(n_points=100):
    f = np.zeros([n_points,2])
    f[:,0] = np.linspace(0,1,n_points)
    f[:,1] = 1.0 - np.sqrt(f[:,0])
    return f

if __name__ == '__main__':

    space = {}
    for i in range(30):
        space['x%d' % (i+1)] = [0.0, 1.0]
    problem_parameters = {}
    objective_names = ['y1', 'y2']

    filepath = os.path.join('/home/tabeeb/scratch/results',
                            os.environ.get("SLURM_JOB_ID"))

    os.makedirs(filepath, exist_ok=True)
    
    # Create an optimizer
    dmosopt_params = {'opt_id': 'dmosopt_zdt1',
                      'obj_fun_name': 'obj_fun',
                      'obj_fun_module': 'example_dmosopt_zdt1',
                      'problem_parameters': problem_parameters,
                      'space': space,
                      'objective_names': objective_names,
                      'population_size': 200,
                      'num_generations': 100,
                      'initial_maxiter': 10,
                      'optimizer': 'age',
                      'termination_conditions': True,
                      'n_initial': 3,
                      'n_epochs': 4,
                      'save_surrogate_eval':True,
                      'save': True,
                      'file_path': os.path.join(filepath, "results.h5"),
                    }
    if MPI.COMM_WORLD.Get_rank() == 0:
        run['hparams'] = dmosopt_params

    best = dmosopt.run(dmosopt_params, verbose=True)

    #if MPI.COMM_WORLD.Get_rank() == 0:
    #run['hparams'] = dmosopt_params
    if best is not None:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        bestx, besty = best
        x, y = dmosopt.sopt_dict['dmosopt_zdt1'].optimizer_dict[0].get_evals()
        besty_dict = dict(besty)
        
        # plot results
        plt.plot(y[:,0],y[:,1],'b.',label='evaluated points')
        plt.plot(besty_dict['y1'],besty_dict['y2'],'r.',label='best points')

        y_true = zdt1_pareto()
        plt.plot(y_true[:,0],y_true[:,1],'k-',label='True Pareto')
        plt.legend()
        
        plt.savefig(os.path.join(filepath, "plot.png"))

        if MPI.COMM_WORLD.Get_rank() == 0:
            aim_figure = Figure(fig)
            run.track(aim_figure, name="plot", step=0)

    referencePoint = [11, 11]
    hv = HyperVolume(referencePoint)
    front = y_true
    volume = hv.compute(front)
    print("vol:")
    print(volume)
    print("-- done!")

    run.track({'hv score': volume}, context={'subset': 'train'})