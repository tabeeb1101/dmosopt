import sys
import numpy as np
import logging, dmosopt
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def zdt1(x):
    ''' This is the Zitzler-Deb-Thiele Function - type A
        Bound: XUB = [1,1,...]; XLB = [0,0,...]
        dim = 30
    '''
    f = np.zeros(2)
    f[0] = x[0]
    g = 1. + 9./29.*np.sum(x[1:])
    h = 1. - np.sqrt(f[0]/g)
    f[1] = g*h
    return f


def obj_fun(pp):
    """ Objective function to be minimized. """
    param_values = np.asarray([pp[k] for k in sorted(pp)])
    res = zdt1(param_values)
    logger.info(f"Iter: \t pp:{pp}, result:{res}")
    return res


def zdt1_pareto():
    n = 100
    f = np.zeros([n,2])
    f[:,0] = np.linspace(0,1,n)
    f[:,1] = 1.0 - np.sqrt(f[:,0])
    return f

if __name__ == '__main__':

    space = {}
    for i in range(30):
        space['x%d' % (i+1)] = [0.0, 1.0]
    problem_parameters = {}
    objective_names = ['y1', 'y2']
    
    # Create an optimizer
    dmosopt_params = {'opt_id': 'dmosopt_zdt1',
                      'obj_fun_name': 'obj_fun',
                      'obj_fun_module': 'example_dmosopt_zdt1',
                      'problem_parameters': problem_parameters,
                      'space': space,
                      'objective_names': objective_names,
                      'n_initial': 1,
                      'n_iter': 4}
    
    best = dmosopt.run(dmosopt_params, verbose=True)
    if best is not None:
        import matplotlib.pyplot as plt
        bestx, besty = best
        x, y = dmosopt.sopt_dict['dmosopt_zdt1'].optimizer_dict[0].get_evals()
        
        # plot results
        plt.plot(y[:,0],y[:,1],'b.',label='evaluated points')
        plt.plot(besty['y1'],besty['y1'],'r.',label='MO-ASMO')
    
        y_true = zdt1_pareto()
        plt.plot(y_true[:,0],y_true[:,1],'k-',label='True Pareto')
        plt.legend()
        
        plt.show()
        

