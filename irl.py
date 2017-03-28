#!/usr/bin/env python
import sys
import getopt
import world
import car
import utils
import theano as th
import theano.tensor as tt
import theano.tensor.nlinalg as tn
import pickle
import feature
import numpy as np

th.config.optimizer_verbose = True

def run_irl(world, car, reward, theta, data):
    def gen():
        for point in data:
            for c, x0, u in zip(world.cars, point['x0'], point['u']):
                c.traj.x0.set_value(x0)
                for cu, uu in zip(c.traj.u, u):
                    cu.set_value(uu)
            yield
    r = car.traj.reward(reward)
    g = utils.grad(r, car.traj.u)
    H = utils.hessian(r, car.traj.u)
    I = tt.eye(utils.shape(H)[0])
    reg = utils.vector(1)
    reg.set_value([1e-1])
    H = H-reg[0]*I
    L = tt.dot(g, tt.dot(tn.MatrixInverse()(H), g))+tt.log(tn.Det()(-H))
    for _ in gen():
        pass
    optimizer = utils.Maximizer(L, [theta], gen=gen, method='gd', eps=0.1, debug=True, iters=10000, inf_ignore=100)
    print type(optimizer.f_and_df)
    optimizer.maximize()
    print theta.get_value()
    return theta.get_value()


if __name__ == '__main__':
    # non-randomized
    theta_list = [[3., -50., 10., 10., -60.]]
    #theta_list= [np.array([(np.random.random()-.5)*100, (np.random.random()-.5)*100.,(np.random.random()-.5)*100., (np.random.random()-.5)*100., (np.random.random()-.5)*100]) for i in range(2)]
    results = []
    for theta_val in theta_list:
        optlist, args = getopt.gnu_getopt(sys.argv, 'w:c:')
        opts = dict(optlist)
        files = args[1:]
        if '-w' in opts:
            the_world = getattr(world, opts['-w'])
        else:
            the_world = getattr(world, (files[0].split('/')[-1]).split('-')[0])
        the_world = the_world()
        if '-c' in opts:
            the_car = the_world.cars[int(opts['-c'])]
        else:
            the_car = None
            for c in the_world.cars:
                if isinstance(c, car.UserControlledCar):
                    the_car = c
        T = the_car.traj.T
        train = []
        for fname in files:
            with open(fname) as f:
                us, xs = pickle.load(f)
                for t in range(T, len(xs[0])-T, T):
                    point = {
                        'x0': [xseq[t-1] for xseq in xs],
                        'u': [useq[t:t+T] for useq in us]
                    }
                    train.append(point)
        theta = utils.vector(5)
        theta.set_value(theta_val)
        r = 0.1*feature.control()
        for lane in the_world.lanes:
            r = r + theta[0]*lane.gaussian()
        for fence in the_world.fences:
            r = r + theta[1]*fence.gaussian()
        for road in the_world.roads:
            r = r + theta[2]*road.gaussian(10.)
        r = r + theta[3]*feature.speed(1.)
        for car in the_world.cars:
            if car!=the_car:
                r = r + theta[4]*car.traj.gaussian()
        res = run_irl(the_world, the_car, r, theta, train)
        results.append(res)
    print '~~~~~~~~~~~~~~~~'
    print theta_list
    print results
