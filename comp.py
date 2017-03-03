import pickle
import os
import sys
import getopt
import world
import car
import utils
import feature
import theano as th
import theano.tensor as tt
import theano.tensor.nlinalg as tn

def extract_traj(filepath):
    train = []
    T=5
    with open(filepath) as f:
        us, xs = pickle.load(f)
        for t in range(T, len(xs[0])-T, T):
            point = {
                'x0': [xseq[t-1] for xseq in xs],
                'u': [useq[t:t+T] for useq in us]
            }
            train.append(point)
    return train


def score_traj(world, car, reward, theta, data):
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
    print(optimizer.f_and_df(theta))


if __name__ == '__main__':
    thetas =  [[3., -50., 10., 20., -60.], [-119.19716735,  -95.85620629,   16.1464593 ,   24.67625254,
        -35.25442831], [  1.57913037,  -0.16351292,  31.55001674,  29.48833567, -43.45338029], [-17.4252014 , -57.41858823,   0.955136  ,  58.84451896, -28.6585965 ], [ 10.87842463,  -4.62697363,  33.26893187,  34.81915794, -31.29544288], [-14.27781722, -21.83462477,  11.89256159,  62.64876998,  -9.7886389 ], [-124.1838848 ,  -92.72151612,  -11.51401053,   26.63874715,
         33.87774694], [ -5.94369924,   1.36314832,  23.26638451,  23.37525333,  37.70258904], [  1.34590050e+00,  -4.42538332e-03,   3.97445470e+01,
         3.06553671e+01,  -2.74143261e+01], [ -4.83309446,   2.704638  ,  -3.23562229,  50.55933541,  34.98453259], [ -17.82166247, -110.81412015,  -15.44308339,   26.90499183,
         10.96541577]]

    demo_x = []
    res_x = []
    world_demo = None 
    world_res=None
    car_demo = None
    car_res = None

    demos_folder = sys.argv[1] # original human demonstrations
    res_folder = sys.argv[2] # results of Simple Optimizer car with different thetas
    
    for demo in os.listdir(demos_folder):
        demo_x.append(extract_traj(demos_folder+demo))
        world_demo = getattr(world, (demo.split('/')[-1]).split('-')[0])()
        car_demo = world_demo.cars[0]
    for res in os.listdir(res_folder):
        res_x.append(extract_traj(res_folder+res))
        world_res = getattr(world, (res.split('/')[-1]).split('-')[0])()
        car_res = world_res.cars[0]

    T = car_res.traj.T

    print T

    for theta in thetas:

        r = 0.1*feature.control()
        for lane in world_demo.lanes:
            r = r + theta[0]*lane.gaussian()
        for fence in world_demo.fences:
            r = r + theta[1]*lane.gaussian()
        for road in world_demo.roads:
            r = r + theta[2]*road.gaussian(10.)
        r = r + theta[3]*feature.speed(1.)
        for car in world_demo.cars:
            if car!=car_demo:
                r = r + theta[4]*car.traj.gaussian()

        weights = []
        for demo in demo_x:
            score_traj(world_demo, car_demo, r, theta, demo) 
        

