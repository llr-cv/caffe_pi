
import os,shutil
import hyperopt
import cv2
import caffe
from caffe import layers as L,params as P,proto,to_proto
from multiprocessing import Process, Queue
from google import protobuf

def get_from_solver_file(filename):
  s = proto.caffe_pb2.SolverParameter()
  with open(filename) as f:
    protobuf.text_format.Parse(f.read(), s)
  return s 

def gen_solver_file(solver_file, hyperopt_args):
  s = get_from_solver_file(solver_file)    
  s.base_lr = hyperopt_args[0]   
  s.weight_decay = hyperopt_args[1]
  s.snapshot_prefix = s.snapshot_prefix  + str(hyperopt_args[0]) +'_' + str(hyperopt_args[1])
  
  if not os.path.exists(s.snapshot_prefix):
    os.mkdir(s.snapshot_prefix)
  
  sovler_file = s.snapshot_prefix + '/solver.prototxt'
  with open(sovler_file, 'w') as f:
    f.write(str(s))
  return sovler_file

def train( hyperopt_args ):
    global options
    solver_file= options['solver']
    gpus =  options['gpus']
    timing = options['timing']
    sovler_file = gen_solver_file(solver_file, hyperopt_args)
    s = get_from_solver_file(sovler_file)
    caffe.flush_log()
    caffe.set_log_path(str(s.snapshot_prefix + '/log_'))
    # NCCL uses a uid to identify a session
    uid = caffe.NCCL.new_uid()
    q = Queue()
    procs = []
    for rank in range(len(gpus)):
        p = Process(target=solve,
                    args=(sovler_file, "", gpus, timing, uid, rank,q))
        p.daemon = True
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    loss = q.get()
    print("training end. snapshot:" + s.snapshot_prefix + ", loss: "+ str(loss))
    return loss

def time(solver, nccl):
    fprop = []
    bprop = []
    total = caffe.Timer()
    allrd = caffe.Timer()
    for _ in range(len(solver.net.layers)):
        fprop.append(caffe.Timer())
        bprop.append(caffe.Timer())
    display = solver.param.display

    def show_time():
        if solver.iter % display == 0:
            s = '\n'
            for i in range(len(solver.net.layers)):
                s += 'forw %3d %8s ' % (i, solver.net._layer_names[i])
                s += ': %.2f\n' % fprop[i].ms
            for i in range(len(solver.net.layers) - 1, -1, -1):
                s += 'back %3d %8s ' % (i, solver.net._layer_names[i])
                s += ': %.2f\n' % bprop[i].ms
            s += 'solver total: %.2f\n' % total.ms
            s += 'allreduce: %.2f\n' % allrd.ms
            print(s)

    solver.net.before_forward(lambda layer: fprop[layer].start())
    solver.net.after_forward(lambda layer: fprop[layer].stop())
    solver.net.before_backward(lambda layer: bprop[layer].start())
    solver.net.after_backward(lambda layer: bprop[layer].stop())
    solver.add_callback(lambda: total.start(), lambda: (total.stop(), allrd.start()))
    solver.add_callback(nccl)
    solver.add_callback(lambda: '', lambda: (allrd.stop(), show_time()))


def solve(proto, snapshot, gpus, timing, uid, rank,q):
    caffe.set_mode_gpu()
    caffe.set_device(gpus[rank])
    caffe.set_solver_count(len(gpus))
    caffe.set_solver_rank(rank)
    caffe.set_multiprocess(True)

    solver = caffe.SGDSolver(str(proto))
    #if snapshot and len(snapshot) != 0:
    #    solver.restore(str(snapshot))

    nccl = caffe.NCCL(solver, uid)
    nccl.bcast()

    if timing and rank == 0:
        time(solver, nccl)
    else:
        solver.add_callback(nccl)

    if solver.param.layer_wise_reduce:
        solver.net.after_backward(nccl)
    loss = solver.step_multi_gpu(solver.param.max_iter)
    if rank == 0 :
      q.put(loss)


def train_one_gpu(solver_file,  # solver proto definition
        net_file,
        snapshot,  # solver snapshot to restore
        gpu_id,  # list of device ids
        hyperopt_args):

  s = gen_solver_file(solver_file,net_file, snapshot, hyperopt_args)
  caffe.set_log_path(str(s.snapshot_prefix + '/log_'))
  caffe.set_device(gpu_id)
  caffe.set_mode_gpu()
  solver = caffe.SGDSolver(solver_file)
  loss =solver.solve()
  return loss 

def hyperopt_train():

  caffe.init_log()
  
  # define a search space
  from hyperopt import hp
  space = [ hp.uniform('base_lr', 0.01, 0.02), hp.uniform('weight_decay', 5e-4 , 5e-3) ]

  # minimize the objective over the space
  from hyperopt import fmin, tpe
  best = fmin(train, space, algo=tpe.suggest, max_evals=5)

  print best
  print hyperopt.space_eval(space, best)


options ={ "solver":"", "gpus":[1,2], "timing":False}

def main():
  global options
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--solver", required=True, help="Solver proto definition.")
  parser.add_argument("--gpus", type=int, nargs='+', default=[0], help="List of device ids.")
  parser.add_argument("--timing", action='store_true', help="Show timing info.")
  args = parser.parse_args()

  if not args.solver and not os.path.exists(args.solver):
    print("please input exist Solver file")
    parser.print_help()
    return

  if not args.gpus:
    print("please input gpus")
    parser.print_help()
    return
  print(args.gpus)
  options['solver'] = args.solver
  options['gpus'] = args.gpus
  options['timing'] = args.timing
  hyperopt_train()

if __name__ == "__main__":
  main()