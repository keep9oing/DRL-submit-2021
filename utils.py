import tsplib95
import lkh

import numpy as np

import torch
from torch.autograd import Variable


CONST = 1000.0

def V_V_cost(p, q): # visiting to visiting cost
    return np.sqrt(((p[1] - q[1])**2)+((p[0] - q[0]) **2)) * CONST

def V_C_cost(p,q): # visiting to coverage cost

    assert len(q) == 3, "to_task must be 3 elements (x,y,r)"

    d = V_V_cost(p, q) / CONST
    r = q[-1]

    return (np.sqrt(d**2 - r**2) + 7*np.pi*r) * CONST

def V_D_cost(p,q): # visiting to delivery cost

    assert len(q) == 3, "to_task must be 3 elements (x,y,l)"

    d = V_V_cost(p, q) / CONST
    l = q[-1]

    return (d+l) * CONST

def cal_cost(source_task, target_task):

    # Task dependence type
    # 001: coverage, 010: visit, 100: Pick&Place
    if   target_task[-1] == 1: # To Coverage(circle)
        TD_type = "C"
    elif target_task[-2] == 1: # To Visiting
        TD_type = "V"
    elif target_task[-3] == 1: # To Delivery
        TD_type = "D"
    else: # To Depot
        TD_type = "V"

    # from task feature selection
    if   source_task[-1] == 1 or source_task[-2] == 1: # Area or point, ciritical position
        from_task = source_task[:2]
    elif source_task[-3] == 1: # Delivery, place position
        from_task = source_task[3:5]
    else: # Depot
        from_task = source_task[:2]

    # To task feature selection
    if   target_task[-1] == 1 or target_task[-3] == 1: # Area, center position + area_info / Delivery, pick position + area_info
        to_task = target_task[:3]
    elif target_task[-2] == 1: # point, visit position / Delivery, pick position
        to_task = target_task[:2]
    else: # Depot
        to_task = target_task[:2]

    if TD_type == "V":
        cost = V_V_cost(from_task, to_task)
    elif TD_type == "C":
        cost = V_C_cost(from_task, to_task)
    elif TD_type == "D":
        cost = V_D_cost(from_task, to_task)
    else:
        raise NotImplementedError

    return cost


def get_ref_reward(pointset):

    if isinstance(pointset, torch.cuda.FloatTensor) or  isinstance(pointset, torch.FloatTensor):
        pointset = pointset.detach().numpy()

    num_points = len(pointset)
    ret_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            if i == j:
                continue
            ret_matrix[i, j] = cal_cost(pointset[i], pointset[j])

    solver_path = './LKH'

    problem_instance = tsplib95.models.StandardProblem()
    problem_instance.type = 'ATSP'
    problem_instance.dimension = ret_matrix.shape[-1]
    problem_instance.edge_weight_type = 'EXPLICIT'
    problem_instance.edge_weight_format = 'FULL_MATRIX'
    problem_instance.edge_weights = ret_matrix.tolist()

    q = lkh.solve(solver_path, problem=problem_instance, runs=100, max_candidates="6 symmetric", move_type=3, patching_c=3, patching_a=2, trace_level=0)
    q = np.array(q[0])-1

    cost = 0
    for i in range(num_points):
        cost += ret_matrix[q[i], q[(i+1) % num_points]]

    return cost / CONST, ret_matrix, q

def Trash(pointset, sol):

    if isinstance(pointset, torch.cuda.FloatTensor) or  isinstance(pointset, torch.FloatTensor):
        pointset = pointset.detach().numpy()

    num_points = len(pointset)
    ret_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            if i == j:
                continue
            ret_matrix[i, j] = cal_cost(pointset[i], pointset[j])

    # q = elkai.solve_float_matrix(ret_matrix, runs=100) # Output: [0, 2, 1]
    cost = 0
    for i in range(num_points):
        cost += ret_matrix[sol[i], sol[(i+1) % num_points]]

    return cost / CONST, sol

def Trash2(sample_solution):
    """
    [Input]
        sample_solution: batch x seq_len x feature_size
    [Return]
        cost: batch
    """

    batch_size, seq_len, _ = sample_solution.size()

    cost = Variable(torch.zeros([batch_size]))

    if isinstance(sample_solution, torch.cuda.FloatTensor):
        cost = cost.cuda()

    for i in range(seq_len -1):
        """
        current_solution: batch x feature_size
        next_solution:    batch x feature_size
        """
        current_solution = sample_solution[:, i, :]
        next_solution    = sample_solution[:, i + 1, :]

        cost += Batch_cal_cost(current_solution, next_solution, batch_size)

    cost += Batch_cal_cost(sample_solution[:, seq_len - 1, :], sample_solution[:, 0, :], batch_size)

    return cost


def Batch_V_V_cost(p, q): # visiting to visiting cost
    """
    [Input]
        p: (arbitrary) x 2
        q: (arbitrary) x 3
    [Return]
        cost: (arbitrary)
    """
    return torch.sqrt(((p[:,1] - q[:,1])**2)+((p[:,0] - q[:,0]) **2))

def Batch_V_C_cost(p,q): # visiting to coverage cost
    """
    [Input]
        p: (arbitrary) x 3
        q: (arbitrary) x 3
    [Return]
        cost: (arbitrary)
    """

    assert q.shape[-1] == 3, "to_task must be 3 elements (x,y,r)"

    d = Batch_V_V_cost(p, q)
    r = q[:,-1]

    return (torch.sqrt(d**2 - r**2) + 7*np.pi*r)

def Batch_V_D_cost(p,q): # visiting to delivery cost
    """
    [Input]
        p: (arbitrary) x 3
        q: (arbitrary) x 3
    [Return]
        cost: (arbitrary)
    """
    assert q.shape[-1] == 3, "to_task must be 3 elements (x,y,l)"

    d = Batch_V_V_cost(p, q)
    l = q[:,-1]

    return (d+l)

def Batch_cal_cost(source_task, target_task, batch_size):
    """
    [Input]
        source_task: batch x feature_size
        target_task: batch x feature_size
        batch_size:  scalar(batch_size)
    [Return]
        cost: batch
    """

    # Task dependence type
    # 000: Depot, 001: coverage, 010: visit, 100: Pick&Place
    # Considering depot looks unnecessary but considering capacity or multi depot problem, it can be used.
    # What a hardcoding..

    if isinstance(source_task, torch.cuda.FloatTensor):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    from_task = torch.zeros(batch_size, 2, device=device)
    to_task = torch.zeros(batch_size, 3, device=device)
    cost = torch.zeros(batch_size, device=device)

    # extract important feature from source_task
    # TODO: should be devided when be more complex
    from_point_mask = torch.logical_or(source_task[:,-1] == 1, source_task[:,-2] == 1)
    from_task[from_point_mask] = source_task[from_point_mask][:,:2]

    from_delivery_mask = source_task[:,-3] == 1
    from_task[from_delivery_mask] = source_task[from_delivery_mask][:,3:5]

    from_depot_mask = (source_task[:,-3:] == torch.FloatTensor([0,0,0]).to(device)).all(dim=1)
    from_task[from_depot_mask] = source_task[from_depot_mask][:,:2]


    # extract important feature from target_task
    to_coverage_mask = target_task[:,-1] == 1
    to_task[to_coverage_mask] = target_task[to_coverage_mask][:,:3]

    to_point_mask = target_task[:,-2] == 1
    to_task[to_point_mask] = target_task[to_point_mask][:,:3]

    to_delivery_mask = target_task[:,-3] == 1
    to_task[to_delivery_mask] = target_task[to_delivery_mask][:,:3]

    to_depot_mask = (target_task[:,-3:] == torch.FloatTensor([0,0,0]).to(device)).all(dim=1)
    to_task[to_depot_mask] = target_task[to_depot_mask][:,:3]

    V_C_mask = to_coverage_mask
    V_V_mask = torch.logical_or(to_depot_mask, to_point_mask)
    V_D_mask = to_delivery_mask

    assert torch.count_nonzero(V_C_mask + V_V_mask + V_D_mask) == batch_size, "Masking for cost calculating is something wrong"

    if V_C_mask.any():
        cost[V_C_mask] = Batch_V_C_cost(from_task[V_C_mask], to_task[V_C_mask])
    if V_V_mask.any():
        cost[V_V_mask] = Batch_V_V_cost(from_task[V_V_mask], to_task[V_V_mask])
    if V_D_mask.any():
        cost[V_D_mask] = Batch_V_D_cost(from_task[V_D_mask], to_task[V_D_mask])

    return cost

def get_solution_reward(sample_solution):
    """
    [Input]
        sample_solution: batch x seq_len x feature_size
    [Return]
        cost: batch
    """

    batch_size, seq_len, _ = sample_solution.size()

    cost = Variable(torch.zeros([batch_size]))

    if isinstance(sample_solution, torch.cuda.FloatTensor):
        cost = cost.cuda()

    for i in range(seq_len -1):
        """
        current_solution: batch x feature_size
        next_solution:    batch x feature_size
        """
        current_solution = sample_solution[:, i, :]
        next_solution    = sample_solution[:, i + 1, :]

        cost += Batch_cal_cost(current_solution, next_solution, batch_size)

    cost += Batch_cal_cost(sample_solution[:, seq_len - 1, :], sample_solution[:, 0, :], batch_size)

    return cost

