from __future__ import absolute_import

import numpy as np
import random
import os
import sys

from sklearn.cluster import k_means

import external.PyMiniSolvers.minisolvers as minisolvers
import torch
import subprocess

def solve_sat(n_vars, iclauses):
    solver = minisolvers.MinisatSolver()
    for i in range(n_vars): solver.new_var(dvar=True)
    for iclause in iclauses: solver.add_clause(iclause)
    is_sat = solver.solve()
    
    sol = list(solver.get_model()) if is_sat else None

    return is_sat, sol


def generate_k_iclause(n, k):
    vs = np.random.choice(n, size=min(n, k), replace=False)
    return [v + 1 if random.random() < 0.5 else -(v + 1) for v in vs]


def gen_full_iclause_pair(args, n_var):
    solver = minisolvers.MinisatSolver()
    for i in range(n_var):
        solver.new_var(dvar=True)
    
    # Generate clause
    iclauses_sat = []
    iclauses_unsat = []
    while True:
        k_base = 1 if random.random() < args.p_k_2 else 2
        k = k_base + np.random.geometric(args.p_geo)

        # 2022.07.19 random k in [1, n_var] ?
        # k = random.randint(1, n_var)
        
        iclause = generate_k_iclause(n_var, k)
        solver.add_clause(iclause)

        if solver.solve():
            iclauses_sat.append(iclause)
            iclauses_unsat.append(iclause)
        else:
            iclauses_unsat.append(iclause)
            break

    return n_var, iclauses_sat, iclauses_unsat

def gen_iclauses(args, n_var, n_clause):
    cnf = []
    solver = minisolvers.MinisatSolver()
    for i in range(n_var):
        solver.new_var(dvar=True)

    for clause_idx in range(n_clause):
        k_base = 1 if random.random() < args.p_k_2 else 2
        k = k_base + np.random.geometric(args.p_geo)
        clause = generate_k_iclause(n_var, k)
        cnf.append(clause)
        solver.add_clause(clause)
    
    sat = solver.solve()
    return sat, cnf

def gen_iclause_pair(args, n_var):
    if args.cv_ratio == -1:
        return gen_full_iclause_pair(args, n_var)
    else:
        sat, sat_cnf = gen_iclauses(args, n_var, int(args.cv_ratio * n_var))
        while not sat:
            sat, sat_cnf = gen_iclauses(args, n_var, int(args.cv_ratio * n_var))
        sat, unsat_cnf = gen_iclauses(args, n_var, int(args.cv_ratio * n_var))
        while sat:
            sat, unsat_cnf = gen_iclauses(args, n_var, int(args.cv_ratio * n_var))
        return n_var, sat_cnf, unsat_cnf

def window_solver(args, n_var, cnf):
    window_size = args.windows_size
    y = []
    while window_size <= len(cnf):
        for start_idx in range(0, len(cnf), window_size):
            solver = minisolvers.MinisatSolver()
            for i in range(n_var):
                solver.new_var(dvar=True)
            for clause_idx in range(start_idx, start_idx+window_size, 1):
                solver.add_clause(cnf[clause_idx])
            if solver.solve():
                y.append(1)
            else:
                y.append(0)
        window_size *= args.windows_size
    return y

# the utility function for Circuit-SAT
def get_sub_cnf(cnf, var, is_inv):
    res_cnf = []
    if not is_inv:
        for clause in cnf:
            if not var in clause:
                tmp_clause = clause.copy()
                for idx, ele in enumerate(tmp_clause):
                    if ele == -var:
                        del tmp_clause[idx]
                res_cnf.append(tmp_clause)
    else:
        for clause in cnf:
            if not -var in clause:
                tmp_clause = clause.copy()
                for idx, ele in enumerate(tmp_clause):
                    if ele == var:
                        del tmp_clause[idx]
                res_cnf.append(tmp_clause)
    return res_cnf

def two_fanin_gate(po_idx, fan_in_list, x, edge_index, gate_type):
    gate_list = fan_in_list.copy()
    new_gate_list = []

    while True:
        if len(gate_list) + len(new_gate_list) == 2:
            for gate_idx in gate_list:
                edge_index.append([gate_idx, po_idx])
            for gate_idx in new_gate_list:
                edge_index.append([gate_idx, po_idx])
            break
        if len(gate_list) == 0:
            gate_list = new_gate_list.copy()
            new_gate_list.clear()
        elif len(gate_list) == 1:
            new_gate_list.append(gate_list[0])
            gate_list = new_gate_list.copy()
            new_gate_list.clear()
        else:
            new_gate_idx = len(x)
            x.append(gate_type)
            edge_index.append([gate_list[0], new_gate_idx])
            edge_index.append([gate_list[1], new_gate_idx])
            gate_list = gate_list[2:]
            new_gate_list.append(new_gate_idx)


def save_cnf(cnf, cnf_idx, x, edge_index, inv2idx):
    cnf_fan_in_list = []
    for clause in cnf:
        if len(clause) == 0:
            continue
        elif len(clause) == 1:
            if clause[0] < 0:
                cnf_fan_in_list.append(inv2idx[abs(clause[0])])
            else:
                cnf_fan_in_list.append(clause[0])
        else:
            clause_idx = len(x)
            x.append(one_hot_gate_type('OR'))
            cnf_fan_in_list.append(clause_idx)
            clause_fan_in_list = []
            for ele in clause:
                if ele < 0:
                    clause_fan_in_list.append(inv2idx[abs(ele)])
                else:
                    clause_fan_in_list.append(ele)
            two_fanin_gate(clause_idx, clause_fan_in_list, x, edge_index, x[clause_idx])

    x[cnf_idx] = one_hot_gate_type('AND')
    two_fanin_gate(cnf_idx, cnf_fan_in_list, x, edge_index, x[cnf_idx])

def merge_cnf(cnf):
    res = []
    clause2bool = {}
    for clause in cnf:
        tmp_clause = tuple(clause)
        if not tmp_clause in clause2bool:
            clause2bool[tmp_clause] = True
            res.append(clause)
    return res

def recursion_generation(cnf, cnf_idx, current_depth, max_depth, n_vars, x, edge_index, inv2idx):
    '''
    Expand the CNF as binary tree
    The expanded CNF can be writen as:
        CNF = OR(B_T, B_F)
        B_T = AND(exp_T_CNF, var)
        B_F = AND(exp_F_CNF, var_inv)
        # exp_T_CNF, exp_F_CNF are new CNFs
    Input:
        cnf: iclauses
        cnf_idx: the cnf PO index in x
        current_depth: current expand time
        max_depth: maximum expand time
        n_vars: number of variables
        x: nodes
        edge_index: edge
        inv2idx: PI_inv index
    '''

    ####################
    # Store as CNF
    ####################
    if current_depth == max_depth:
        save_cnf(cnf, cnf_idx, x, edge_index, inv2idx)
        return

    ####################
    # Sort
    ####################
    var_times = [0] * (n_vars + 1)
    for idx in range(1, n_vars + 1, 1):
        for clause in cnf:
            if idx in clause:
                var_times[abs(idx)] += 1

    var_sort = np.argsort(var_times)
    most_var = var_sort[-1]
    if var_times[most_var] == 0:
        save_cnf(cnf, cnf_idx, x, edge_index, inv2idx)
        return


    ####################
    # Expansion
    ####################
    for most_var in var_sort[::-1]:
        var_idx = most_var
        next_var = False
        # Get sub-CNFs
        exp_T_cnf = get_sub_cnf(cnf, most_var, 0)
        exp_F_cnf = get_sub_cnf(cnf, most_var, 1)

        for clause in exp_T_cnf:
            if len(clause) == 0:
                next_var = True
                break
        for clause in exp_F_cnf:
            if len(clause) == 0:
                next_var = True
                break
        if not next_var:
            break
    if most_var == 0:
        save_cnf(cnf, cnf_idx, x, edge_index, inv2idx)
        return

    if not most_var in inv2idx:
        inv2idx[most_var] = len(x)
        x.append(one_hot_gate_type('NOT'))
        edge_index.append([most_var, inv2idx[most_var]])
    var_inv_idx = inv2idx[most_var]

    exp_T_cnf = merge_cnf(exp_T_cnf)
    exp_F_cnf = merge_cnf(exp_F_cnf)

    # ------------------------------------------
    # Construct (exp_T_CNF) and (B_T)
    if len(exp_T_cnf) == 0:
        edge_index.append([var_idx, cnf_idx])
    elif len(exp_T_cnf) == 1:
        # Construct (B_T): B_T = AND(var_idx, exp_T)
        B_T_idx = len(x)
        x.append(one_hot_gate_type('AND'))
        exp_T_cnf = exp_T_cnf[0]
        if len(exp_T_cnf) == 1:  # The clause only have one var
            exp_T_idx = exp_T_cnf[0]
            if exp_T_idx < 0:
                exp_T_idx = inv2idx[abs(exp_T_idx)]
        else:  # The clause have many vars
            exp_T_idx = len(x)
            x.append(one_hot_gate_type('OR'))
            for ele in exp_T_cnf:
                if ele < 0:
                    ele_idx = inv2idx[abs(ele)]
                else:
                    ele_idx = ele
                edge_index.append([ele_idx, exp_T_idx])
        edge_index.append([exp_T_idx, B_T_idx])
        edge_index.append([var_idx, B_T_idx])
        edge_index.append([B_T_idx, cnf_idx])
    else:
        # Construct(exp_T_CNF)
        exp_T_cnf_idx = len(x)
        x.append(one_hot_gate_type('OR'))
        recursion_generation(exp_T_cnf, exp_T_cnf_idx, current_depth + 1, max_depth,
                             n_vars, x, edge_index, inv2idx)
        # Construct (B_T)
        B_T_idx = len(x)
        x.append(one_hot_gate_type('AND'))
        edge_index.append([exp_T_cnf_idx, B_T_idx])
        edge_index.append([var_idx, B_T_idx])
        edge_index.append([B_T_idx, cnf_idx])

    # ------------------------------------------
    # Construct (exp_F_CNF) and (B_F)
    if len(exp_F_cnf) == 0:
        edge_index.append([var_inv_idx, cnf_idx])
    elif len(exp_F_cnf) == 1:
        # Construct (B_F): B_F = AND(var_idx, exp_F)
        B_F_idx = len(x)
        x.append(one_hot_gate_type('AND'))
        exp_F_cnf = exp_F_cnf[0]
        if len(exp_F_cnf) == 1:  # The clause only have one var
            exp_F_idx = exp_F_cnf[0]
            if exp_F_idx < 0:
                exp_F_idx = inv2idx[abs(exp_F_idx)]
        else:  # The clause have many vars
            exp_F_idx = len(x)
            x.append(one_hot_gate_type('OR'))
            for ele in exp_F_cnf:
                if ele < 0:
                    ele_idx = inv2idx[abs(ele)]
                else:
                    ele_idx = ele
                edge_index.append([ele_idx, exp_F_idx])
        edge_index.append([exp_F_idx, B_F_idx])
        edge_index.append([var_inv_idx, B_F_idx])
        edge_index.append([B_F_idx, cnf_idx])
    else:
        # Construct(exp_F_CNF)
        exp_F_cnf_idx = len(x)
        x.append(one_hot_gate_type('OR'))
        recursion_generation(exp_F_cnf, exp_F_cnf_idx, current_depth + 1, max_depth,
                             n_vars, x, edge_index, inv2idx)
        # Construct (B_F)
        B_F_idx = len(x)
        x.append(one_hot_gate_type('AND'))
        edge_index.append([exp_F_cnf_idx, B_F_idx])
        edge_index.append([var_inv_idx, B_F_idx])
        edge_index.append([B_F_idx, cnf_idx])


def one_hot_gate_type(gate_type):
    res = []
    if gate_type == 'PI':
        res = [1, 0, 0, 0]
    elif gate_type == 'AND':        res = [0, 1, 0, 0]
    elif gate_type == 'OR':
        res = [0, 0, 1, 0]
    elif gate_type == 'NOT':
        res = [0, 0, 0, 1]
    else:
        print('[ERROR] Unknown gate type')
    return res



def write_dimacs_to(n_vars, iclauses, out_filename):
    with open(out_filename, 'w') as f:
        f.write("p cnf %d %d\n" % (n_vars, len(iclauses)))
        for c in iclauses:
            for x in c:
                f.write("%d " % x)
            f.write("0\n")

def cnf_to_netlist(iclauses, n_vars):
    '''
    A function to parse the cnf to aig, then to circuit to `Pytorch Geometric` Data.
    Input:
        iclause: clauses list
        n_vars: number of variables
        n_clauses: number of clauses
    Return:
        x: one_hot encoding of [PI, AND, NOT]
        edge_index: edge connection pairs: each pair [x, y] from x to y
    For AIG, the nodes can be categorized as the Literal node, internal AND nodes, internal NOT node. The type values for each kind of nodes are as follows:
        * Literal input node: 0;
        * Internal AND nodes: 1;
        * Internal NOT nodes: 2;
    '''
    
    if not os.path.exists('./tmp'):
        os.makedirs('./tmp/')

    if os.getcwd()[-3:] == 'src':
        external_folder = './external/'
    else:
        external_folder = './src/external/'
    
    # step 1: store dimacs format
    dimacs_tmp = './tmp/sat.dimacs'
    write_dimacs_to(n_vars, iclauses, dimacs_tmp)
    # step 2: dimacs to aig
    aig_tmp = './tmp/sat.aig'
    subprocess.call([external_folder + "./aiger/cnf2aig/cnf2aig", dimacs_tmp, aig_tmp])
    # step 3: aig to abc opimized aig
    aig_abc_tmp = './tmp/aig_abc.aig'
    subprocess.call(["abc", "-c", "read %s; balance; \
        balance; rewrite -lz; rewrite -lz; balance; rewrite -lz; balance; cec; write %s" \
            % (aig_tmp, aig_abc_tmp)])
    # step 4: aig to aag
    aag_abc_tmp = './tmp/aig_abc.aag'
    subprocess.call([external_folder + "./aiger/aiger/aigtoaig", aig_abc_tmp, aag_abc_tmp])
    # step 4: read aag
    with open(aag_abc_tmp, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split(" ")
    assert header[0] == 'aag', 'The header of AIG file is wrong.'
    # “M”, “I”, “L”, “O”, “A” separated by spaces.
    n_variables = eval(header[1])
    n_inputs = eval(header[2])
    n_outputs = eval(header[4])
    n_and = eval(header[5])
    if n_outputs != 1 or n_variables != (n_inputs + n_and) or n_variables == n_inputs:
        return [], []
    assert n_outputs == 1, 'The AIG has multiple outputs.'
    assert n_variables == (n_inputs + n_and), 'There are unused AND gates.'
    assert n_variables != n_inputs, '# variable equals to # inputs'
    # Construct AIG graph
    x_data = []
    edge_index = []
    # node_labels = []
    not_dict = {}
    
    # Add Literal node
    for i in range(n_inputs):
        x_data.append([len(x_data), 0])
        # node_labels += [0]

    # Add AND node
    for i in range(n_inputs+1, n_inputs+1+n_and):
        x_data.append([len(x_data), 1])
        # node_labels += [1]


    # sanity-check
    for (i, line) in enumerate(lines[1:1+n_inputs]):
        literal = line.strip().split(" ")
        assert len(literal) == 1, 'The literal of input should be single.'
        assert int(literal[0]) == 2 * (i + 1), 'The value of a input literal should be the index of variables mutiplying by two.'

    literal = lines[1+n_inputs].strip().split(" ")[0]
    assert int(literal) == (n_variables * 2) or int(literal) == (n_variables * 2) + 1, 'The value of the output literal shoud be (n_variables * 2)'
    sign_final = int(literal) % 2
    index_final_and = int(literal) // 2 - 1

    for (i, line) in enumerate(lines[2+n_inputs: 2+n_inputs+n_and]):
        literals = line.strip().split(" ")
        assert len(literals) == 3, 'invalidate the definition of two-input AND gate.'
        assert int(literals[0]) == 2 * (i + 1 + n_inputs)
    var_def = lines[2+n_variables].strip().split(" ")[0]

    assert var_def == 'i0', 'The definition of variables is wrong.'
    # finish sanity-check

    # Add edge
    for (i, line) in enumerate(lines[2+n_inputs: 2+n_inputs+n_and]):
        line = line.strip().split(" ")
        # assert len(line) == 3, 'The length of AND lines should be 3.'
        output_idx = int(line[0]) // 2 - 1
        # assert (int(line[0]) % 2) == 0, 'There is inverter sign in output literal.'

        # 1. First edge
        input1_idx = int(line[1]) // 2 - 1
        sign1_idx = int(line[1]) % 2
        # If there's a NOT node
        if sign1_idx == 1:
            if input1_idx in not_dict.keys():
                not_idx = not_dict[input1_idx]
            else:
                x_data.append([len(x_data), 2])
                # node_labels += [2]
                not_idx = len(x_data) - 1
                not_dict[input1_idx] = not_idx
                edge_index += [[input1_idx, not_idx]]
            edge_index += [[not_idx, output_idx]]
        else:
            edge_index += [[input1_idx, output_idx]]


        # 2. Second edge
        input2_idx = int(line[2]) // 2 - 1
        sign2_idx = int(line[2]) % 2
        # If there's a NOT node
        if sign2_idx == 1:
            if input2_idx in not_dict.keys():
                not_idx = not_dict[input2_idx]
            else:
                x_data.append([len(x_data), 2])
                # node_labels += [2]
                not_idx = len(x_data) - 1
                not_dict[input2_idx] = not_idx
                edge_index += [[input2_idx, not_idx]]
            edge_index += [[not_idx, output_idx]]
        else:
            edge_index += [[input2_idx, output_idx]]
    
    
    if sign_final == 1:
        x_data.append([len(x_data), 2])
        # node_labels += [2]
        not_idx = len(x_data) - 1
        edge_index += [[index_final_and, not_idx]]

    return x_data, edge_index

def cnf_simulation(args, g, batch_mask, patterns):
    y = [0] * len(batch_mask)
    batch_var_node = g.forward_index[batch_mask & (g.gate_type == args.gate2index['VAR'])]
    n_vars = len(batch_var_node)
    for k in range(n_vars):
        y[batch_var_node[k]] = patterns[k]
        if y[batch_var_node[k]] == 0:
            y[batch_var_node[k]+n_vars] = 1
        else:
            y[batch_var_node[k]+n_vars] = 0
    
    for edge_idx in range(len(g.edge_index[0])):
        src = g.edge_index[0][edge_idx]
        dst = g.edge_index[1][edge_idx]
        if batch_mask[src] and batch_mask[dst]:
            if y[src] == 1 and g.gate_type[dst] == args.gate2index['CLAUSE']:
                y[dst] = 1
    
    # Check SAT
    sat = True
    for idx in range(len(g.gate_type)):
        if batch_mask[idx] and g.gate_type[idx] == args.gate2index['CLAUSE'] and y[idx] == 0:
            sat = False
            break
    return sat
