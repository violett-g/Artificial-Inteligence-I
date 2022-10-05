import itertools
import random
import re
import string
from collections import defaultdict, Counter
from functools import reduce
from operator import eq, neg

from sortedcontainers import SortedSet
import search
from utils import argmin_random_tie, count, first, extend, linear_kernel
import os
import time
from tabulate import tabulate
import sys
import re
from interruptingcow import timeout
from termcolor import colored


#STRUCTURES
v_list = []
ctr_dict = {} #item=tuple(x,y,operator,k)
dom_list =[] #item = tuple(dom_name, num_of_values, list_of_values)
var_list = [] #item = tuple(var_name, dom_name)
neighbors = {} #dictionary for neighbor values, item = value:list_of_neighbour_values
assigments = {} #dictionary for assigments, item = var:val
dom_dict = {} #dictionary for domain of values, item =  val:domain
checks = 1

#STRUCTURE FUNCTIONS
def update_weight(X, Y):
    for item in ctr_dict.get(X): #find constraint X:[Y,op,k,w]
        y,op,k,w = item
        if y == Y:
            w +=1
            new = (y,op,k,w)
            list = ctr_dict.get(X)
            list.remove(item)
            list.append(new)
            break
    for item in ctr_dict.get(Y): #find constraint Y:[X,op,k,w]
        x,op,k,w = item
        if x == X:
            w +=1
            new = (x,op,k,w)
            list = ctr_dict.get(Y)
            list.remove(item)
            list.append(new)
            break

def constraints(A,a,B,b):
    global checks
    checks += 1
    #if str(B) in neighbors.get(str(A)):
    A_ctr = ctr_dict.get(str(A))
    #if str(a) in dom_dict.get(str(A)) and str(b) in dom_dict.get(str(B)): #check if given values exist on the domains of varables
    for y,op,k,w in A_ctr:
        if y == str(B): #test constraints that include B
            if op == '=':
                if abs(int(a)-int(b)) == int(k): #checki if the constraint is satisfied
                    return True
            elif op == '>':
                if abs(int(a)-int(b)) > int(k): #check if the constraint is satisfied
                    return True
    return False

#________________________________________
#OPEN FILES/ COLLECT DATA
def structures(link,instance):#give instance as parameter

    for dir_path,dir_name,files in os.walk(link):
        for file in files:
            file_name = dir_path +'/'+file
            if file_name.find(instance) > 0:
                with open(file_name, 'r') as f:
                    if file_name.find("ctr") > 0 :
                        lines=[line.rstrip('\n') for line in f] #list of file lines
                        for line in lines:#line = x,y,operator,k or num_of_constraints
                            split_data = line.split() #split returns a list of strings-> [x,y,operator,k]
                            if len(split_data) >1:
                                k1_ctr = (split_data[1],split_data[2],split_data[3],1)
                                k2_ctr = (split_data[0],split_data[2],split_data[3],1)
                                
                                key1 = split_data[0]
                                ex_key1 = ctr_dict.get(key1)
                                if ex_key1 is None:
                                    ctr_dict.update({key1:[k1_ctr]})
                                else:
                                    ex_key1.append(k1_ctr)
                                    ctr_dict.update({key1:ex_key1})
                                    

                                key2= split_data[1]
                                ex_key2 = ctr_dict.get(key2)
                                if ex_key2 is None:
                                    ctr_dict.update({key2:[k2_ctr]})
                                else:
                                    ex_key2.append(k2_ctr)
                                    ctr_dict.update({key2:ex_key2})
                                    

                    elif file_name.find("dom") >0:
                        lines=[line.rstrip('\n') for line in f] #list of file lines
                        for line in lines:#line = num_of domains or dom_num+num_of_values+values
                            split_data = line.split() #split returns a list of strings-> [dom_num,num_of_dom,value0,value1,..,valuen]
                            if len(split_data) >1:
                                dom_name = split_data[0]
                                num_of_values = split_data[1]
                                #remove dom_name and num_of_values in order to have a list of values
                                del split_data[0]
                                del split_data[0]
                                dom = (dom_name,num_of_values,split_data)#create a tupple of dom information to insert on list
                                dom_list.append(dom)
                                    
                    elif file_name.find("var") >0:
                        lines=[line.rstrip('\n') for line in f] #list of file lines
                        for line in lines:#line = var+domain_name or num_of_vars
                            split_data = line.split() #split returns a list of strings-> [var,dom_name]
                            if len(split_data) >1:
                                v_list.append(split_data[0])
                                var_dom=tuple(split_data)
                                var_list.append(var_dom)
                    #add instance to dictionary

def other_structs():
    for var in var_list:
        for dom in dom_list:
            if var[1] == dom[0]:
                dict1={var[0]:dom[2]}
                dom_dict.update(dict1)


    #a value y is a neigbor of value x, if y and x has a constreint relation that means 
    # for each x(key) on ctr_list create a list of y(values of key)
    for var in var_list:
        neighbors_list = []
        x_ctr=ctr_dict.get(var[0])
        for y,op,k,w in x_ctr:
            neighbors_list.append(y)
            neighbors.update({var[0]:neighbors_list})

    def is_var(x):
        for var,d in var_list:
            if x== var:
                return True

    for x in ctr_dict:
        if(is_var(x)):
            neighbors_list = []
            x_ctr=ctr_dict.get(x)
            for y,op,k,w in x_ctr:
                neighbors_list.append(y)
                neighbors.update({x:neighbors_list})
    
    # for var,d in var_list:
    #     if var not in neighbors:
    #         neighbors_list = []
    #         x_ctr=ctr_dict.get(var[0])
    #         for y,op,k,w in x_ctr:
    #             neighbors_list.append(y)
    #             neighbors.update({var:neighbors_list})
    #             if y in v_list:
    #                 if y not in neighbors:
    #                     y_n_list = []
    #                     y_n_list.append(var)
    #                     neighbors.update({y:y_n_list})
    #                 else:
    #                     y_n_list = neighbors.get(y)
    #                     y_n_list.append(var)
    #                     neighbors.update({y:y_n_list})

    #     else:
    #         neighbors_list = neighbors.get(var)
    #         x_ctr=ctr_dict.get(var[0])
    #         for y,op,k,w in x_ctr:
    #             neighbors_list.append(y)
    #             neighbors.update({var:neighbors_list})
    #             if y in v_list:
    #                 if y not in neighbors:
    #                     y_n_list = []
    #                     y_n_list.append(var)
    #                     neighbors.update({y:y_n_list})
    #                 else:
    #                     y_n_list = neighbors.get(y)
    #                     y_n_list.append(var)
    #                     neighbors.update({y:y_n_list})





#_______________________________________________________________
#CSP

class CSP(search.Problem):

    def __init__(self, variables, domains, neighbors, constraints):
        """Construct a CSP problem. If variables is empty, it becomes domains.keys()."""
        super().__init__(())
        variables = variables or list(domains.keys())
        self.variables = variables
        self.domains = domains
        self.neighbors = neighbors
        self.constraints = constraints
        self.curr_domains = None
        self.nassigns = 0

    def assign(self, var, val, assignment):
        """Add {var: val} to assignment; Discard the old value if any."""
        assignment[var] = val
        self.nassigns += 1

    def unassign(self, var, assignment):
        """Remove {var: val} from assignment.
        DO NOT call this if you are changing a variable to a new value;
        just call assign for that."""
        if var in assignment:
            del assignment[var]

    def nconflicts(self, var, val, assignment):
        """Return the number of conflicts var=val has with other variables."""

        # Subclasses may implement this more efficiently
        def conflict(var2):
            return var2 in assignment and not self.constraints(var, val, var2, assignment[var2])

        return count(conflict(v) for v in self.neighbors[var])

    def display(self, assignment):
        """Show a human-readable representation of the CSP."""
        # Subclasses can print in a prettier way, or display with a GUI
        print(assignment)

    # These methods are for the tree and graph-search interface:

    def actions(self, state):
        """Return a list of applicable actions: non conflicting
        assignments to an unassigned variable."""
        if len(state) == len(self.variables):
            return []
        else:
            assignment = dict(state)
            var = first([v for v in self.variables if v not in assignment])
            return [(var, val) for val in self.domains[var]
                    if self.nconflicts(var, val, assignment) == 0]

    def result(self, state, action):
        """Perform an action and return the new state."""
        (var, val) = action
        return state + ((var, val),)

    def goal_test(self, state):
        """The goal is to assign all variables, with all constraints satisfied."""
        assignment = dict(state)
        return (len(assignment) == len(self.variables)
                and all(self.nconflicts(variables, assignment[variables], assignment) == 0
                        for variables,d in self.variables))

    # These are for constraint propagation

    def support_pruning(self):
        """Make sure we can prune values from domains. (We want to pay
        for this only if we use it.)"""
        if self.curr_domains is None:
            self.curr_domains = {v: list(self.domains[v]) for v,d in self.variables}

    def suppose(self, var, value):
        """Start accumulating inferences from assuming var=value."""
        self.support_pruning()
        removals = [(var, a) for a in self.curr_domains[var] if a != value]
        self.curr_domains[var] = [value]
        return removals

    def prune(self, var, value, removals):
        """Rule out var=value."""
        self.curr_domains[var].remove(value)
        if removals is not None:
            removals.append((var, value))

    def choices(self, var):
        """Return all values for var that aren't currently ruled out."""
        return (self.curr_domains or self.domains)[var]
        
    def infer_assignment(self):
        """Return the partial assignment implied by the current inferences."""
        self.support_pruning()
        return {v: self.curr_domains[v][0]
                for v in self.variables if 1 == len(self.curr_domains[v])}

    def restore(self, removals):
        """Undo a supposition and all inferences from it."""
        for B, b in removals:
            self.curr_domains[B].append(b)

    # This is for min_conflicts search

    def conflicted_vars(self, current):
        """Return a list of variables in current assignment that are in conflict"""
        return [var for var,d in self.variables
                if self.nconflicts(var, current[var], current) > 0]

    #----My Functions------#
    def is_empty(self,var):
        #return True if the domain of the given variable is empty
        if len(self.curr_domains[var]) == 0:
            return True



# ______________________________________________________________________________
# Constraint Propagation with AC3


def no_arc_heuristic(csp, queue):
    return queue


def dom_j_up(csp, queue):
    return SortedSet(queue, key=lambda t: neg(len(csp.curr_domains[t[1]])))


def AC3(csp, queue=None, removals=None, arc_heuristic=dom_j_up):
    """[Figure 6.3]"""
    if queue is None:
        queue = {(Xi, Xk) for Xi,xd in csp.variables for Xk in csp.neighbors[Xi]}
    csp.support_pruning()
    queue = arc_heuristic(csp, queue)
    checks = 0
    while queue:
        (Xi, Xj) = queue.pop()
        revised, checks = revise(csp, Xi, Xj, removals, checks)
        if revised:
            if not csp.curr_domains[Xi]:
                return False, checks  # CSP is inconsistent
            for Xk in csp.neighbors[Xi]:
                if Xk != Xj:
                    queue.add((Xk, Xi))
    return True, checks  # CSP is satisfiable


def revise(csp, Xi, Xj, removals, checks=0):
    """Return true if we remove a value."""
    revised = False
    for x in csp.curr_domains[Xi][:]:
        
        # If Xi=x conflicts with Xj=y for every possible y, eliminate Xi=x
        # if all(not csp.constraints(Xi, x, Xj, y) for y in csp.curr_domains[Xj]):
        conflict = True
        for y in csp.curr_domains[Xj]:
            if csp.constraints(Xi, x, Xj, y):
                conflict = False
            checks += 1
            if not conflict:
                break
        if conflict:
            csp.prune(Xi, x, removals)
            revised = True
    if csp.is_empty(Xi):#check if domain Xi wipes uot
        update_weight(Xi,Xj) #increase wieght of constraint Xi-Xj
    return revised, checks

#________________________________________________________________
# CSP Backtracking Search

# Variable ordering

def dom_wdeg(assigment,csp):
    ratio = 100000000000000000000000000000000000000000000000
    dom_size = 0
    min_var = None
    wdeg = 1 
 
    for var,d in csp.variables:
        if var not in assigment:
            sum=0
            for y,op,k,w in ctr_dict.get(var):
                sum += w #sum of the constraint weight of all the ctr that include var
            var_neighbors = neighbors.get(var) #return the list of the neighbours of the current value
            unassigned = False
            for neighbor in var_neighbors: #check if there is at least one neigbour with unassigned value
                if neighbor not in assigment:
                    unassigned = True
            if unassigned == True:
                for neighbour in var_neighbors:
                    neighbor_ctr = ctr_dict.get(neighbour)
                    for y,op,k,w in neighbor_ctr:
                        sum += w #sum of the constraint weight of all the ctr that include var
            if csp.curr_domains is not None:
                dom_size = len(csp.curr_domains[var])
            else:
                dom_size = len(csp.domains[var])
                #ratio = min(ratio, (dom_size/sum))
            if ratio > dom_size/wdeg:
                ratio = dom_size/wdeg
                min_var = var 
    
    return min_var


# Value ordering


def unordered_domain_values(var, assignment, csp):
    """The default value order."""
    return csp.choices(var)


# def lcv(var, assignment, csp):
#     """Least-constraining-values heuristic."""
#     return sorted(csp.choices(var), key=lambda val: csp.nconflicts(var, val, assignment))


# Inference


def no_inference(csp, var, value, assignment, removals):
    return True


def forward_checking(csp, var, value, assignment, removals):
    """Prune neighbor values inconsistent with var=value."""
    csp.support_pruning()
    for B in csp.neighbors[var]:
        if B not in assignment:
            for b in csp.curr_domains[B][:]:
                if not csp.constraints(var, value, B, b):
                    csp.prune(B, b, removals)
            if not csp.curr_domains[B]:
                return False
    return True


def mac(csp, var, value, assignment, removals, constraint_propagation=AC3):
    """Maintain arc consistency."""
    return constraint_propagation(csp, {(X, var) for X in csp.neighbors[var]}, removals)


# The search, proper


def backtracking_search(csp, select_unassigned_variable=dom_wdeg,
                        order_domain_values=unordered_domain_values, inference=no_inference):
    """[Figure 6.5]"""
    
    def backtrack(assignment):
        if len(assignment) == len(csp.variables):
            return assignment
        var = select_unassigned_variable(assignment,csp)
        for value in order_domain_values(var, assignment, csp):
            if 0 == csp.nconflicts(var, value, assignment):
                csp.assign(var, value, assignment)
                removals = csp.suppose(var, value)
                if inference(csp, var, value, assignment, removals):
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                csp.restore(removals)
        csp.unassign(var, assignment)
        return None
    total = 0
    result= backtrack({})
    # print(total)
    assert result is None or csp.goal_test(result)
    return result

# ______________________________________________________________________________
# Min-conflicts Hill Climbing search for CSPs


def min_conflicts(csp, max_steps=100000):
    """Solve a CSP by stochastic Hill Climbing on the number of conflicts."""
    # Generate a complete assignment for all variables (probably with conflicts)
    csp.current = current = {}
    for var,d in csp.variables:
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    # Now repeatedly choose a random conflicted variable and change it
    for i in range(max_steps):
        conflicted = csp.conflicted_vars(current)
        if not conflicted:
            return current
        var = dom_wdeg(current,csp)
        if var is None: 
            var = random.choice(conflicted)
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    return None


def min_conflicts_value(csp, var, current):
    """Return the value that will give var the least number of conflicts.
    If there is a tie, choose at random."""
    return argmin_random_tie(csp.domains[var], key=lambda val: csp.nconflicts(var, val, current))


# ______________________________________________________________________________


#_____________________RLFAP____________________

class rlfap(CSP):

    def __init__(self, variables, domains, neighbors, constraints):
        for var,dom in variables:
            random.shuffle(domains[var])
        CSP.__init__(self,variables, domains, neighbors, constraints)
#______________________________________________
#other functions
end = 2
txt=1
txt_name = None
def get_input():
    print("\nGive Instance name and algorithm")
    print("\n for MAC type: mac    for FC type: fc     for MINCONFLICTS type: min_conflicts\n ")
    raw = input()
    inp = raw.split()
    if len(inp) > 1:
        # instance,algorithm = raw
        instance = inp[0]
        algorithm = inp[1]
        return instance,algorithm
    elif len(inp) == 1:
        global txt 
        txt = 1
        global txt_name 
        txt_name = inp
    else:
        if raw == 'x':
            global end 
            end = 0

def create_link(link,instance):
    def_link = link +'/'
    return def_link


#_________________________________________________________________________________________

#get comand line arguments and create link instances dictionary
link = None
txt = 0
length = len(sys.argv)
if length > 1:    
    link = sys.argv[1]
else:
    print("Provide file url")




while end >0:
    inp = get_input()
    """instance=None#--------->uncoment for script, fix the tabs
    algo=None#--------->uncoment for script
    if txt==1:#--------->uncoment for script
        with open(txt_name[0], 'r') as f:#--------->uncoment for script
            lines=[line.rstrip('\n') for line in f] #--------->uncoment for script
            for line in lines:#--------->uncoment for script
                print(line)#--------->uncoment for script
                split_data = line.split() #--------->uncoment for script
                if len(split_data) >1:#--------->uncoment for script
                    instance = split_data[0]#--------->uncoment for script, 
                    algo = split_data[1]#--------->uncoment for script"""
                    
    #get instance and algorithm from input
    instance, algo = inp #---------->"""COMMENT for script"""
    link = create_link(link,instance)
    structures(link,instance)
    other_structs()
    results = []
    # for i in range(60):
    #     if str(i) in ctr_dict:
    #         print(i)
    #         print(ctr_dict[str(i)])

    if algo == 'min_conflicts':
        for a in range(5):
            try:
                with timeout(60*10, exception=RuntimeError):
                    checks = 0
                    start = time.time()
                    inst = rlfap(var_list,dom_dict,neighbors,constraints=constraints)
                    #val,check = AC3(inst)
                    result = min_conflicts(inst, max_steps=1000)
                    end = time.time()
                    timer = end-start
                    timer = round(timer,2)
                    #print(tabulate([[timer, checks, inst.nassigns]], headers=['Time', 'Checks', 'Assigments'], tablefmt='orgtbl'))
                    #append results on the final results list
                    res = []
                    res.append(timer)
                    res.append(checks)
                    res.append(inst.nassigns)
                    if result is None:
                        res.append('UNSAT')
                    else:
                        res.append('SAT')
                    results.append(res)
            except RuntimeError:
                print("Iteration", a,  "aborted, too much time taken (max time = 10 min)")
                pass
            #print("Iteration", a, "Successful")

    elif algo == 'fc':
        for a in range(5):
            try:
                with timeout(60*10, exception=RuntimeError):
                    checks = 0
                    start = time.time()
                    inst = rlfap(var_list,dom_dict,neighbors,constraints=constraints)
                    #val,check = AC3(inst)
                    result = backtracking_search(inst, select_unassigned_variable=dom_wdeg,order_domain_values=unordered_domain_values, inference=forward_checking)
                    end = time.time()
                    timer = end-start
                    timer = round(timer,2)
                    #print(tabulate([[timer, checks, inst.nassigns]], headers=['Time', 'Checks', 'Assigments'], tablefmt='orgtbl'))
                    #append results on the final results list
                    res = []
                    res.append(timer)
                    res.append(checks)
                    res.append(inst.nassigns)
                    if result is None:
                        res.append('UNSAT')
                    else:
                        res.append('SAT')
                    results.append(res)
            except RuntimeError:
                print("Iteration", a," aborted, too much time taken (max time = 10 min)")
                pass
            #print("Iteration", a, "Successful")
    elif algo == 'mac':
        
        for a in range(5):
            try:
                with timeout(60*10, exception=RuntimeError):
                    checks = 0
                    inst = rlfap(var_list,dom_dict,neighbors,constraints=constraints)
                    val,check = AC3(inst)
                    start = time.time()
                    result = backtracking_search(inst, select_unassigned_variable=dom_wdeg,order_domain_values=unordered_domain_values, inference=mac)
                    end = time.time()
                    timer = end-start
                    timer = round(timer,2)
                    #print(tabulate([[timer, checks, inst.nassigns]], headers=['Time', 'Checks', 'Assigments'], tablefmt='orgtbl'))
                    #append results on the final results list
                    res = []
                    res.append(timer)
                    res.append(checks)
                    res.append(inst.nassigns)
                    if result is None:
                        res.append('UNSAT')
                    else:
                        res.append('SAT')
                    results.append(res)
            except RuntimeError:
                print("Iteration" ,a ," aborted, too much time taken (max time = 10 min)")
                pass
            #print("Iteration" ,a, "Successful")


    #_________PRINT RESULTS_______________       
    if results is not None:
        av_t = 0
        av_ch =0
        av_a = 0
        #calculate average
        iteration = len(results)
        if iteration == 5:
            av_t = (results[0][0]+results[1][0]+results[2][0]+results[3][0]+results[4][0])/iteration
            av_ch = (results[0][1]+results[1][1]+results[2][1]+results[3][1]+results[4][1])/iteration
            av_a = (results[0][2]+results[1][2]+results[2][2]+results[3][2]+results[4][2])/iteration
        elif iteration == 4:
            av_t = (results[0][0]+results[1][0]+results[2][0]+results[3][0])/iteration
            av_ch = (results[0][1]+results[1][1]+results[2][1]+results[3][1])/iteration
            av_a = (results[0][2]+results[1][2]+results[2][2]+results[3][2])/iteration
        elif iteration == 3:
            av_t = (results[0][0]+results[1][0]+results[2][0])/iteration
            av_ch = (results[0][1]+results[1][1]+results[2][1])/iteration
            av_a = (results[0][2]+results[1][2]+results[2][2])/iteration
        elif iteration == 2:
            av_t = (results[0][0]+results[1][0])/iteration
            av_ch = (results[0][1]+results[1][1])/iteration
            av_a = (results[0][2]+results[1][2])/iteration
        elif iteration == 1:
            av_t = (results[0][0])/iteration
            av_ch = (results[0][1])/iteration
            av_a = (results[0][2])/iteration
        teams_list = ['Time', 'Checks', 'Assigments', 'Result']
        data = results
        row_format ="{:>15}" * (len(teams_list) + 1)
        print("\u0332".join(row_format.format("", *teams_list)))
        for row in  data:
            print("\u0332".join(row_format.format("", *row)))
        
        print(colored('Average Time:', 'green'), colored(av_t, 'green'))
        print(colored('Average Checks:', 'green'), colored(av_ch,'green'))
        print(colored('Average Assigments:','green'), colored(av_a,'green'))
        end +=1
    #_______________________________________________________________________
    # print(neighbors['0'])
    # print(neighbors['1'])
    # print(neighbors['3'])
    # print(neighbors['61'])
    # print(neighbors['146'])
