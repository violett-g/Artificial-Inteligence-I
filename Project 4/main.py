import os
import operator


"""
mia domi me instances kai tis domes tous gia to kathe ena
gia kathe instance dimiourgoume ena csp provlima meso tou init tis class csp
opou pername san orismata tis domes kai kratame ton ctr_dict san eksoteriko global 
kai trexoume tous algorithmous
    """

ctr_dict = {} #item=tuple(x,y,operator,k) #EKSOTERIKA GLOBAL
dom_list =[] #item = tuple(dom_name, num_of_values, list_of_values)
var_list = [] #item = tuple(var_name, dom_name)

# Walking a directory tree and printing the names of the directories and files
#os.walk() returns directory path , directory name, containing file list
#-----------------------------------> prepei na diavazoume san isodos to link tou fakelou
"""instances = {} #item = instance_name:[ctr_dict,dom_dict,var_list,neighbors]
    
    """
instances = {}

for dir_path,dir_name,files in os.walk('///home/violett_gk/Documents/AI/Project_3/rlfap/test'):
    #print(f'Found on  directory: {files}')
    for file in files:
        file_name = dir_path +'/'+file
        with open(file_name, 'r') as f:

            # #instance name
            # def split_word(word):
	        #     return[char for char in word]
            # char_list=split_word(file_name)
            # size = len(char_list)
            # instance = 0
            # for item in char_list[3:len(char_list)-4]:
	        #     instance = instance+item
            # if instances not in instances:



            #fill instance data structures
            if file_name.find("ctr") >0 :
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
                        var_dom=tuple(split_data)
                        var_list.append(var_dom)
            #add instance to dictionary

#print(var_list)
# var='0'
# print(ctr_dict.get(var))
#print(dom_list)
assigments = {} #dictionary for assigments, item = var:val

#----->auto prepei na gnei gia kathe instance metatropei to dom_list se dom dict
dom_dict = {} #dictionary for domain of values, item =  val:domain
for var in var_list:
    for dom in dom_list:
        if var[1] == dom[0]:
            dict1={var[0]:dom[2]}
            dom_dict.update(dict1)
#print(dom_dict)


neighbors = {} #dictionary for neighbor values, item = value:list_of_neighbour_values
#a value y is a neigbor of value x, if y and x has a constreint relation that means for each x(key) on ctr_list create a list of y(values of key)
for var in var_list:
    neighbors_list = []
    x_ctr=ctr_dict.get(var[0])
    for y,op,k,w in x_ctr:
        neighbors_list.append(y)
        neighbors.update({var[0]:neighbors_list})
    # for ctr in ctr_list:#ctr = (x,y,operator,z)
    #     x,y,op,z = ctr
    #     if var[0] == x:
    #         neighbors_list.append(y)
    #         dict1={x:neighbors_list}
    #         neighbors.update(dict1)

"""functions"""
print(neighbors.get('14'))
def constraints(A,a,B,b):  #--------------->global function
    if str(B) in neighbors.get(str(A)):
        # for ctr in ctr_list: #iterate throught constraint list
        #     x,y,op,z = ctr
        #     if x==str(A) and y==str(B): #keep the constraint that includes these variables
        #         if str(a) in dom_dict.get(str(A)) and str(b) in dom_dict.get(str(B)): #check if given values exist on the domains of varables
        #             if op is '=':
        #                 if a + b == int(z): #checki if the constraint is satisfied
        #                     return True
        #             elif op is '>':
        #                 if a + b > int(z): #check if the constraint is satisfied
        #                     return True
        A_ctr = ctr_dict.get(str(A))
        if str(a) in dom_dict.get(str(A)) and str(b) in dom_dict.get(str(B)): #check if given values exist on the domains of varables
            for y,op,k,w in A_ctr:
                if y == str(B): #test constraints that include B
                    if op == '=':
                        if abs(a-b) == int(k): #checki if the constraint is satisfied
                            return True
                    elif op == '>':
                        if abs(a-b) > int(k): #check if the constraint is satisfied
                            return True
    return False

#print(constraints(0,30,3,58))

"""
///general comments to make on the code from aima:   curr_domains is a dictionary var:dom like the on e dom_dict but for the actual values that remains the domain values dom_dict saves all from the files

revision-> remove a valu from dom if it is incosistnet and update the wight
return ???


heuristic->variable ordering, selects first the vriable with the smallest ratio=dom/deg
heuristic(csp)
    #csp.curr_domains
    #var_weight = domain/sum(constraint wieght of var with neighbour->at least one neigbour should have no value assigned)
    ratio = +inf
    dom_size = 0
    min_var = nan
    
    for var in var_list:
        sum=0
        var_neighbors = neighbors[var] #return the list of var neighbours
        unassigned = False
        for neighbor on var_neigbor: #checks if there is at least on e neigbour with unassigned value
            if not assigment.haskey(neighbour)
                unassigned = True

    
        if unassigned == True:
            for neighbour in var_neighbour:
                neigbour_ctr = ctr_dict.get(neighbour)
                for y,op,k,w in neighbor ctr:
                    sum += weight #sum of the constraint weight of all the ctr that include var
        
        dom_size = len(ctr.curr_domains[var]) ->check an epistrefei to pragmatiko size h ton arithmo ton value sto dom
        #need a check if ratio = -inf to be set equal to dom/deg
        #ratio = min(ratio, (dom_size/sum))
        if ratio > dom_size/sum:
            ratio = dom_size/sum
            min_var = var 

    return min_var

    """

def dom_wdeg(csp,assigment):
    ratio = 100000000000000000000000
    wdeg = 1
    dom_size = 0
    min_var = 0

    for var in var_list:
        sum=0
        var_neighbors = neighbors.get(var) #return the list of the neighbours of the current value
        unassigned = False
        for neighbor in var_neighbors: #checks if there is at least one neigbour with unassigned value
            if not assigment.haskey(neighbor):
                unassigned = True
        if unassigned == True: #if there is an unsigned neigbor 
            for neighbour in var_neighbors:
                neighbor_ctr = ctr_dict.get(neighbour)
                for y,op,k,w in neighbor_ctr:
                    sum += w #sum of the constraint weight of all the ctr that include var
                    wdeg=sum
        dom_size = len(csp.curr_domains[var])
        if ratio > dom_size/wdeg:
            ratio = dom_size/wdeg
            min_var = var 
    return min_var

"""
TO DO:
constraint is a dict--------> DONE
for every constraint -------> DONE
    x is a key
    y is a key
neighprs to be changed ---------> DONE
constraint to be changed--------> DONE
heuristic -------->DONE
csp class -------->
"""