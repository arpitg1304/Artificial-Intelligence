# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 19:36:01 2017

@author: Arpit
"""

class Problem():
    def __init__(self, cl,ml,boat,cr,mr):
        self.cl = cl
        self.ml = ml
        self.boat = boat
        self.cr = cr
        self.mr = mr
        self.parent = None
        
    def goal(self):
        if self.cr == 3 and self.mr ==3:
            return True
        else:
            return False
        
    def valid_state(self):
       if self.cl >=0 and self.ml >=0 and self.ml>=0 and self.mr>=0 and (self.cl<=self.ml or self.ml==0) and (self.cr<=self.mr or self.mr==0):
           return True
       else:
           return False
       
    def __eq__(self,other):
        if self.cl == other.cl and self.ml == other.ml and self.cr == other.cr and self.mr == other.mr and self.boat == other.boat:
            return True
        else:
            return False
    
    def __hash__(self):
        return hash((self.cl, self.ml, self.boat, self.cr, self.mr))

def breadth_first_search():
    start_state = Problem(3,3,0,0,0)
    if start_state.goal():
        return start_state
    frontier = list()
    explored = set()
    frontier.append(start_state)
    while frontier:
        state = frontier.pop(0)
        if state.goal():
            print("goal reached")
            return state
        explored.add(state)
        children = get_children(state)
        for child in children:
            if (child not in explored) or (child not in frontier):
                frontier.append(child)
    return None
                
def get_children(state):
    children = []
    if state.boat == 0:
        child = Problem(state.cl-2, state.ml, 1, state.cr+2, state.mr)
        if child.valid_state():
            children.append(child)
            child.parent = state
        child = Problem(state.cl-1, state.ml-1, 1, state.cr+1, state.mr+1)
        if child.valid_state():
            children.append(child)
            child.parent = state
        child = Problem(state.cl-1, state.ml, 1, state.cr+1, state.mr)
        if child.valid_state():
            children.append(child)
            child.parent = state
        child = Problem(state.cl, state.ml-2, 1, state.cr, state.mr+2)
        if child.valid_state():
            children.append(child)
            child.parent = state
        child = Problem(state.cl, state.ml-1, 1, state.cr, state.mr+1)
        if child.valid_state():
            children.append(child)
            child.parent = state
            
    if state.boat == 1:
        child = Problem(state.cl+1, state.ml, 0, state.cr-1, state.mr)
        if child.valid_state():
            children.append(child)
            child.parent = state
        child = Problem(state.cl+1, state.ml+1, 0, state.cr-1, state.mr-1)
        if child.valid_state():
            children.append(child)
            child.parent = state
        child = Problem(state.cl, state.ml+1, 0, state.cr, state.mr-1)
        if child.valid_state():
            children.append(child)
            child.parent = state
        child = Problem(state.cl+2, state.ml, 0, state.cr-2, state.mr)
        if child.valid_state():
            children.append(child)
            child.parent = state
        child = Problem(state.cl, state.ml+2, 0, state.cr, state.mr-2)
        if child.valid_state():
            children.append(child)
            child.parent = state
    return children
    

def main():
    x = breadth_first_search()
    solution = []
    solution.append(x)
    parent = x.parent
    while parent:
        solution.append(parent)
        parent = parent.parent
    print("in " + str(len(solution)) + " steps") 
    for i in range (len(solution)):
        step = solution[len(solution) - i-1]
        print(str(step.cl) + " " + str(step.ml) + " " + str(step.boat) + " " + str(step.cr) + " " + str(step.mr))
        
if __name__ == "__main__":
    main()
    