# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 18:26:43 2017

@author: Arpit
"""

import math
def cities():
    distances = [[0 for i in range(8)] for j in range(8)];
    x_y = [[2,3],[4,5],[7,3],[8,2],[2,7],[2,2],[3,7],[12,4]]
    for i in range(len(x_y)):
        for j in range(len(x_y)):
            x = x_y[j][0] - x_y[i][0]
            y = x_y[j][1]- x_y[i][1]
            distances[i][j] = math.sqrt(x*x + y*y)
    return distances

def cost(path):
    l = len(path)
    print(path)
    cost = 0
    distances = cities()
    
    for i in range(l-1):
        cost = cost + distances[path[i]-1][path[i+1]-1]
    cost = cost + distances[path[l-1]-1][path[0]-1]
    print(cost)
    return cost

def hill_climb(path):
    path_cost = cost(path)
    temp_cost = path_cost
    pathf = list(path)
    done = 0
    
    while done == 0:
        done = 1
        
        for i in range(len(path)-1):
            for j in range(i+1, len(path)):
                neighbor = list(path)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                cost_neighbor = cost(neighbor)
                if cost_neighbor < temp_cost:
                    done = 0
                    temp_cost = cost_neighbor
                    pathf = list(neighbor)
                    break
        
        
        path_cost = temp_cost
        path = list(pathf)
        print("Shortest path is:")
        print(pathf)
        print("And cost for this is:")
        print(temp_cost)


def main():
    path = [1,2,3,4,5,6,7,8]
    hill_climb(path)
    
if __name__ == "__main__":
    main()