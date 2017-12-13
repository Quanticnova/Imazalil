#!/usr/bin/env python3
# general testing script for class related stuff
import uuid
import random as rd
import matplotlib.pyplot as plt


Agents = dict()

class Foo:
    """
    hurrdurr docstring
    """

    def __init__(self, food, location):
        self._FoodReserve = food
        self._Loc = location
        self._ID = str(uuid.uuid4())
        Agents[self._ID] = self._Loc

    #def __del__(self):
    #    print("\n:: test destructor - I'm dead now. ")

    def Die(self):
        """
        If an agent dies, it is removed from the list of agents
        """
        del Agents[self._ID]

    def Action(self):
        """
        Every action a starts with a reduced FoodReserve.
        """
        self._FoodReserve -= 1

        if(self._FoodReserve == 0):
            self.Die()

        else:
            print('\n:: doing stuff')

bar = Foo(2, 3)
bar.Action()
print(bar._ID)



class derivedFoo(Foo):
    """
    even more docstring
    """

    def Move(self):
        print(self._Loc)

# baz = derivedFoo(9,8)
# baz.Move()
#
# A = Foo(4,5)
# B = Foo(4,5)
# C = Foo(4,5)
# D = Foo(4,5)
# print(Agents, sep='\n')
# for _ in range(D._FoodReserve):
#     D.Action()
# print(Agents, sep='\n')
N = 1000

l = []
id_dic = dict()
for t in range(N):
    l.append(Foo(2, t))
    id_dic[l[-1]._ID] = t

#print(id_dic, l,  sep='\n')

keys_shuffled = list(id_dic.keys())
rd.shuffle(keys_shuffled)

hurrdurr = []
for key in keys_shuffled:
     hurrdurr.append(id_dic[key])
