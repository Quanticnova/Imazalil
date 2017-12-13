#!/usr/bin/env python3
# general testing script for class related stuff
import uuid

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

baz = derivedFoo(9,8)
baz.Move()
