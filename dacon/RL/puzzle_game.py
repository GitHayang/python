#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np


# In[6]:


class Game:
    def __init__(self):
        self.board=np.array([[7,1,4,15],
                             [0,9,12,13],
                             [5,3,14,10],
                             [6,8,11,2]])
        self.empty_field=(1,0)
        self.number_of_moves = 100
        self.number_of_moves_played = 0
    
    def check_right_position(self, field_value):
        niz=self.board.reshape(-1)
        if(niz[field_value-1] == field_value):
            return True
        
        return False    

    #levo, gore, desno, dole
    def play_turn(self, action):
        x, y = self.empty_field
        if (action == [1, 0, 0, 0]) and y != 0:
            y -= 1
        if (action == [0, 1, 0, 0]) and x != 0:
            x -= 1    
        if (action == [0, 0, 1, 0]) and y != 3:
            y += 1
        if (action == [0, 0, 0, 1]) and x != 3:
            x += 1
        
        self.swap_field_values(self.empty_field, (x, y))
        self.empty_field = (x,y)
        
    def swap_field_values(self, field1, field2):
        self.board[field1[0], field1[1]], self.board[field2[0], field2[1]] = self.board[field2[0], field2[1]], self.board[field1[0], field1[1]]
    
    
    
a = Game()
print(a.board)
a.play_turn([0, 1, 0, 0])
a.play_turn([0, 0, 1, 0])
a.play_turn([0, 1, 0, 0])
print(a.board)


# In[ ]:




