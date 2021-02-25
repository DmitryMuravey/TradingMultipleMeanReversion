#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 08:33:38 2020

@author: dm
"""

import collections

Card = collections.namedtuple('Card', ['rank', 'suit'])
ranks = [str(n) for n in range(2, 11)]+ list('JQKA')
class FrenchDeck :
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()
    
    def __init__(self) :
        self._cards = [Card(rank, suit) for suit in self.suits for rank in self.ranks]

    def __len__(self) : return len(self._cards) 

    def __getitem__(self, idx) : return self._cards[idx] 

f_deck = FrenchDeck()    
len(f_deck)
#%%