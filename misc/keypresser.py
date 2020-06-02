# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:23:27 2020

@author: Mikko Impiö
"""

import keyboard
import random 
import time
import string
import pyautogui

time.sleep(5)
while True:
    key = random.choice(string.ascii_letters)
    keyboard.write(key)
    d = random.randint(1,20)
    time.sleep((20/d) + 5 )
    
    if random.randint(1,100) > 95:
        keyboard.write('\n')
        
    if random.randint(1,100) > 50:
        pyautogui.click(500,500)
        
#    if random.randint(1,100) > 50:
#        pyautogui.scroll(-10000)

