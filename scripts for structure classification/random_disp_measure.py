import matplotlib.pyplot as plt
import matplotlib 
plt.rcParams.update({'figure.max_open_warning': 0})
import os 
from collections import OrderedDict
import numpy as np
import sys 

def adaptYPos(y):
	if y >= grid[1]:
		return y - grid[1]
	elif y < 0:
		return y + grid[1]
	else:
		return y 

def adaptXPos(x):
	if x >= grid[0]:
		return x - grid[0]
	elif x < 0:
		return x + grid[0]
	else:
		return x 

# main part 

if len(sys.argv) >= 2:
	file = sys.argv[1]
else:
	file = 'chips_trajectory'

f = open(file, 'r')
pos = []

# read in file with agent positions 
for line in f: 	
	# store generation number 
	if line[:4] == "Gen:":
		gen = int(line[5:].rstrip())

	elif line[:5] == "Grid:":
		row = line[6:].split(', ')
		grid = [int(row[0]), int(row[1].rstrip())]

	elif line[:6] == "Chips:":
		agents = int(line[7:].rstrip())

	elif line[:4] != "1000": # do nothing as long as time step not last time step 
		continue 

	elif line.split():   
		row = line.split(': ') # first split
		p = row[1].split(', ')
		pos.append([int(p[0]), int(p[1])])

# check for line 
disp = [ ] 
noDisp = [ ] 

for i in range(0, len(pos)):
	count = 0
	
	# 8 agents in Moore neighborhood 
	if [adaptXPos(pos[i][0]+1), pos[i][1]] in pos:
		count = 8

	if [adaptXPos(pos[i][0]-1), pos[i][1]] in pos: 
		count = 8

	if [pos[i][0], adaptYPos(pos[i][1]+1)] in pos: 
		count = 8

	if [pos[i][0], adaptYPos(pos[i][1]-1)] in pos:
		count = 8

	if [adaptXPos(pos[i][0]+1), adaptYPos(pos[i][1]+1)] in pos: 
		count += 1
	
	if [adaptXPos(pos[i][0]+1), adaptYPos(pos[i][1]-1)] in pos:
		count += 1 
	
	if [adaptXPos(pos[i][0]-1), adaptYPos(pos[i][1]+1)] in pos:
		count += 1
	
	if [adaptXPos(pos[i][0]-1), adaptYPos(pos[i][1]-1)] in pos: 
		count += 1

	if count <= 1: 
		disp.append(pos[i])
	else: 
		noDisp.append(pos[i])

# calculate & print diff 

f = open('eval_random_disp_blocks.txt', 'w')
f.write('blocks not randomly dispersed: %d\n' % len(noDisp))
f.write('percentage of blocks not within structure: %f\n' % (float(len(noDisp))/float(agents)))
f.write('blocks randomly dispersed: %d\n' % len(disp))
f.write('percentage of blocks within structure: %f\n \n' % (float(len(disp))/float(agents)))
