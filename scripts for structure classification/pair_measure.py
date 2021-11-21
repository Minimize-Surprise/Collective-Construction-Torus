import matplotlib.pyplot as plt
import matplotlib 
plt.rcParams.update({'figure.max_open_warning': 0})
import os 
from collections import OrderedDict
import numpy as np
import sys 
import math 
import line_measure as lines 
import cluster_measure as cluster 


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

def measure_pairs(file):

	noLine = lines.measure_line(file)
	noCluster = cluster.measure_cluster(file)

	f = open(file, 'r')
	pos = []
	global grid
	global agents 

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
			# array with all agents w/o headings 
			pos.append([int(p[0]), int(p[1])])

	global pairs
	global none 

	pairs = []
	none = [] 

	for i in range(0, len(pos)):
		pair_horizontal = True
		pair_vertical = True 

		count = 0
		down = False

		## check horizontal 
		if [adaptXPos(pos[i][0]+1), pos[i][1]] in pos:
			count += 1
			if [adaptXPos(pos[i][0]+2), pos[i][1]] in pos: # line indicator 
				pair_horizontal = False

		if pair_horizontal and [adaptXPos(pos[i][0]-1), pos[i][1]] in pos: 
			count += 1
			down = True 
			if [adaptXPos(pos[i][0]-2), pos[i][1]] in pos: # line indicator 
				pair_horizontal = False 

		if pair_horizontal and count == 1: # only 1 neighbor in horizontal direction 

			if down: 
				if not [adaptXPos(pos[i][0]-1), pos[i][1]] in noLine: # check that not part of line 
					pair_horizontal = False 

				elif [adaptXPos(pos[i][0]-1), adaptYPos(pos[i][1]-1)] in pos and [pos[i][0], adaptYPos(pos[i][1]-1)] in pos: 
					pair_horizontal = False 

				elif [adaptXPos(pos[i][0]-1), adaptYPos(pos[i][1]+1)] in pos and [pos[i][0], adaptYPos(pos[i][1]+1)] in pos: 
					pair_horizontal = False 

			else: 
				if not [adaptXPos(pos[i][0]+1), pos[i][1]] in noLine:
					pair_horizontal = False 

				if [adaptXPos(pos[i][0]+1), adaptYPos(pos[i][1]-1)] in pos and [pos[i][0], adaptYPos(pos[i][1]-1)] in pos: 
					pair_horizontal = False 

				if [adaptXPos(pos[i][0]+1), adaptYPos(pos[i][1]+1)] in pos and [pos[i][0], adaptYPos(pos[i][1]+1)] in pos: 
					pair_horizontal = False 
		else: # count > 1 
			pair_horizontal = False 

		## check vertical  
		count = 0 

		if [pos[i][0], adaptYPos(pos[i][1]+1)] in pos:
			count += 1
			if [pos[i][0], adaptYPos(pos[i][1]+2)] in pos: # line indicator 
				pair_vertical = False

		if pair_vertical and [pos[i][0], adaptYPos(pos[i][1]-1)] in pos: 
			count += 1
			down = True 
			if [pos[i][0], adaptYPos(pos[i][1]-2)] in pos: # line indicator 
				pair_vertical = False 

		if pair_vertical and count == 1: # only 1 neighbor in vertical direction 

			if down:  
				if not [pos[i][0], adaptYPos(pos[i][1]-1)] in noLine:
					pair_vertical = False 

				if [adaptXPos(pos[i][0]-1), adaptYPos(pos[i][1]-1)] in pos and [adaptXPos(pos[i][0]-1), pos[i][1]] in pos: 
					pair_vertical = False 

				if [adaptXPos(pos[i][0]+1), adaptYPos(pos[i][1]-1)] in pos and [adaptXPos(pos[i][0]+1), pos[i][1]] in pos: 
					pair_vertical = False 

			else: 
				if not [pos[i][0], adaptYPos(pos[i][1]+1)] in noLine:
					pair_vertical = False 

				if [adaptXPos(pos[i][0]-1), adaptYPos(pos[i][1]+1)] in pos and [adaptXPos(pos[i][0]-1), pos[i][1]] in pos: 
					pair_vertical = False 

				if [adaptXPos(pos[i][0]+1), adaptYPos(pos[i][1]+1)] in pos and [adaptXPos(pos[i][0]+1), pos[i][1]] in pos: 
					pair_vertical = False 

		else: 
			pair_vertical = False 


		if pos[i] in noLine and pos[i] in noCluster and (pair_vertical or pair_horizontal): 
			pairs.append(pos[i])
		else: 
			none.append(pos[i])

	return none 



if __name__ == "__main__":

	if len(sys.argv) >= 2:
		file = sys.argv[1]
	else:
		file = 'chips_trajectory'

	measure_pairs(file)

	# calculate & print diff 
	f = open('eval_pairs_blocks.txt', 'w')
	f.write('blocks not within pair structure: %d\n' % len(none))
	f.write('percentage of blocks not within structure: %f\n' % (float(len(none))/float(agents)))
	f.write('blocks within pair structure: %d\n' % (len(pairs)))
	f.write('percentage of blocks within structure: %f\n \n' % (len(pairs)/float(agents)))


