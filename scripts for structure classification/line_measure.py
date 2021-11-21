import matplotlib.pyplot as plt
import matplotlib 
plt.rcParams.update({'figure.max_open_warning': 0})
import os 
from collections import OrderedDict
import numpy as np
import sys 
import math 
import line_measure as lines 
import itertools


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

def measure_line(file):

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

	global lines
	global none 

	lines = []
	none = [] 

	for i in range(0, len(pos)):
		line_horizontal = False
		line_vertical = False 

		## check horizontal 
		indices = [] 
		indices.append(i)

		check = True
		count = 1
		j = 1
		while check: # check in direction right 
			if [adaptXPos(pos[i][0]+j), pos[i][1]] in pos: 
				count += 1
				indices.append(pos.index([adaptXPos(pos[i][0]+j), pos[i][1]]))
				j += 1
				if len(indices) == grid[0]: 
					check = False 
			else: 
				check = False 

		check = True 
		j = 1 
		while check: # check in direction left 
			if [adaptXPos(pos[i][0]-j), pos[i][1]] in pos: 
				count += 1
				indices.append(pos.index([adaptXPos(pos[i][0]-j), pos[i][1]]))
				j += 1
				if len(indices) == grid[0]: 
					check = False 
			else: 
				check = False 

		if count > 2 : # check line neighbors  

			# max neighbors per side is half the line length 
			max_neighbors = math.ceil(len(indices)/2.0)

			j = 0
			check = True 
			neighbor_count = 0 

			for el in indices:
				if [pos[el][0], adaptYPos(pos[el][1]+1)] in pos: 
					
					neighbor_count += 1

					if [adaptXPos(pos[el][0]+1), adaptYPos(pos[el][1]+1)] in pos: 

						if [adaptXPos(pos[el][0]+1), pos[el][1]] in pos: 
							idx = pos.index([adaptXPos(pos[el][0]+1), pos[el][1]])
							if idx in indices: 
								neighbor_count = max_neighbors+1
								break 

			if neighbor_count <= max_neighbors:
				neighbor_count = 0 

				for el in indices:
					if [pos[el][0], adaptYPos(pos[el][1]-1)] in pos: 
						neighbor_count += 1

						if [adaptXPos(pos[el][0]+1), adaptYPos(pos[el][1]-1)] in pos: 
							if [adaptXPos(pos[el][0]+1), pos[el][1]] in pos: 
								idx = pos.index([adaptXPos(pos[el][0]+1), pos[el][1]])

								if idx in indices:
									neighbor_count = max_neighbors+1
									break 

				if neighbor_count <= max_neighbors:
					line_horizontal = True 

		## check vertical
		indices = [ ] 
		indices.append(i)

		check = True
		count = 1
		j = 1

		while check: # check in direction right 
			if [pos[i][0], adaptYPos(pos[i][1]+j)] in pos: 
				count += 1
				indices.append(pos.index([pos[i][0], adaptYPos(pos[i][1]+j)]))
				j += 1
				if len(indices) == grid[1]: 
					check = False 
			else: 
				check = False 

		check = True 
		j = 1 
		while check: # check in direction left 
			if [pos[i][0], adaptYPos(pos[i][1]-j)] in pos: 
				count += 1
				indices.append(pos.index([pos[i][0], adaptYPos(pos[i][1]-j)]))
				j += 1
				if len(indices) == grid[1]: 
					check = False 
			else: 
				check = False 
		
		if count > 2 : # check line neighbors  

			# max neighbors per side is half the line length 
			max_neighbors = math.ceil(len(indices)/2.0)

			j = 0
			check = True 
			neighbor_count = 0 

			for el in indices:
				if [adaptXPos(pos[el][0]+1), pos[el][1]] in pos: 
					neighbor_count += 1

					if [adaptXPos(pos[el][0]+1), adaptYPos(pos[el][1]+1)] in pos:
						if [pos[el][0], adaptYPos(pos[el][1]+1)] in pos: 
							idx = pos.index([pos[el][0], adaptYPos(pos[el][1]+1)])

							if idx in indices:
								neighbor_count = max_neighbors+1
								break 

			if neighbor_count <= max_neighbors:
				neighbor_count = 0 

				for el in indices:
					if [adaptXPos(pos[el][0]-1), pos[el][1]] in pos: 
						neighbor_count += 1

						if [adaptXPos(pos[el][0]-1), adaptYPos(pos[el][1]+1)] in pos: 
							if [pos[el][0], adaptYPos(pos[el][1]+1)] in pos: 
								idx = pos.index([pos[el][0], adaptYPos(pos[el][1]+1)])

								if idx in indices:
									neighbor_count = max_neighbors+1
									break 

				if neighbor_count <= max_neighbors:
					line_vertical = True 


		if line_horizontal or line_vertical: 
			lines.append(pos[i])
		else: 
			none.append(pos[i])

	return none 



if __name__ == "__main__":

	if len(sys.argv) >= 2:
		file = sys.argv[1]
	else:
		file = 'chips_trajectory'

	measure_line(file)

	# calculate & print diff 
	f = open('eval_line_blocks.txt', 'w')
	f.write('blocks not within line structure: %d\n' % len(none))
	f.write('percentage of blocks not within structure: %f\n' % (float(len(none))/float(agents)))
	f.write('blocks within line structure: %d\n' % (len(lines)))
	f.write('percentage of blocks within structure: %f\n \n' % (len(lines)/float(agents)))


