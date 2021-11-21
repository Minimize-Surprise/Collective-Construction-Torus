//
//  STD.h
//  minimizeSurpriseSelfAssembly
//
//  Created by Tanja Kaiser on 14.03.18.
//  Copyright © 2018 Tanja Kaiser. All rights reserved.
//

#ifndef STD_h
#define STD_h

// experimental setup
#define MUTATION 0.1 // 0.1 - mutation rate
#define POP_SIZE 50 // population

#define MAX_TIME 1000 //500  // time per run
#define MAX_GENS 100  // maximum generations

#define REPETITIONS 10 // repetitions of each individual
#define NUM_AGENTS 50
#define NUM_CHIPS 50

// movement
#define STRAIGHT 0
#define TURN 1
#define UP 1
#define DOWN -1

// sensors
#define S0 0
#define S1 1
#define S2 2
#define S3 3
#define S4 4
#define S5 5
#define S6 6
#define S7 7
#define S8 8
#define S9 9
#define S10 10
#define S11 11

#define PI 3.14159265

// fitness evaulation
#define MIN 0
#define MAX 1
#define AVG 2
#define FIT_EVAL MIN

// define fitness function
#define PRED 0   // prediction

// define sensor model
#define STDS 0 // standard: 6 sensors - in heading direction forward / right / left (1 & 2 blocks ahead)

// define manipulation models
#define NONE 0 // none
#define PRE 1 // prediction

// agent types
#define NOTYPE -1
#define LINE 1
#define BLOCK 7
#define EMPTY 8

#endif /* STD_h */
