#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// header file for changing betw. different scenarios (MANL, ...)
#include "STDSC.h"

int FIT_FUN; // fitness function
int MANIPULATION; // options: 'NONE', 'PRE' (Predefined), 'MAN' (Manipulation)
int EVOL; // evolution vs replay of genome

// 2D grid size
int SIZE_X;
int SIZE_Y;

// positions
struct pos {
    int x;
    int y;
};

// agent data
struct agent {
    int type; // type of sensor prediction (None, line, ...)
    struct pos coord;  // position coordinates
    struct pos heading; // heading vector
};

struct agent *p; //current position
struct agent *p_next; //next position

// for wood chipping (only grid cell position necessary!)
struct pos *chips;

// ANN genomes
// current weights
float weight_actionNet[POP_SIZE][LAYERS][CONNECTIONS];
float weight_predictionNet[POP_SIZE][LAYERS][CONNECTIONS];
// hidden states prediction network
float hiddenBefore_predictionNet[NUM_AGENTS][HIDDENP];
// mutated weights
float newWeight_actionNet[POP_SIZE][LAYERS][CONNECTIONS];
float newWeight_predictionNet[POP_SIZE][LAYERS][CONNECTIONS];

// average predictions during run
float pred_return[SENSORS];
float pred_return_last[SENSORS]; // for fitness = LAST only

// predictions of agents
int predictions[NUM_AGENTS][SENSORS];

// action values
int current_action[NUM_AGENTS][MAX_TIME];

// numbering of evolutionary run
int COUNT = 0;


/* Function: activation
 * activation function (tanh)
 *
 * x: input value
 * returns: activation
 *
 */
float activation(float x) {
    return 2.0/(1.0+exp(-2.0*x))-1.0;
}

/* Function: propagate_actionNet
 * propagation of action network
 *
 * weight_action: weights of individual
 * input: input array [sensors, last action value]
 * returns: action output [action value, turn direction]
 *
 */
int *propagate_actionNet(float weight_action[LAYERS][CONNECTIONS], int input[INPUTA]) {
    float hidden_net[HIDDENA];
    float i_layer[INPUTA]; //inputs: x sensor values + 1 action value
    float net[OUTPUTA]; // output: action value + turn direction
    int i, j;
    int *action_output = malloc(OUTPUTA * sizeof(int));
    
    // calculate activation of input neurons
    // input value + threshold value / bias (= extra input of 1 and weight)
    for(i=0;i<INPUTA;i++)
        i_layer[i] = activation((float)input[i]*weight_action[0][2*i]
                                -weight_action[0][2*i+1]);
    
    // hidden layer - 4 hidden neurons
    for(i=0;i<HIDDENA;i++) {
        hidden_net[i] = 0.0;
        
        //calculate input of hidden layer
        for(j=0;j<INPUTA;j++)
            hidden_net[i] += i_layer[j]*weight_action[1][INPUTA*i+j];
    }
    // calculate input of output layer
    for(i=0; i<OUTPUTA; i++){ // outputs
        net[i] = 0.0;
        for(j=0;j<HIDDENA;j++) // hidden layers
            net[i] += activation(hidden_net[j])*weight_action[2][HIDDENA*i+j];
    }
    
    // map outputs to binary values
    // first output: action value
    if(activation(net[0]) < 0.0)
        action_output[0] = STRAIGHT;
    else
        action_output[0] = TURN;
    
    // second output: turn direction
    if(activation(net[1]) < 0.0)
        action_output[1] = UP;
    else
        action_output[1] = DOWN;
    
    return action_output;
}

/* Function: prediction_output
 *
 * value: output of prediction network
 * returns: 0 or 1
 *
 */
int prediction_output(float value){
    if(activation(value) < 0.0)
        return 0;
    else
        return 1;
}

/*
 * Function: propagate_predictionNet
 * propagate prediction network
 *
 * weight_prediction: weights of individual
 * a: index of agent
 * input: input array
 *
 */
void propagate_predictionNet(float weight_prediction[LAYERS][CONNECTIONS], int a, int input[INPUTP]) {
    float hidden_net[HIDDENP];
    float i_layer[INPUTP];
    float net[OUTPUTP];
    int i, j;
    
    for(i=0;i<INPUTP;i++) // outputs of input layer
        i_layer[i] = activation((float)input[i]*weight_prediction[0][2*i]
                                - weight_prediction[0][2*i+1]);
    
    for(i=0;i<HIDDENP;i++) { // hidden layer neurons
        hidden_net[i] = 0.0;
        for(j=0;j<INPUTP;j++) // inputs
            hidden_net[i] += i_layer[j]*weight_prediction[1][(INPUTP+1)*i+j]; // inputs: InputP + 1 recurrent
        hidden_net[i] += hiddenBefore_predictionNet[a][i]*weight_prediction[1][(INPUTP+1)*i+INPUTP];
        hiddenBefore_predictionNet[a][i] = activation(hidden_net[i]);
    }
    
    for(i=0;i<OUTPUTP;i++) { //outputs (== # of sensors)
        net[i] = 0.0;
        for(j=0;j<HIDDENP;j++) { // hidden layers
            net[i] += activation(hidden_net[j])*weight_prediction[2][i*HIDDENP+j];
        }
    }
    
    if(MANIPULATION == NONE){
        // maps prediction values to binary sensor values
        for(i=0;i<OUTPUTP;i++)  // output values
            predictions[a][i] = prediction_output(net[i]);
    }
}

/* Function: selectAndMutate(int maxID)
 * selects and mutates genomes for next generation
 *
 * maxID: population with maximum fitness
 * fitness: fitness of whole population (i.e. number of evaluated genomes)
 *
 */
void selectAndMutate(int maxID, float fitness[POP_SIZE]) {
    float sum1 = 0.0;
    float sum2 = 0.0;
    float pr[POP_SIZE], r;
    int i, j, k, ind;
    
    // total fitness - used for mutation
    for(ind=0;ind<POP_SIZE;ind++)
        sum1 += fitness[ind];
    
    // relative fitness over individuals
    for(ind=0;ind<POP_SIZE;ind++) {
        sum2 += fitness[ind];
        pr[ind] = sum2/sum1;
    }
    
    for(ind=0;ind<POP_SIZE;ind++) {
        if(ind==maxID) { //population with maximum fitness - elitism of 1
            // keep weights as they are
            for(j=0;j<LAYERS;j++)
                for(k=0;k<CONNECTIONS;k++) {
                    newWeight_predictionNet[ind][j][k] = weight_predictionNet[maxID][j][k];
                    newWeight_actionNet[ind][j][k] = weight_actionNet[maxID][j][k];
                }
        } else { // mutate weights of networks
            r = (float)rand()/(float)RAND_MAX;
            i = 0;
            while( (r > pr[i]) && (i<POP_SIZE-1) )
                i++;
            for(j=0;j<LAYERS;j++)
                for(k=0;k<CONNECTIONS;k++) {
                    newWeight_predictionNet[ind][j][k] = weight_predictionNet[i][j][k];
                    newWeight_actionNet[ind][j][k] = weight_actionNet[i][j][k];
                }
            for(j=0;j<LAYERS;j++)
                for(k=0;k<CONNECTIONS;k++) {
                    // 0.1 == mutation operator
                    if((float)rand()/(float)RAND_MAX < MUTATION) // prediction network
                        newWeight_predictionNet[ind][j][k]
                        += 0.8 * (float)rand()/(float)RAND_MAX - 0.4;  // 0.8 * [0,1] - 0.4 --> [-0.4, 0.4]
                    if((float)rand()/(float)RAND_MAX < MUTATION) // action network
                        newWeight_actionNet[ind][j][k]
                        += 0.8 * (float)rand()/(float)RAND_MAX - 0.4;
                }
        }
    }
    
    // update weights
    for(ind=0;ind<POP_SIZE;ind++) {
        for(j=0;j<LAYERS;j++)
            for(k=0;k<CONNECTIONS;k++) {
                weight_predictionNet[ind][j][k] = newWeight_predictionNet[ind][j][k];
                weight_actionNet[ind][j][k] = newWeight_actionNet[ind][j][k];
            }
    }
}

/* Function: adjustXPosition
 * adjusts position for movement on torus
 *
 * x: x position
 * return: updated x value
 *
 */
int adjustXPosition(int x){
    if(x < 0)
        x += SIZE_X;
    else if(x > SIZE_X-1)
        x -= SIZE_X;
    
    return x;
}

/* Function: adjustYPosition
 * adjusts position for movement on torus
 *
 * y: y position
 * return: updated y value
 *
 */
int adjustYPosition(int y){
    if(y < 0)
        y += SIZE_Y;
    else if(y > SIZE_Y-1)
        y -= SIZE_Y;
    
    return y;
}


/* Function: sensorModelSTD
 * sensor model with 6 binary values in heading direction
 *
 * i: index of current agent
 * grid: array with agents positions (0 - no agent, 1 - agent)
 * returns: sensor values
 *
 */
int * sensorModelSTD(int i, int grid[SIZE_X][SIZE_Y]){
    int j;
    int dy, dx, dxl, dyl, dxplus, dxmin, dyplus, dymin;
    int *sensors = malloc((SENSORS/2) * sizeof(int)); // 6 Sensors
    
    // initialise values / reset sensor values
    for(j=0; j<(SENSORS/2); j++)
        sensors[j] = 0;
    
    // determine coordinates of cells looked at
    // includes adjustment to torus grid
    
    // short range forward
    dx = adjustXPosition(p[i].coord.x + p[i].heading.x);
    dy = adjustYPosition(p[i].coord.y + p[i].heading.y);
    
    // long range forward
    dxl = adjustXPosition(p[i].coord.x + 2*p[i].heading.x);
    dyl = adjustYPosition(p[i].coord.y + 2*p[i].heading.y);
    
    // points for left and right sensors
    dyplus = adjustYPosition(p[i].coord.y + 1); // y coordinate + 1
    dymin = adjustYPosition(p[i].coord.y - 1); // y coordinate - 1
    
    dxplus = adjustXPosition(p[i].coord.x + 1); // x coordinate + 1
    dxmin = adjustXPosition(p[i].coord.x - 1); // x coordinate - 1
    
    // forward looking sensor / in direction of heading
    sensors[S0] = grid[dx][dy]; // FORWARD SHORT
    sensors[S3] = grid[dxl][dyl]; // FORWARD LONG
    
    // headings in x-direction (i.e., y equals 0)
    if(p[i].heading.y == 0){
        if(p[i].heading.x == 1){
            sensors[S2] = grid[dx][dyplus]; // y+1; x + 1 * heading
            sensors[S5] = grid[dxl][dyplus]; // y+1; x + 2 * heading
            sensors[S1] = grid[dx][dymin]; // y-1; x + 1 * heading
            sensors[S4] = grid[dxl][dymin]; // y-1; x + 2 * heading
        }
        else{
            sensors[S1] = grid[dx][dyplus]; // y+1; x + 1 * heading
            sensors[S4] = grid[dxl][dyplus]; // y+1; x + 2 * heading
            sensors[S2] = grid[dx][dymin]; // y-1; x + 1 * heading
            sensors[S5] = grid[dxl][dymin]; // y-1; x + 2 * heading
        }
    }
    
    // headings in y-direction (i.e., x equals 0)
    else if(p[i].heading.x == 0){
        if(p[i].heading.y == 1){
            sensors[S1] = grid[dxplus][dy]; // y + 1 * heading; x + 1
            sensors[S4] = grid[dxplus][dyl]; // y + 2 * heading; x + 1
            sensors[S2] = grid[dxmin][dy]; // y + 1 * heading; x - 1
            sensors[S5] = grid[dxmin][dyl]; // y + 2 * heading; x - 1
        }
        else{
            sensors[S2] = grid[dxplus][dy]; // y + 1 * heading; x + 1
            sensors[S5] = grid[dxplus][dyl]; // y + 2 * heading; x + 1
            sensors[S1] = grid[dxmin][dy]; // y + 1 * heading; x - 1
            sensors[S4] = grid[dxmin][dyl]; // y + 2 * heading; x - 1
        }
    }
    
    return sensors;
}


/*
 * Function: doRun
 * contains loop for evaluation of one individual (genome)
 *
 * gen: generation of evolutionary process
 * ind: index of individual
 * p_initial: array with initial agent positions
 * chips_initial: array with initial wood chip positions
 * maxTime: run time in time steps
 * log: logging of agent trajectory
 * noagents: number of agents which are currently used (use for self-repair/replay scenario only! otherwise to be set to NUM_AGENTS)
 * returns: average fitness of agents
 *
 */
float doRun(int gen, int ind, struct agent p_initial[NUM_AGENTS], struct pos chips_initial[NUM_AGENTS], int maxTime, int log, int noagents){
    FILE *f;
    int i, j, k; // for loops
    int timeStep = 0; //current time step
    int fit = 0;
    float fit_return = 0.0;
    struct agent *temp; // to switch p and p_next
    int occupied = 0;
    int predReturn[SENSORS] = { 0 }; // agent prediction counter
    int predReturnLast[SENSORS] = { 0 }; // last sensor predictions
    double angle;
    struct pos tmp_agent_next; // temporary next agent position
    struct pos tmp_chip_next; // temporary next chip position
    int inputA[INPUTA]; // input array action network
    int inputP[INPUTP]; // input array prediction network
    int *sensors = NULL;
    int *action_output = NULL;
    int grid_next[SIZE_X][SIZE_Y];
    int grid_agents[SIZE_X][SIZE_Y];
    int grid_chips[SIZE_X][SIZE_Y];
    // file name agent trajectory
    char str[12];
    char trajectory_file[100];
    char chips_trajectory_file[100];
    int no_agents = noagents; // for replay runs or rather self-repair
    
    sprintf(str, "%d_%d_%d", COUNT, 0, no_agents);
    strcpy(trajectory_file, "agent_trajectory");
    strcat(trajectory_file, str);
    
    sprintf(str, "%d_%d_%d", COUNT, 0, NUM_CHIPS);
    strcpy(chips_trajectory_file, "chips_trajectory");
    strcat(chips_trajectory_file, str);
    
    if(log==1){
        f = fopen(trajectory_file, "a");
        fprintf(f, "Gen: %d\n", gen);
        fprintf(f, "Grid: %d, %d\n", SIZE_X, SIZE_Y);
        fprintf(f, "Agents: %d\n", no_agents);
        fclose(f);
        
        f = fopen(chips_trajectory_file, "a");
        fprintf(f, "Gen: %d\n", gen);
        fprintf(f, "Grid: %d, %d\n", SIZE_X, SIZE_Y);
        fprintf(f, "Chips: %d\n", NUM_CHIPS);
        fclose(f);
    }
    
    // initialise hidden neurons and predictions, prediction counter to zero
    memset(pred_return, 0, sizeof(pred_return));
    memset(pred_return_last, 0, sizeof(pred_return_last));
    memset(predictions, 0, sizeof(predictions));
    memset(hiddenBefore_predictionNet, 0, sizeof(hiddenBefore_predictionNet));
    
    //initialise agents
    for(i=0; i<no_agents; i++){
        // set initial agent positions
        p[i].coord.x = p_initial[i].coord.x;
        p[i].coord.y = p_initial[i].coord.y;
        p[i].heading.x = p_initial[i].heading.x;
        p[i].heading.y = p_initial[i].heading.y;
        
        // next position
        p_next[i].coord.x = p_initial[i].coord.x;
        p_next[i].coord.y = p_initial[i].coord.y;
        p_next[i].heading.x = p_initial[i].heading.x;
        p_next[i].heading.y = p_initial[i].heading.y;
        
        // set manipulation initialisation based on option
        if(MANIPULATION != NONE){
            if(p[i].type == LINE){
                predictions[i][S0 + (SENSORS/2)] = 1;
                predictions[i][S3 + (SENSORS/2)] = 1;
            }
            else if(p[i].type == BLOCK){ // all one
                for(j=SENSORS/2; j<SENSORS; j++)
                    predictions[i][j]=1;
            }
            // else if EMPTY = all zeros (as initialised)
        }
    }
    
    //initialise chips
     for(i=0; i<NUM_CHIPS; i++){
        chips[i].x = chips_initial[i].x;
        chips[i].y = chips_initial[i].y;
    }
    
    while(timeStep < maxTime){
        
        // determine occupied grid cells
        // set all cells to zero
        memset(grid_agents, 0, sizeof(grid_agents));
        memset(grid_next, 0, sizeof(grid_next));
        memset(grid_chips, 0, sizeof(grid_chips));
        
        // agent occupied grid cells
        for(i=0; i<no_agents; i++)
            grid_agents[p[i].coord.x][p[i].coord.y] = 1;
        
        // chips occupied grid cells
        for(i=0; i<NUM_CHIPS; i++)
            grid_chips[chips[i].x][chips[i].y] = 1;
        
        // print all chips positions
        if(log == 1){
            f = fopen(chips_trajectory_file, "a");
            for(k=0; k<NUM_CHIPS; k++)
                fprintf(f, "%d: %d, %d\n", timeStep, chips[k].x, chips[k].y);
            fclose(f);
        }
        
        // iterate through all agents
        for(i=0;i<no_agents;i++){
            
            // store agent trajectory
            if(log == 1){
                f = fopen(trajectory_file, "a");
                // print position and heading
                fprintf(f, "%d: %d, %d, %d, %d\n", timeStep, p[i].coord.x, p[i].coord.y, p[i].heading.x, p[i].heading.y);
                fclose(f);
            }
            
            /*
             * Determine current sensor values (S of t)
             */
            free(sensors);
            sensors = NULL;
            
            // agent sensors
            sensors = sensorModelSTD(i, grid_agents);
            
            for(j=0;j<(SENSORS/2);j++){
                
                // set sensor values as first SENSORS/2 ANN input values
                inputA[j] = sensors[j];
                inputP[j] = sensors[j];
                
                // count correct predictions
                if(sensors[j] == predictions[i][j]){
                    fit++;
                }
                
                // set prediction counters
                predReturn[j] += predictions[i][j];
                
            } // for loop sensor values
            
            // wood chip sensors
            free(sensors);
            sensors = NULL;
            
            sensors = sensorModelSTD(i, grid_chips);
            
            for(j=0;j<(SENSORS/2);j++){
            
                // set sensor values as ANN input values
                 inputA[(SENSORS/2)+j] = sensors[j];
                 inputP[(SENSORS/2)+j] = sensors[j];
                
                // count correct predictions
                  if(sensors[j] == predictions[i][(SENSORS/2)+j]){
                    fit++;
                 }
            
            // set prediction counters
                predReturn[(SENSORS/2)+j] += predictions[i][(SENSORS/2)+j];
             } // for loop sensor values wood chips
            
            /*
             * propagate action network with current sensor values
             * and last done action to determine next action
             * output: [0, 1]
             */
            
            if(timeStep <= 0) // initialize for first time step
                inputA[SENSORS] = STRAIGHT; // 0 - SENSORS-1 = sensor values; SENSORS = action value
            else
                inputA[SENSORS] = current_action[i][timeStep-1]; // action of time step before
            
            free(action_output);
            action_output = NULL;
            
            // propagate action network
            action_output = propagate_actionNet(weight_actionNet[ind], inputA);
            
            // store actions over course of time per agent
            current_action[i][timeStep] = action_output[0];
            
            /*
             * propagate prediction network with current sensor
             * values and next action (current action returned
             * by action ANN) to determine next sensor values
             * output: prediction of sensor values (per agent)
             */
            inputP[SENSORS] = current_action[i][timeStep];
            
            // prediction network only necessary when predictions aren't completely predefined
            if(MANIPULATION != PRE){
                propagate_predictionNet(weight_predictionNet[ind], i, inputP);
            }
            
            /* Update position / heading according to action value returned by Action ANN */
            p_next[i] = p[i];  // copy values
            
            // action value = action output[0]
            // 0 == move straight; 1 == turn
            if(current_action[i][timeStep] == STRAIGHT){ // move 1 grid cell straight - update position
                occupied = 0;
                
                // no wood chip, no agent in front
                if(inputP[S0] == 0 && inputP[S0+(SENSORS/2)] == 0){
                    // move in heading direction (i.e. straight)
                    tmp_agent_next.x = adjustXPosition(p[i].coord.x + p[i].heading.x);
                    tmp_agent_next.y = adjustYPosition(p[i].coord.y + p[i].heading.y);
                    
                    // check if agent or wood chip already moved to that position
                    if(grid_next[tmp_agent_next.x][tmp_agent_next.y] == 1){
                        occupied = 1;
                    }
                    
                    if(!occupied){ // cell not occupied - agent can move
                        p_next[i].coord.x = tmp_agent_next.x;
                        p_next[i].coord.y = tmp_agent_next.y;
                        grid_next[tmp_agent_next.x][tmp_agent_next.y] = 1;
                    }
                }
                // no agent in front, wood chip in front, cell two steps ahead not blocked
                else if(inputP[S0] == 0 && inputP[S0+(SENSORS/2)] == 1 && inputP[S3] == 0 &&  inputP[S3+(SENSORS/2)] == 0){
                    
                    // next position wood chip
                    tmp_chip_next.x = adjustXPosition(p[i].coord.x + 2*p[i].heading.x);
                    tmp_chip_next.y = adjustYPosition(p[i].coord.y + 2*p[i].heading.y);
                    
                    // targeted grid cell already occupied by a moving item (wood chip or agent)
                    if(grid_next[tmp_chip_next.x][tmp_chip_next.y] == 1){
                        occupied = 1;
                    }
                    
                    if(!occupied){ // cell not occupied - agent can move
                        // move wood chip
                        
                        tmp_agent_next.x = adjustXPosition(p[i].coord.x + p[i].heading.x);
                        tmp_agent_next.y = adjustYPosition(p[i].coord.y + p[i].heading.y);
                        
                        for(k=0; k<NUM_CHIPS; k++){
                            // find wood chip in heading direction of agent + move it
                            // if it was already moved, no chip will be moved + agent stick on his cell (as when wood chip got moved away a different agent must be on its place)
                            if(chips[k].x == tmp_agent_next.x && chips[k].y == tmp_agent_next.y){ // chip in front of agent = next position of agent
                                // next chip position
                                chips[k].x = tmp_chip_next.x;
                                chips[k].y = tmp_chip_next.y;
                                grid_next[tmp_chip_next.x][tmp_chip_next.y] = 1;
                                
                                // next agent position
                                p_next[i].coord.x = tmp_agent_next.x;
                                p_next[i].coord.y = tmp_agent_next.y;
                                grid_next[p_next[i].coord.x][p_next[i].coord.y] = 1;
            
                                break;
                            }
                        }
                    }
                }
            }
            else if (current_action[i][timeStep] == TURN) {  // turn - update heading
                // turn direction = action output 1
                // [-1, 1] --> defines turn direction
                
                angle = atan2(p[i].heading.y, p[i].heading.x); // calculate current orientation
                p_next[i].heading.x = cos(angle + action_output[1]*(PI/2));
                p_next[i].heading.y = sin(angle + action_output[1]*(PI/2));
            }
        } // agents

        // increase time step
        timeStep++;
        
        // update position & heading
        temp = p;
        p = p_next;
        p_next = temp;
        
    } // while time
    
    // prediction counter
    for(i=0; i<SENSORS; i++){
        pred_return[i] = (float)predReturn[i]/(float)(maxTime*no_agents);
        pred_return_last[i] = (float)predReturnLast[i]/(float)no_agents;
    }
    
    if(log==1){ // print last time step
        // print agents
        f = fopen(trajectory_file, "a");
        for(i=0; i<no_agents; i++)
            fprintf(f, "%d: %d, %d, %d, %d\n", maxTime, p[i].coord.x, p[i].coord.y, p[i].heading.x, p[i].heading.y);
        fclose(f);
        
        // print chips
        f = fopen(chips_trajectory_file, "a");
        // print position and heading
        for(k=0; k<NUM_CHIPS; k++)
            fprintf(f, "%d: %d, %d\n", maxTime, chips[k].x, chips[k].y);
        
        fclose(f);
    }
    
    fit_return = (float)fit/(float)(no_agents * maxTime * SENSORS);
    
    // return average fitness
    return fit_return;
}

/* Function: evolution
 *
 * no_agents: evolving agents
 *
 */
void evolution(int no_agents){
    // variables
    time_t *init_rand;
    int i, b, j, ind, k, gen, rep;
    int maxID = -1;
    float max, avg;
    struct agent p_initial[REPETITIONS][no_agents];
    struct pos chips_initial[REPETITIONS][NUM_CHIPS];
    float tmp_fitness;
    int max_rep = 0;
    int store = 0; //bool to store current agent and block data
    float agentPrediction[SENSORS];
    float agentPredictionLast[SENSORS];
    float fitness[POP_SIZE]; // store fitness of whole population for roulette wheel selection
    float pred[SENSORS];
    float predLast[SENSORS]; // for fitness = LAST only - predictions of maxTime-1 (aka last timestep where fitness can be calculated)
    int action_values[NUM_AGENTS][MAX_TIME]; // action values of maximum run
    int grid[SIZE_X][SIZE_Y];
    FILE *f;
    
    // store agent movement
    struct agent agent_maxfit[no_agents];
    struct agent tmp_agent_maxfit_final[no_agents];
    struct agent agent_maxfit_beginning[no_agents];
    int tmp_action[no_agents][MAX_TIME]; // tmp storage action values
    
    // store chips movement
    struct pos chips_maxfit[NUM_CHIPS];
    struct pos tmp_chips_maxfit_final[NUM_CHIPS];
    struct pos chips_maxfit_beginning[NUM_CHIPS];
    
    // file names
    char str[12];
    char predGen_file[100];
    char fit_file[100];
    char actVal_file[100];
    char agent_file[100];
    char actGen_file[100];
    char chips_file[100];
    
    // file names
    sprintf(str, "%d_%d_%d", COUNT, 0, no_agents);
    strcpy(fit_file, "fitness");
    strcat(fit_file, str);
    
    strcpy(predGen_file, "prediction_genomes");
    strcat(predGen_file, str);
    
    strcpy(actGen_file, "action_genomes");
    strcat(actGen_file, str);
    
    strcpy(actVal_file, "actionValues");
    strcat(actVal_file, str);
    
    strcpy(agent_file, "agents");
    strcat(agent_file, str);
    
    sprintf(str, "%d_%d_%d", COUNT, 0, NUM_CHIPS);
    strcpy(chips_file, "chips");
    strcat(chips_file, str);
    
    // initialise random number generator
    init_rand = malloc(sizeof(time_t));
    srand((unsigned int)time(init_rand));
    free(init_rand);
    
    // initialise weights of neural nets
    for(ind=0; ind<POP_SIZE; ind++){
        for(j=0; j<LAYERS; j++){
            for(k=0; k<CONNECTIONS; k++){
                weight_actionNet[ind][j][k] = 1.0 * (float)rand()/(float)RAND_MAX - 0.5;
                weight_predictionNet[ind][j][k] = 1.0 * (float)rand()/(float)RAND_MAX - 0.5;
            }
        }
    }
    
    // evolutionary runs
    for(gen=0;gen<MAX_GENS;gen++){
        // population
        max = 0.0; //fitness
        avg = 0.0;
        maxID = -1;
        
        // initialisation - per repetition unique block and agent positions
        for(k=0; k<REPETITIONS; k++){
            
            memset(grid, 0, sizeof(grid));
            
            // SEED BLOCKS
            /*
            chips_initial[k][0].x = 10;
            chips_initial[k][0].y = 10;
             grid[chips_initial[k][0].x][chips_initial[k][0].y] = 1;

            chips_initial[k][1].x = 11;
            chips_initial[k][1].y = 10;
             grid[chips_initial[k][1].x][chips_initial[k][1].y] = 1;

            chips_initial[k][2].x = 10;
            chips_initial[k][2].y = 11;
             grid[chips_initial[k][2].x][chips_initial[k][2].y] = 1;
            
            chips_initial[k][3].x = 11;
            chips_initial[k][3].y = 11;
             grid[chips_initial[k][3].x][chips_initial[k][3].y] = 1;

             */
            
            // generate wood chip positions
            // i = 4 for SEED 
            for(i=0;i<NUM_CHIPS;i++){
                b = 1;
                
                while(b){
                    b = 0;
                    
                    chips_initial[k][i].x = rand()%(SIZE_X);
                    chips_initial[k][i].y = rand()%(SIZE_Y);
                    
                    // no agent or chip already on position
                    if(grid[chips_initial[k][i].x][chips_initial[k][i].y] == 1)
                        b = 1;
                    else{
                        grid[chips_initial[k][i].x][chips_initial[k][i].y] = 1; // general grid
                    }
                }
            }
            
            // generate agent positions
            for(i=0;i<no_agents;i++){
                // initialise agent positions to random discrete x & y values
                // min = 0 (rand()%(max + 1 - min) + min)
                // no plus 1 as starting from 0 and size = SIZE_X/Y
                b = 1;
                
                while(b){
                    b = 0;
                    
                    p_initial[k][i].coord.x = rand()%(SIZE_X);
                    p_initial[k][i].coord.y = rand()%(SIZE_Y);
                    
                    // set agent heading values randomly (north, south, west, east possible)
                    int directions[2] = {1, -1};
                    int randInd = rand() % 2;
                    
                    if((double)rand()/(double)RAND_MAX < 0.5){
                        p_initial[k][i].heading.x = directions[randInd];
                        p_initial[k][i].heading.y = 0;
                    }
                    else {
                        p_initial[k][i].heading.x = 0;
                        p_initial[k][i].heading.y = directions[randInd];
                    }
                    
                    if(grid[p_initial[k][i].coord.x][p_initial[k][i].coord.y] == 1)
                        b = 1;
                    else{
                        grid[p_initial[k][i].coord.x][p_initial[k][i].coord.y] = 1;
                    }
                }
                

            }
            

        }
        
        for(ind=0;ind<POP_SIZE;ind++){
            // fitness evaluation - initialisation based on case
            if(FIT_EVAL == MIN){ // MIN - initialise to value higher than max
                fitness[ind] = (float)(SENSORS+1);
                tmp_fitness = (float)(SENSORS+1);
            }
            else { // MAX, AVG - initialise to zero
                fitness[ind] = 0.0;
                tmp_fitness = 0.0;
            }
            
            // reset prediction storage
            memset(pred, 0, sizeof(pred));
            memset(predLast, 0, sizeof(predLast));
            
            // repetitions
            for(rep=0;rep<REPETITIONS;rep++){
                store = 0;
                
                tmp_fitness = doRun(gen, ind, p_initial[rep], chips_initial[rep], MAX_TIME, 0, no_agents);
                
                if(FIT_EVAL == MIN){ //min
                    if(tmp_fitness < fitness[ind]){
                        fitness[ind] = tmp_fitness;
                        
                        for(int s = 0; s<SENSORS; s++){ // average sensor predictions
                            pred[s] = pred_return[s];
                        }
                        store = 1;
                    }
                }
                else if(FIT_EVAL == MAX){ //max
                    if(tmp_fitness > fitness[ind]){
                        fitness[ind] = tmp_fitness;
                        for(int s = 0; s<SENSORS; s++){
                            pred[s] = pred_return[s];
                        }
                        
                        store = 1;
                    }
                }
                else if(FIT_EVAL == AVG){ // avg
                    fitness[ind] += (float)(tmp_fitness/REPETITIONS); // avg
                    for(int s = 0; s<SENSORS; s++){
                        pred[s] += (float)pred_return[s]/(float)REPETITIONS;
                    }
                    
                    if(rep == REPETITIONS-1) // store data of last repetition
                        store = 1;
                }
                
                // store best fitness + id of repetition
                if(store){
                    max_rep = rep;
                    
                    for(i=0; i<no_agents; i++){ // store agent final positions
                        tmp_agent_maxfit_final[i].coord.x = p[i].coord.x;
                        tmp_agent_maxfit_final[i].coord.y = p[i].coord.y;
                        tmp_agent_maxfit_final[i].heading.x = p[i].heading.x;
                        tmp_agent_maxfit_final[i].heading.y = p[i].heading.y;
                        tmp_agent_maxfit_final[i].type = p[i].type;
                    }
                    
                    for(i=0; i<no_agents; i++){  // store action values of best try of repetition
                        for(j=0; j<MAX_TIME; j++){
                            tmp_action[i][j] = current_action[i][j];
                        }
                    }
                    
                    // store chips positions
                    for(i=0; i<NUM_CHIPS; i++){
                        tmp_chips_maxfit_final[i].x = chips[i].x;
                        tmp_chips_maxfit_final[i].y = chips[i].y;
                    }
                }
            } // repetitions
            
            // average values
            avg += fitness[ind];
            
            // store maximum / chosen fitness of the generation
            if(fitness[ind]>max){
                max = fitness[ind];
                maxID = ind;
                
                // store agent predictions
                for(int s = 0; s<SENSORS; s++){
                    agentPrediction[s] = pred[s];
                    agentPredictionLast[s] = predLast[s];
                }
                
                // store initial and final agent positions
                for(i=0; i<no_agents; i++){
                    agent_maxfit[i].coord.x = tmp_agent_maxfit_final[i].coord.x;
                    agent_maxfit[i].coord.y = tmp_agent_maxfit_final[i].coord.y;
                    agent_maxfit[i].heading.x = tmp_agent_maxfit_final[i].heading.x;
                    agent_maxfit[i].heading.y = tmp_agent_maxfit_final[i].heading.y;
                    agent_maxfit[i].type = tmp_agent_maxfit_final[i].type;
                    
                    agent_maxfit_beginning[i].coord.x = p_initial[max_rep][i].coord.x;
                    agent_maxfit_beginning[i].coord.y = p_initial[max_rep][i].coord.y;
                    agent_maxfit_beginning[i].heading.x = p_initial[max_rep][i].heading.x;
                    agent_maxfit_beginning[i].heading.y = p_initial[max_rep][i].heading.y;
                }
                
                // store action values of best run in generation
                for(j=0; j<no_agents; j++){
                    for(i=0; i<MAX_TIME; i++){
                        action_values[j][i] = tmp_action[j][i];
                    }
                }
                
                // store chips positions
                for(i=0; i<NUM_CHIPS; i++){
                    chips_maxfit[i].x = tmp_chips_maxfit_final[i].x;
                    chips_maxfit[i].y = tmp_chips_maxfit_final[i].y;
                    
                    chips_maxfit_beginning[i].x = chips_initial[max_rep][i].x;
                    chips_maxfit_beginning[i].y = chips_initial[max_rep][i].y;
                }
                
            }
        } // populations
        
        // print results: Generation, max fitness, max movement, id
        printf("#%d %f (%d)\n", gen, max, maxID);
        
        // write to files
        f = fopen(fit_file, "a");
        // write size of grid, generation, maximum fitness, average fitness,
        // maximum movement, average movement, sensor value predictions
        fprintf(f, "%d %d %d %e %e (%d) ",
                SIZE_X, SIZE_Y, gen, max, avg/(float)POP_SIZE, maxID);
        
        for(i=0; i<SENSORS; i++)
            fprintf(f, "%f ", agentPrediction[i]);
        
        fprintf(f, "\n");
        fclose(f);
        
        f = fopen(agent_file, "a");
        fprintf(f, "Gen: %d\n", gen);
        fprintf(f, "Grid: %d, %d\n", SIZE_X, SIZE_Y);
        fprintf(f, "Fitness: %f \n", max);
        
        for(i=0; i<no_agents; i++){
            fprintf(f, "%d, %d, %d, %d, %d, %d, %d, %d, %d\n", agent_maxfit[i].coord.x, agent_maxfit[i].coord.y,
                    agent_maxfit_beginning[i].coord.x, agent_maxfit_beginning[i].coord.y, agent_maxfit[i].heading.x,
                    agent_maxfit[i].heading.y, agent_maxfit_beginning[i].heading.x,
                    agent_maxfit_beginning[i].heading.y, agent_maxfit[i].type);
        }
        
        fprintf(f, "\n");
        fclose(f);
        
        f = fopen(chips_file, "a");
        fprintf(f, "Gen: %d\n", gen);
        fprintf(f, "Grid: %d, %d\n", SIZE_X, SIZE_Y);
        fprintf(f, "Fitness: %f \n", max);
        
        for(i=0; i<NUM_CHIPS; i++){
            fprintf(f, "%d, %d, %d, %d\n", chips_maxfit_beginning[i].x, chips_maxfit_beginning[i].y, chips_maxfit[i].x, chips_maxfit[i].y);
        }
        
        fprintf(f, "\n");
        fclose(f);
        
        f = fopen(actVal_file, "a");
        fprintf(f, "Gen: %d\n", gen);
        fprintf(f, "Grid: %d, %d\n", SIZE_X, SIZE_Y);
        fprintf(f, "Fitness: %f \n", max);
        
        for(i=0; i<no_agents; i++){
            fprintf(f, "Agent: %d\n", i);
            fprintf(f, "[");
            for(j=0; j<MAX_TIME; j++){
                fprintf(f, "%d, ", action_values[i][j]);
            }
            fprintf(f, "]\n");
        }
        
        fprintf(f, "\n");
        fclose(f);
        
        // store genomes
        f = fopen(actGen_file, "a");
        for(i=0; i<LAYERS; i++){
            for(j=0; j<CONNECTIONS; j++){
                fprintf(f, "%f ", weight_actionNet[maxID][i][j]);
            }
            fprintf(f, "\n");
        }
        fprintf(f, "\n");
        fclose(f);
        
        f = fopen(predGen_file, "a");
        for(i=0; i<LAYERS; i++){
            for(j=0; j<CONNECTIONS; j++){
                fprintf(f, "%f ", weight_predictionNet[maxID][i][j]);
            }
            fprintf(f, "\n");
        }
        fprintf(f, "\n");
        fclose(f);
        
        // selection and mutation for evolution
        selectAndMutate(maxID, fitness);
        
    } // generations
    
    // re-run last / best controller of evolutionary run
    doRun(gen, maxID, agent_maxfit_beginning, chips_maxfit_beginning, MAX_TIME, 1, no_agents);
    
    COUNT++;
}

/* main function
 * starts depending on number of arguments a re-run of existing genomes
 * or a new evolutionary run
 *
 */
int main(int argc, char** argv){
    
    int i = 0;
    int type1 = NOTYPE;
    
    // position of agents, current action value
    p = (struct agent*)malloc(NUM_AGENTS*sizeof(struct agent));
    p_next = (struct agent*)malloc(NUM_AGENTS*sizeof(struct agent));
    
    // position of wood
    chips = (struct pos*)malloc(NUM_CHIPS*sizeof(struct pos));
    
    // Replay
    if(!strcmp(argv[1], "EVOL")){
        printf("EVOLUTION.\n");
        EVOL = 1;
        if(argc != 7){
            // initialisation of variables via arguments
            printf("Please specify 6 input values: EVOL [Grid size x direction] [Grid size y direction] PRED [Manipulation] [Agent Type (Manipulation)]\n");
            exit(0);
        }
    }
    else{ // no valid option
        printf("Please specify EVOL as the first option.\n");
        exit(0);
    }
    
    if(EVOL){
        // Grid size
        SIZE_X = atof(argv[2]);
        SIZE_Y = atof(argv[3]);
        
        // set fitness function
        printf("Fitness function: prediction\n");
        FIT_FUN = PRED;

        // set manipulation value
        if(!strcmp(argv[5], "NONE")){
            printf("Manipulation: None\n");
            MANIPULATION = NONE;
        }
        else if(!strcmp(argv[5], "PRE")){
            printf("Manipulation: Predefined\n");
            MANIPULATION = PRE;
        }
        else{
            printf("No valid manipulation option specified.\n");
            exit(0);
        }
        
        if(MANIPULATION != NONE){
            
            // type1 - 1st evolutionary run
            if(!strcmp(argv[6], "LINE")){
                type1 = LINE;
            }
            else if(!strcmp(argv[6], "BLOCK")){
                type1 = BLOCK;
            }
            else if(!strcmp(argv[6], "EMPTY")){
                type1 = EMPTY;
            }
            else{
                printf("Unknown Agent Type.\n");
                exit(0);
            }
        }
        
        // set agents to chosen type - set to NOTYPE if no manipulation
        for(i=0; i<NUM_AGENTS; i++){
            p[i].type = type1;
            p_next[i].type = type1;
        }
        
        printf("Sensor type: %d\n", SENSOR_MODEL);
        printf("Grid size = [%d, %d] \n", SIZE_X, SIZE_Y);
        
        // do evolutionary run
        evolution(NUM_AGENTS); // agents using best genome, number of genome set / evolutionary run
        
    }
    
     free(p);
     free(p_next);
     free(chips);
}


