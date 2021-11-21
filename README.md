#  Minimize Surprise: Self-Assembly Scenario (2D torus grid world)

## Running one Evolutionary Run in the Simulation

compile with gcc first 

```
.\main EVOL [Grid length in x-direction] [Grid length in y-direction] PRED [Manipulation: None / MAN / PRE ] [Manipulation Parameter]
```

## Manipulation

* None: Minimize Surprise with complete freedom
* PRE: predefined predictions

## Manipulation Parameter

* if Manipulation = None: specify random parameter (will be ignored)
* LINE: aiming for lines of blocks (PRE)
* BLOCK: aiming for block clusters
* EMPTY: aiming for no blocks in agent sensor view 

## Sensor Models

The sensor model is set by default to STDL. 
Changing the sensor model is possible by including the respective header file in main.c. 

* STD6: 2x 6 sensors in heading direction, one set detecting agents, one set detecting blocks 

            . . .
            . . .
              X

