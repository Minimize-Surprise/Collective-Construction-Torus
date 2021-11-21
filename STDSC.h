//
//  STDSC.h
//
//  Created by Tanja Kaiser on 14.03.18.
//

#include "STD.h"

#ifndef STDSC_h
#define STDSC_h

// ANN parameter
#define LAYERS 3  // ANN layers

#define INPUTA 13  // input action network (12 sensors + 1 action value)
#define HIDDENA  7 // hidden action network
#define OUTPUTA  2 // output action network

#define INPUTP  13  // input prediction network (12 sensors + 1 action value)
#define HIDDENP  12 // hidden prediction network
#define OUTPUTP  12 // output prediction network (12 sensor predictions)

#define CONNECTIONS  168 // maximum connections

#define SENSORS 12 // use two times STDS

#define SENSOR_MODEL STDS

#endif /* STD_h */
