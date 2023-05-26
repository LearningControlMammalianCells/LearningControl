/* 
                        GalUp.cpp
 Code by Gianfranco Fiore, gianfrancofiore@inwind.it

 This script moves UP galactose syringe and, moves DOWN glucose syringe
 Motor1 -> TL2 -> Galactose 
 Motor2 -> TL1 -> Glucose 
 
 IMPORTANT : if you want to call the compiled exe with Matlab don't use i/o 
 operations.
*/

#include <windows.h>
#include <math.h>
#include <cstdlib>
#include <iostream>
#include "st.h"


using namespace std;

int main(int argc, const char *argv[])
{ // variables used to analyze motors status
int M1Active, M2Active, M1Steps, M2Steps, Inputs;
int StepNumber = atoi(argv[1]);
int StepInterval = atoi(argv[2]);
int status;
/*  ----------------------------------------------------------------
    			Loading Library
    ----------------------------------------------------------------
    IMPORTANT : stp.dll must be in the same path of the exe file
*/

 HINSTANCE HStpDll;
 HStpDll = LoadLibrary("stp.dll"); 
 if(HStpDll!=NULL)
	{
		//cout << "stp.dll loaded\n";
	
	    Type_InitStp InitStp = (Type_InitStp)GetProcAddress( HStpDll, "InitStp");
        Type_RunMotor1 RunMotor1 = (Type_RunMotor1)GetProcAddress( HStpDll, "RunMotor1");
        Type_RunMotor2 RunMotor2 = (Type_RunMotor2)GetProcAddress( HStpDll, "RunMotor2");
        Type_StopMotor1 StopMotor1 = (Type_StopMotor1)GetProcAddress( HStpDll, "StopMotor1");
        Type_StopMotor2 StopMotor2 = (Type_StopMotor2)GetProcAddress( HStpDll, "StopMotor2");
        Type_SetStepMode SetStepMode = (Type_SetStepMode)GetProcAddress( HStpDll, "SetStepMode");
        Type_GetCurrentStatus GetCurrentStatus = (Type_GetCurrentStatus)GetProcAddress(HStpDll,"GetCurrentStatus");

        status = InitStp (); // library init

	  if (status==1){
	    RunMotor2(StepNumber,StepInterval,0,0); 
        RunMotor1(StepNumber,StepInterval,0,0); 
        }
     else {
          cout<<"Error";
          }
    }
	else
	{
		//cout << "Could not load stp.dll\n";
	}
 
    //system("PAUSE");
    return 0;
}
