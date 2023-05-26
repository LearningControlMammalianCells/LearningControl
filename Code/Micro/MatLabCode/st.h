// Header file for use with stp.dll

typedef int		(*Type_InitStp)();
typedef bool	(*Type_RunMotor1)(int steps, int interval, int direction, int outputs);
typedef bool	(*Type_StopMotor1)(int outputs);
typedef bool	(*Type_RunMotor2)(int steps, int interval, int direction, int outputs);
typedef bool	(*Type_StopMotor2)(int outputs);
typedef bool	(*Type_SetStepMode)(int M1Mode, int M2Mode);
typedef bool	(*Type_GetCurrentStatus)(int *M1Active, int *M2Active, int *M1Steps, int *M2Steps,  int *Inputs);


