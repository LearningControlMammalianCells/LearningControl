                               %stepperDOWN.m
%                      Smith predictor/FeedBack Control/Relay
%=========================================================================
%|                   Code written by Gianfranco Fiore                     |
%|                       gianfrancofiore@inwind.it                        | 
%=========================================================================      
%|                     Stand Alone Control Algorithm                      |
%|                            code version 0.0.3                          |
%|                               24/07/2011                               |
%=========================================================================

function [e_time,state] = stepperDOWN()
tic
dos('GalDown 800 2');
pause(6);
%dos('StopMotor');
e_time = toc;
state = 0;
end