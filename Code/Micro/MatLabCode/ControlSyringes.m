
flag = 0;
state = 0; %no tetra [down up]



while 0==0


while flag==0
    pause(1)
    
    flag = importdata('flag.txt');
end
action = importdata('action.txt');

% action = 1 dai tetraciclina
flag = 0;
writematrix(flag, 'flag.txt')

if action==1 & state==0
    %DOWN
    disp('stepperDOWN - alpha_mem+Tetra') % cells OFF
    stepperDOWN();
    state=1; %[up down]
    

elseif action==0 & state==1
    %UP
    disp('stepperUP - alpha_mem')
    stepperUP();
    
    state=0; %[down up]
end

end

