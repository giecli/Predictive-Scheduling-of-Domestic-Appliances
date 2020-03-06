clc
clear all
close all

%%
%%%%%%%Heat pump parameters%%%%%%%%

%heat capacity of room air (kJ/C)
Cpr = 810;
%heat capacity of floor (kJ/C)
Cpf = 3315;
%heat capacity of water (kJ/C)
Cpw = 836;
%heat transfer coefficient between room air and ambient(kJ/Ch)
UAra = 28;
%heat transfer coefficient between floor and room air(kJ/Ch)
UAfr = 624;
%heat transfer coefficient between water anf floor(kJ/Ch)
UAwf = 28;
%fraction of solar radiations on floor
p = 0.2;
%heat pump efficiency
eta = 3;
%area of glazing (m^2)
Aw = 1.67;
%sample time = 0.25 hours
Ts = 1/4;
%system matrices
a11 = (-UAfr-UAra)/Cpr;
a22 = (-UAwf-UAfr)/Cpf;
a12 = UAfr/Cpr;
a21 = UAfr/Cpf;
a33 = -UAwf/Cpw;
a23 = UAwf/Cpf;
a32 = UAwf/Cpw;
A = [a11 a12 0; a21 a22 a23; 0 a32 a33];
B = [0; 0; eta/Cpw];
C = [1 0 0];
E = [UAra/Cpr Aw*(1-p)/Cpr; 0 Aw*p/Cpf; 0 0];
%discretize the matrices
sysd=c2d(ss(A,[B E],C,zeros(1,3)),Ts,'zoh');
% access data from state space form
[Ad BdEd Cd Dd] = ssdata(sysd);
Bd = BdEd(:,1);
Ed = BdEd(:,2:3);
%number of states
nx = 3;
%number of inputs
nu = 1;
%constraints
ymax = 22;
ymin = 18;
umax = 3600*1;
umin = 0;
dumax = umax/2;
dumin = -umax/2;
%intial temperature
x0_in = [20;23;25];

%%%%%%%%water heater parametrs%%%%%%%%

%water heater efficiency = 1
eta_heater = 1;
%specific heat capacity of the tank (kJ/K)
c_t = 3881.3;
%Thermal codunctance of water heater (kJ/hK)
UA = 8.29;
%density of water (kg/L)
rho_w = 1;
%specific heat of water (kJ/kgK)
c_p = 4.187;
%inlet water temperature (C)
T_in = 20;
%read the water draw off volume file
file = fopen('DHW0001_DHW.txt');
scan_file = textscan(file,'%f');
water = double(scan_file{1});
%draw off energy consumption
Q_c = zeros(1,length(water));
%state space matrices
A = -(UA)/c_t; B = eta_heater/c_t; C = 1; E = [-1/c_t,UA/c_t];

sysd=c2d(ss(A,[B E],C,zeros(1,1)),Ts,'zoh');
[Adw BdwEdw Cdw Ddw]= ssdata(sysd);
Bdw=BdwEdw(:,1);
Edw=BdwEdw(:,2:3);
%Consumption disturbance
for i = 1 : length(Q_c)
    Q_c(i) = (60-T_in)*rho_w*water(i)*c_p;
end
%constraints
ymax_w = 70;
ymin_w = 50;
u_w_max = 2*3600;
u_w_min = 0;
du_w_max = u_w_max/2;
du_w_min = u_w_min/2;
%number of control inputs
nu_w = 1;
%number of states
nx_w = 1;
%initial temperature of the tank
x0_w_in = 20;

%%%%%%% Simulation parameters%%%%%%%%%

%Prediction horizon
N = 96;
%load the prices of electricity.
%they are given every hour
%so convert them to half hour interval
load priceH;
priceHH = zeros(length(priceH)*(1/Ts),1);
for i = 1:length(priceH)
    in = (1/Ts)*(i-1)+1;
    fin = in+1;
    priceHH(in:fin) = priceH(i);
end
%load the forecasted ambient temperature and solar radiations
temp = xlsread('temp_data.xlsx');
for i = 1:length(temp)
    in = (1/Ts)*(i-1)+1;
    fin = in+1;
    tempHH(in:fin) = temp(i);
end
GHI = xlsread('GHI_data.xlsx');
for i = 1:length(GHI)
    in = (1/Ts)*(i-1)+1;
    fin = in+1;
    GHIHH(in:fin) = GHI(i);
end
%number of simulation days
days_sim = 7;
%running time of request 3 and request 4
T_run3 = 2*(1/Ts);
T_run4 = 2.5*(1/Ts);
%generate request 3 and request 4 deadline for number of days
deadline_3 = zeros(1,days_sim);
deadline_4 = zeros(1,days_sim);
request_3 = zeros(1,days_sim);
request_4 = zeros(1,days_sim);
for i = 1 : days_sim
    flag = 1;
    while(flag ==1)
        deadline_3(i) = 24*(1/Ts)*(i-1) + round(24*(1/Ts)*rand());
        deadline_4(i) = 24*(1/Ts)*(i-1) + round(24*(1/Ts)*rand());
        request_3(i) = 24*(1/Ts)*(i-1) + round(24*(1/Ts)*rand());
        request_4(i) = 24*(1/Ts)*(i-1) + round(24*(1/Ts)*rand());
        %time difference between deadline and request time should be
        %greater than the request running time
        if (deadline_3(i) - request_3(i) > T_run3 && deadline_4(i) - request_4(i) > T_run4 && request_3(i) ~= 0 && request_4(i) ~= 0 && request_3(i) ~= 24*(1/Ts)*days_sim && request_3(i) ~= 24*(1/Ts)*days_sim)
            flag = 0;
        end
    end
end
%set the deadline to use in simulation time step
deadline_sim_3 = zeros(1,days_sim*24*(1/Ts));
deadline_sim_4 = zeros(1,days_sim*24*(1/Ts));
for i = 1 : days_sim
    deadline_sim_3((i-1)*24*(1/Ts)+1 : i*24*(1/Ts)) = deadline_3(i);
    deadline_sim_4((i-1)*24*(1/Ts)+1 : i*24*(1/Ts)) = deadline_4(i);
end
%set the requests flag
request_3_flag = zeros(1,days_sim*24*(1/Ts));
request_4_flag = zeros(1,days_sim*24*(1/Ts));
for i = 1 : days_sim
    request_3_flag(request_3(i) : deadline_3(i)) = 1;
    request_4_flag(request_4(i) : deadline_4(i)) = 1;
end
%set output variables
Nsim = days_sim*24*(1/Ts);
tsim=0:Ts:Nsim-Ts;
Toss=length(tsim);
I1_t = zeros(1, Nsim);
I2_t = zeros(1, Nsim);
I3_t = zeros(1, Nsim);
I4_t = zeros(1, Nsim);
x_t = zeros(nx, Nsim+1);
x_t_w = zeros(nx_w, Nsim+1);
%initial value of x_t and x_t_w
x_t(:,1) = [20;23;25];
x_t_w(1) = 20;
%set capacity limit for 2 more day than the simulation time
cap = ones(1,(days_sim + 2)*24*(1/Ts))*4*3600; %7 kW
%running time variable for request 3 and request 4
total_I3 = zeros(1,days_sim*24*(1/Ts));
total_I4 = zeros(1,days_sim*24*(1/Ts));
%execution flag
execution_flag_3 = deadline_sim_3;
execution_flag_4 = deadline_sim_4;

%% Contollers
objective = 0;
constraints = [];
%heat pump
v = sdpvar(nu, N);
y = sdpvar(nu, N);
I1 = sdpvar(nu, N);
%water heater
v_w = sdpvar(nu_w, N);
y_w = sdpvar(nu_w, N);
I2 = sdpvar(nu_w, N);
v_capacity = sdpvar(1, N);
x0_w = sdpvar(1, 1);
x0 = sdpvar(3, 1);
x_w = x0_w;
x = x0;
cost = sdpvar(N, 1);
weather = sdpvar(N, 2);
dist = sdpvar(2, N);
capacity = sdpvar(1, N);
for k = 1 : N
    %heat pump
    x = Ad*x + Bd*I1(k) + Ed*weather(k,:)';
    y(k) = Cd*x;
    objective = objective + cost(k)*I1(k) + 10^6*v(k);
    constraints = [constraints, umin <= I1(k) <= umax];
    if(k>1)
        constraints = [constraints, dumin <= I1(k) - I1(k - 1) <= dumax];
    end
    constraints = [constraints, ymin - v(k) <= y(k) <= ymax + v(k) ];
    constraints = [constraints, v(k) >= 0 ];

    %water heater
    x_w = Adw*x_w + Bdw*I2(k) + Edw*dist(:,k);
    y_w(k) = Cdw*x_w;
    objective = objective+ cost(k)*I2(k) + 10^6*v_w(k);
    constraints = [constraints, u_w_min <= I2(k) <= u_w_max];
    if(k>1)constraints = [constraints, du_w_min <= I2(k) - I2(k - 1) <= du_w_max];end
    constraints = [constraints, ymin_w - v_w(k)  <= y_w(k) <= ymax_w + v_w(k)];
    constraints = [constraints, v_w(k) >= 0 ];
    constraints = [constraints, v_capacity(k) >= 0];
    %capacity constraints
    constraints = [constraints, I1(k) + I2(k) <= capacity(k) + v_capacity(k)];
    objective = objective + 10*10*10*v_capacity(k);
end
ops = sdpsettings('solver','gurobi','verbose', 2);
controller_1 = optimizer(constraints, objective, ops, {x0,x0_w,cost,weather,dist,capacity},{I1, I2});

%%
x_t(:, 1) = x0_in;
x_t_w(1) = x0_w_in;
tic
for t = 1 : days_sim*24*(1/Ts)
    
    %calculate day of the simulation
    day = 1 + floor(t/(24*(1/Ts)));
    %parameters to be used by matrices
    cost = priceHH(t : t+N-1);
    weather(:,1) = tempHH(t : t+N-1);
    weather(:,2) = GHIHH(t : t+N-1);
    dist(1,:) = Q_c(t : t+N-1);
    dist(2,:) = tempHH(t : t+N-1);
    capacity = cap(t : t+N-1);
    %check request3 and request 4
    if(request_3_flag(t) == 0 && request_4_flag(t) == 0)
        disp('Part1');
        %only run water heater and heat pump
        sol = controller_1(x_t(:, t),x_t_w(t),cost,weather,dist,capacity);
        temp1 = sol{1};
        temp2 = sol{2};
        I1_t(t) = temp1(1);
        I2_t(t) = temp2(1);
        I3_t(t) = 0;
        I4_t(t) = 0;
    elseif(request_3_flag(t) == 1 && request_4_flag(t) == 0)
        %I4_t is initialized 0 already
        % run water heater, heat pump and washing machine
        disp('part2')
        [I1_t(t), I2_t(t), I3_t(t), I4_t(t)] = MPC_HP_WH_3(x_t(:,t),Ad,Bd,Cd,Ed,nu,ymax,ymin,cost,weather,x_t_w(t),Adw,Bdw,Cdw,Edw,nu_w,ymax_w,ymin_w,dist,capacity,N,T_run3,total_I3(t),deadline_sim_3(t),umax,umin,u_w_max,u_w_min,dumax,dumin,du_w_max,du_w_min,t);
        %rest of the deadlines are made 0 for that day so that the request doeasn't
        %execute again
    elseif(request_3_flag(t) == 0 && request_4_flag(t) == 1)
        %I3_t(t) is already defined zero
        % run water heater, heat pump and clothes dryer
        disp('part3')
        [I1_t(t), I2_t(t), I3_t(t), I4_t(t)] = MPC_HP_WH_4(x_t(:,t),Ad,Bd,Cd,Ed,nu,ymax,ymin,cost,weather,x_t_w(t),Adw,Bdw,Cdw,Edw,nu_w,ymax_w,ymin_w,dist,capacity,N,T_run4,total_I4(t),deadline_sim_4(t),umax,umin,u_w_max,u_w_min,dumax,dumin,du_w_max,du_w_min,t);
        %rest of the deadlines are made 0 for that day so that the request doeasn't
        %execute again
    else
        disp('part4')
        % run every appliance
        [I1_t(t), I2_t(t), I3_t(t), I4_t(t)] = MPC_HP_WH_3_4(x_t(:,t),Ad,Bd,Cd,Ed,nu,ymax,ymin,cost,weather,x_t_w(t),Adw,Bdw,Cdw,Edw,nu_w,ymax_w,ymin_w,dist,capacity,N,T_run3,T_run4,total_I3(t),total_I4(t),deadline_sim_3(t),deadline_sim_4(t),umax,umin,u_w_max,u_w_min,dumax,dumin,du_w_max,du_w_min,t);
        %rest of the deadlines are made 0 for that day so that the request doeasn't
    end
    x_t(:,t+1) = Ad*x_t(:,t) + Bd*I1_t(t) + Ed*[tempHH(t), GHIHH(t)]';
    x_t_w(t+1) = Adw*x_t_w(t) + Bdw*I2_t(t) + Edw*[Q_c(t), tempHH(t)]';
    total_I3(t+1) = total_I3(t) + I3_t(t);
    total_I4(t+1) = total_I4(t) + I4_t(t);
    if(t == day*24*(1/Ts)-1)
          total_I3(t+1) = 0;
          total_I4(t+1) = 0;
    end
    t
end
toc
yalmip('clear')

%total cost

%results visualization
thours = 0 : Ts : days_sim*24-1;
con = zeros(1,days_sim*24*(1/Ts));
for i = 1 : days_sim*24*(1/Ts)
    con(i)=((I1_t(i)+I2_t(i))/3600)+I3_t(i)*3+I4_t(i)*4;
end
total_cost = sum(priceHH(1:length(con))'.*con)

figure
plot(thours, con(1 : length(thours)), 'Linewidth', 2 );
hold on
plot(thours, cap(1:length(thours))./3600,'k', 'Linewidth', 2, ...
    'Linestyle', '--');
xlabel('Time [Hours]','FontSize',14,'FontWeight','bold');
ylabel('Power consumption [kW]','FontSize',14,'FontWeight','bold')
legend('consumption', 'capacity constraint')
grid on
ax = gca;
set(ax,'FontSize',14,'FontWeight','bold')
title('Total Power Consumption')
axis([0,180,0,11])
hold off

%heat pump
figure
plot(thours, x_t(1,1 : length(thours)),'Linewidth',2)
hold on
plot(thours,ymax*ones(1,length(thours)),'k','LineWidth',2,'LineStyle','--')
plot(thours,ymin*ones(1,length(thours)),'k','LineWidth',2,'LineStyle','--')
xlabel('Time [Hours]', 'Fontsize',14,'FontWeight','bold');
ylabel('Room temperature, [\circ C]', 'FontSize',14,'fontWeight','bold')
legend('temp', 'constraint')
grid on
title('Room Temperature')
ax = gca;
set(ax,'FontSize',14,'FontWeight','bold')
hold off

figure
plot(thours,I1_t(1 : length(thours))./3600,'Linewidth',2);
hold on
plot(thours,umax*ones(1,length(thours))./3600,'k','LineWidth',2,'LineStyle','--')
plot(thours,umin*ones(1,length(thours))./3600,'k','LineWidth',2,'LineStyle','--')
xlabel('Time [Hours]', 'Fontsize',14,'FontWeight','bold');
ylabel('Power consumption [kW]','FontSize',14,'FontWeight','bold')
grid on
legend('consumption', 'constraint')
ax = gca;
set(ax,'FontSize',14,'FontWeight','bold')
title('Power consumption of heat pump')
hold off

%water heater
figure
plot(thours, x_t_w(1,1 : length(thours)),'Linewidth',2)
hold on
plot(thours,ymax_w*ones(1,length(thours)),'k','LineWidth',2,'LineStyle','--')
plot(thours,ymin_w*ones(1,length(thours)),'k','LineWidth',2,'LineStyle','--')
xlabel('Time [Hours]', 'Fontsize',14,'FontWeight','bold');
ylabel('Tank temperature, [\circ C]', 'FontSize',14,'fontWeight','bold')
legend('temp', 'constraint')
grid on
ax = gca;
set(ax,'FontSize',14,'FontWeight','bold')
title('Tank water Temperature')
hold off

figure
plot(thours,I2_t(1 : length(thours))./3600,'Linewidth',2);
hold on
plot(thours,u_w_max*ones(1,length(thours))./3600,'k','LineWidth',2,'LineStyle','--')
plot(thours,u_w_min*ones(1,length(thours))./3600,'k','LineWidth',2,'LineStyle','--')
xlabel('Time [Hours]', 'Fontsize',14,'FontWeight','bold');
ylabel('Power consumption [kW]','FontSize',14,'FontWeight','bold')
grid on
legend('consumption', 'constraint')
ax = gca;
set(ax,'FontSize',14,'FontWeight','bold')
title('Power consumption of water heater')
hold off

figure
plot(thours,3*I3_t(1:length(thours)),'Linewidth',2)
xlabel('Time [Hours]', 'Fontsize',14,'FontWeight','bold');
ylabel('Power consumption [kW]','FontSize',14,'FontWeight','bold')
grid on
legend('consumption')
ax = gca;
set(ax,'FontSize',14,'FontWeight','bold')
title('Power consumption of washing machine')
axis([0,168,0,4])
xbounds = xlim;
set(gca,'XTick',0:24:168);

figure
plot(thours,4*I4_t(1:length(thours)),'Linewidth',2)
xlabel('Time [Hours]', 'Fontsize',14,'FontWeight','bold');
ylabel('Power consumption [kW]','FontSize',14,'FontWeight','bold')
grid on
legend('consumption')
ax = gca;
set(ax,'FontSize',14,'FontWeight','bold')
title('Power consumption of dishwasher')
axis([0,168,0,5])
set(gca,'XTick',0:24:168);

P = zeros(1,length(I1_t));
for i = 1 : length(I1_t)
    if(I1_t(i) > 0)
        P(i) = 1;
    end
end
Q = zeros(1,length(I2_t));
for i = 1 : length(I2_t)
    if(I2_t(i) > 0)
        Q(i) = 1;
    end
end
%number of appliances
n = 4;
R = I3_t;
S = I4_t;
T = [P; Q; R; S];

violation = zeros(1, days_sim*24*(1/Ts));
for i = 1 : days_sim*24*(1/Ts)
    if (con(i) > 4)
        violation(i) = con(i) - 4;
    end
end
total_violation = sum(violation);