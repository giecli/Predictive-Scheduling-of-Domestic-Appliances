function [out1, out2, out3, out4] = MPC_HP_WH_3_4(x0,Ad,Bd,Cd,Ed,nu,ymax,ymin,cost,weather,x0_w,Adw,Bdw,Cdw,Edw,nu_w,ymax_w,ymin_w,dist,capacity,N,T_run_3,T_run_4,total_I3_0,total_I4_0,deadline_sim_3,deadline_sim_4,umax,umin,u_w_max,u_w_min,dumax,dumin,du_w_max,du_w_min,t)
objective = 0;
constraints = [];
%heat pump
v = sdpvar(repmat(nu,1,N),repmat(1,1,N));
y = sdpvar(repmat(nu,1,N),repmat(1,1,N));
I1 = sdpvar(repmat(1,1,N),repmat(1,1,N));
%water heater
v_w = sdpvar(repmat(nu_w,1,N),repmat(1,1,N));
y_w = sdpvar(repmat(nu_w,1,N),repmat(1,1,N));
I2 = sdpvar(repmat(1,1,N),repmat(1,1,N));
v_capacity = sdpvar(repmat(1,1,N),repmat(1,1,N));
%request3
I3 = binvar(repmat(1,1,N),repmat(1,1,N));
u_3 = sdpvar(repmat(1,1,N),repmat(1,1,N));
total_I3 = sdpvar(repmat(1,1,N+1),repmat(1,1,N+1));
total_I3{1} = total_I3_0;

%request4
I4 = binvar(repmat(1,1,N),repmat(1,1,N));
u_4 = sdpvar(repmat(1,1,N),repmat(1,1,N));
total_I4 = sdpvar(repmat(1,1,N+1),repmat(1,1,N+1));
total_I4{1} = total_I4_0;
x_w = x0_w;
x = x0;
d1 = binvar(1, 1);
d2 = binvar(1, 1);
for k = 1 : N
     x = Ad*x + Bd*I1{k} + Ed*weather(k,:)';
    y{k} = Cd*x;
    objective = objective + cost(k)*I1{k} + 10^6*v{k};
    constraints = [constraints, umin <= I1{k} <= umax];
    if(k>1)constraints = [constraints, dumin <= I1{k} - I1{k-1} <= dumax];end
    constraints = [constraints, ymin - v{k} <= y{k} <= ymax + v{k} ];
    constraints = [constraints, v{k} >= 0 ];

    %water heater
    x_w = Adw*x_w + Bdw*I2{k} + Edw*dist(:,k);
    y_w{k} = Cdw*x_w;
    objective = objective+ cost(k)*I2{k} + 10^6*v_w{k};
    constraints = [constraints, u_w_min <= I2{k} <= u_w_max];
    if(k>1)constraints = [constraints, du_w_min <= I2{k} - I2{k-1} <= du_w_max];end
    constraints = [constraints, ymin_w - v_w{k}  <= y_w{k} <= ymax_w + v_w{k}];
    constraints = [constraints, v_w{k} >= 0 ];
    %request_3
    objective = objective + cost(k)*I3{k} + 10^2*I3{k};
    total_I3{k+1} = total_I3{k} + I3{k};
    u_3{k} = 3600*3*I3{k};
    %request_4
    objective = objective + cost(k)*I4{k} + 10^2*I4{k};
    total_I4{k+1} = total_I4{k} + I4{k};
    u_4{k} = 3600*4*I4{k};
    %capacity constraint
    constraints = [constraints, I1{k} + I2{k} + u_3{k} + u_4{k} <= capacity(k) + v_capacity{k}];
    constraints = [constraints, v_capacity{k} >= 0];
    objective = objective + 10*10*10*v_capacity{k};
    %capacity constraint
    if (k < deadline_sim_4 - t)
        constraints = [constraints, total_I4{deadline_sim_4 - t} == T_run_4];
    end
% constraints = [constraints, implies(k < deadline_sim_3-t, total_I3{deadline_sim_3 - t} == T_run_3)];
% constraints = [constraints, implies(k < deadline_sim_4-t, total_I4{deadline_sim_4 - t} == T_run_4)];

%     constraints = [constraints, implies(k <= deadline_sim_3-t, d1)];
%     constraints = [constraints, implies(d1, total_I3{deadline_sim_3 - t} == T_run_3)];

    if (k < deadline_sim_3 - t)
        constraints = [constraints, total_I3{deadline_sim_3 - t} == T_run_3];
    end
%     constraints = [constraints, implies(k <= deadline_sim_4 - t, d2)];
%     constraints = [constraints, implies(d2, total_I4{deadline_sim_4 - t} == T_run_4)];    
end

ops = sdpsettings('solver','gurobi','gurobi.mipgap', 0.1,'gurobi.timelimit', 50, 'verbose', 0);
optimize(constraints,objective,ops);
out1 = double(I1{1});
out2 = double(I2{1});
out3 = double(I3{1});
out4 = double(I4{1});
end
