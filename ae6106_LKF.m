clc 
clear all

rng("default")  % put this as the very first line

% Parameters
Tk = 2;
sigma_r = 250;
sigma_rdot = 3;
sigma_theta = deg2rad(1);
sigma_w = 9.81/(sqrt(Tk));

% number of steps
n1 = 45;  % North to South straight part for 90s
%n2 = 0; %6;   % 90 deg turn in 12 sec  (3g)
n2 = 13;   % straight towards west
%n4 = 0; %6;   % 90 deg turn in 12 sec  (3g)
n3 = 30;   % continue moving straight towards the radar
N = n1 + n2 + n3;   % total number of points

% matrices
F = [1 Tk 0 0;
     0  1 0 0;
     0  0 1 Tk;
     0  0 0 1;];

G = [(Tk^2)/2  0;
       Tk      0;
       0  (Tk^2)/2;
       0       Tk;];

Qk = [sigma_w^2  0;
       0   sigma_w^2;];

syms x_k y_k x_kDot y_kDot

hx = [x_k;
      y_k;
      (x_k*x_kDot + y_k*y_kDot)/sqrt(x_k^2 + y_k^2)];

vars = [x_k, x_kDot, y_k, y_kDot];

Hx = jacobian(hx, vars);

H = double(subs(Hx, [x_k, x_kDot, y_k, y_kDot], [0, 0, 60000, -800*(5/18)]));
hx_func = matlabFunction(hx,     'Vars', {x_k, x_kDot, y_k, y_kDot});

xn = zeros(4,N);
zn = zeros(3,N);
xn(:,1) = [0;   0;   60000;    -800*(5/18)];



xnn = zeros(4,N);
xnn(:,1) = 0.95*[0;   0;   60000;    -800*(5/18)];

Pxxnn = zeros(4,4,N);
Pxxnn(:,:,1) = diag([sigma_r^2, (sigma_r/Tk)^2, sigma_r^2, (sigma_r/Tk)^2]);

% NO NOISE MODEL

% only model no noise
xnM = zeros(4,N);
xnM(:,1) = [0;   0;   60000;    -800*(5/18)];

for i = 1:N-1
    % without noise state
    xnM(:,i+1) = F*xnM(:,i);
    % no noise
    xnM(:,n1+1) = [xnM(1,n1);   -800*(5/18);   xnM(3,n1);    0];
    xnM(:,n1+n2+1) = [xnM(1,n1+n2);   0;   xnM(3,n1+n2);    -800*(5/18)];
end

% PLOT SETUP
% Figure setup
% Storage
true_traj = [];
est_traj  = [];
sensor = [];

figure;
hold on; grid on; axis equal;
xlabel('X'); ylabel('Y');

% Plot reference (static)
plot(xnM(1,:), xnM(3,:), 'k', 'LineWidth', 1);

% Create plot handles (IMPORTANT)
h_true = plot(NaN, NaN, 'r', 'LineWidth', 1.5);
h_est  = plot(NaN, NaN, 'b o', 'LineWidth', 1.5);
h_sensor  = plot(NaN, NaN, 'g o', 'LineWidth', 1.5);

legend('Reference', 'True', 'Estimated', 'Raw Sensor');
title("LKF")



for i = 1:N-1
    % error terms
    xk = xn(1,i);
    yk = xn(3,i);
    xkDot = xn(2,i);
    ykDot = xn(4,i);
    rkDot = abs((xk*xkDot + yk*ykDot)/sqrt(xk^2 + yk^2));


    theta_k = atan2(yk,xk);
    lambda = exp(-0.5*sigma_theta^2);
    rk = sqrt(xk^2 + yk^2);
    
    sigma_Xk2 = (lambda^-2 -2)*(rk*cos(theta_k))^2 + (rk^2 + sigma_r^2)*(1 + (lambda^4)*cos(2*theta_k))/2;
    sigma_Yk2 = (lambda^-2 -2)*(rk*sin(theta_k))^2 + (rk^2 + sigma_r^2)*(1 - (lambda^4)*cos(2*theta_k))/2;
    sigma_XkYk = (lambda^-2 -2)*(rk^2)*cos(theta_k)*sin(theta_k) + (rk^2 + sigma_r^2)*(lambda^4)*sin(2*theta_k)/2;
    
    Rk = [sigma_Xk2   sigma_XkYk   0;
          sigma_XkYk  sigma_Yk2    0;
              0           0    sigma_rdot^2];


    % True states
    uk = mvnrnd(zeros(2,1), Qk);
    wk = mvnrnd(zeros(3,1), Rk);

    xn(:,i+1) = F*xn(:,i); % + G*uk';

    % substitute xk in h(x)
    x_val = xn(:,i+1);   % [x_k, y_k, x_kDot, y_kDot]

    % Replace hx_val subs:
    hx_val = hx_func(xn(1,i+1), xn(2,i+1), xn(3,i+1), xn(4,i+1));
    hx_val = double(hx_val);

    % sensor reading actual
    % Condition for Blind Doppler
    if rkDot >= 100*(5/18)
        zn(:,i+1) = hx_val + wk';
        
        % only store valid sensor points
        sensor = [sensor, zn(1:2,i+1)];
    end

    % Kalman Filter
    xnn1 = F*xnn(:,i);
    Pxxnn1 = F*Pxxnn(:,:,i)*F' + G*Qk*G';

    znn1 = H*xnn1;
    Pzznn1 = H*Pxxnn1*H' + Rk;
    Pxznn1 = Pxxnn1*H';
    
    if rkDot >= 100*(5/18)
        Kn = Pxznn1/Pzznn1;
    else
        Kn = zeros(4,3);
    end
    xnn(:,i+1) = xnn1 + Kn*(zn(:,i+1) - znn1);
    Pxxnn(:,:,i+1) = Pxxnn1 - Kn*Pzznn1*Kn';

    % update for turns
    xn(:,n1+1) = [xn(1,n1);   -800*(5/18);   xn(3,n1);    0];
    xn(:,n1+n2+1) = [xn(1,n1+n2);   0;   xn(3,n1+n2);    -800*(5/18)];


    % UPDATE PLOT
    % Store trajectories
    x_true = [xn(1,i); xn(3,i)];
    x_est  = [xnn(1,i); xnn(3,i)];
    
    true_traj = [true_traj, x_true];
    est_traj  = [est_traj,  x_est];
    %sensor    = [sensor, zn(1:2,i+1)];

    % Update plots
    set(h_true, 'XData', true_traj(1,:), 'YData', true_traj(2,:));
    set(h_est,  'XData', est_traj(1,:),  'YData', est_traj(2,:));
    set(h_sensor,  'XData', sensor(1,:),  'YData', sensor(2,:));

    drawnow;

    % disp(cond(Pzznn1))  % condition number of innovation covariance
    % disp(norm(H))   % norm of Jacobian
end


figure;

subplot(2,2,1)
plot(xnn(1,:), '-o', Color='b')
hold on
plot(xn(1,:), Color='r')
title("x_{k}")
legend(["Estimated", "Actual"])

subplot(2,2,2)
plot(xnn(3,:), '-o', Color='b')
hold on
plot(xn(3,:), Color='r')
title("y_{k}")
legend(["Estimated", "Actual"])

subplot(2,2,3)
plot(xnn(2,:), '-o', Color='b')
hold on
plot(xn(2,:), Color='r')
title("xDot_{k}")
legend(["Estimated", "Actual"])

subplot(2,2,4)
plot(xnn(4,:), '-o', Color='b')
hold on
plot(xn(4,:), Color='r')
title("yDot_{k}")
legend(["Estimated", "Actual"])


% figure;
% subplot(2,2,1)
% plot(xnM(1,:), '-o', Color='b')
% title("x_{k}")
% 
% 
% subplot(2,2,2)
% plot(xnM(3,:), '-o', Color='b')
% title("y_{k}")
% 
% subplot(2,2,3)
% plot(xnM(2,:), '-o', Color='b')
% title("xDot_{k}")
% 
% subplot(2,2,4)
% plot(xnM(4,:), '-o', Color='b')
% title("yDot_{k}")