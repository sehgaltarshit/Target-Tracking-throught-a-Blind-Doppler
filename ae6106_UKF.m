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

% syms x_k y_k x_kDot y_kDot


% vars = [x_k, x_kDot, y_k, y_kDot];
% 
% Hx = jacobian(hx, vars);
% 
% % Convert both to fast numerical functions ONCE
% H_func  = matlabFunction(Hx, 'Vars', {x_k, x_kDot, y_k, y_kDot});
% hx_func = matlabFunction(hx,     'Vars', {x_k, x_kDot, y_k, y_kDot});


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
h_sensor = plot(NaN, NaN, 'g o', 'LineWidth', 1.5);

legend('Reference', 'True', 'Estimated', 'Raw Sensor');

prob = zeros(1,N-1);

% UKF Parameter
k = 1;
nx = 4;
rho = nx + k;
v = (0.5/rho)*ones(2*nx +1,1);
v(1) = k/rho;


for i = 1:N-1
    % error terms
    xk = xn(1,i);
    yk = xn(3,i);
    xkDot = xn(2,i);
    ykDot = xn(4,i);

    theta_k = atan2(yk,xk);
    lambda = exp(-0.5*sigma_theta^2);
    rk = sqrt(xk^2 + yk^2);
    rkDot = abs((xk*xkDot + yk*ykDot)/sqrt(xk^2 + yk^2));
    
    sigma_Xk2 = (lambda^-2 -2)*(rk*cos(theta_k))^2 + (rk^2 + sigma_r^2)*(1 + (lambda^4)*cos(2*theta_k))/2;
    sigma_Yk2 = (lambda^-2 -2)*(rk*sin(theta_k))^2 + (rk^2 + sigma_r^2)*(1 - (lambda^4)*cos(2*theta_k))/2;
    sigma_XkYk = (lambda^-2 -2)*(rk^2)*cos(theta_k)*sin(theta_k) + (rk^2 + sigma_r^2)*(lambda^4)*sin(2*theta_k)/2;
    
    Rk = [sigma_Xk2   sigma_XkYk   0;
          sigma_XkYk  sigma_Yk2    0;
              0           0    sigma_rdot^2];

    % True states
    uk = mvnrnd(zeros(2,1), Qk);
    wk = mvnrnd(zeros(3,1), Rk);
    xn(:,i+1) = F*xn(:,i); %+ G*uk';

    % sensor reading actual
    hx = [xk;
          yk;
         (xk*xkDot + yk*ykDot)/rk];
    % Condition for Blind Doppler
    if rkDot >= 100*(5/18)
        zn(:,i+1) = hx + wk';
        
        % only store valid sensor points
        sensor = [sensor, zn(1:2,i+1)];
    end
    

    %% UNSCENTED KALMAN FILTER
    % Generate sigma points for the conditional prior belief
    sigmaPts_n1n1 = GenerateSigmaPts(xnn(:,i), Pxxnn(:,:,i), nx, k);

    % Use the known nonlinear system equation 𝒇 to transform the sigma points
    xnn1i = zeros(4,2*nx+1);
    for j = 1:2*nx + 1
        xnn1i(:,j) = F*sigmaPts_n1n1(:,j);
    end

    % Combine the vectors to obtain the a priori state estimate at time 𝑛
    xnn1 = zeros(4,1);
    for j = 1:2*nx + 1
        xnn1 = xnn1 + v(j)*xnn1i(:,j);
    end

    % Estimating the a priori error covariance
    Pxxnn1 = zeros(4,4);
    for j = 1:2*nx + 1
        Pxxnn1 = Pxxnn1 + v(j)*(xnn1i(:,j) - xnn1)*(xnn1i(:,j) - xnn1)';
    end

    Pxxnn1 = Pxxnn1 + G*Qk*G';

    % Generate sigma points for the conditional predicted prior belief
    sigmaPts_nn1 = GenerateSigmaPts(xnn1, Pxxnn1, nx, k);

    % Apply Nonlinear measurement equation 𝒉 to transform the sigma points

    znn1i = zeros(3,2*nx+1);
    for j = 1:2*nx + 1
        x_k = sigmaPts_nn1(1,j);
        x_kDot = sigmaPts_nn1(2,j);
        y_k = sigmaPts_nn1(3,j);
        y_kDot = sigmaPts_nn1(4,j);

        r_k = sqrt(x_k^2 + y_k^2);

        hx2 = [x_k;
              y_k;
             (x_k*x_kDot + y_k*y_kDot)/r_k];

        znn1i(:,j) = hx2;
    end

    % Combine the znn1i vectors to obtain the a priori state estimate at time 𝑛
    znn1 = zeros(3,1);
    for j = 1:2*nx + 1
        znn1 = znn1 + v(j)*znn1i(:,j);
    end

    % Estimate the a priori measurement error covariance as
    Pzznn1 = zeros(3,3);
    for j = 1:2*nx + 1
        Pzznn1 = Pzznn1 + v(j)*(znn1i(:,j) - znn1)*(znn1i(:,j) - znn1)';
    end
    Pzznn1 = Pzznn1 + Rk;

    Pxznn1 = zeros(4,3);
    for j = 1:2*nx + 1
        Pxznn1 = Pxznn1 + v(j)*(xnn1i(:,j) - xnn1)*(znn1i(:,j) - znn1)';
    end

    % Condition for Blind Doppler
    if rkDot >= 100*(5/18)
        Pd = 1;
    else
        Pd = 0;
    end
    prob(i) = Pd;

    % Kalman filter correction/update
    Kn = Pd*Pxznn1/Pzznn1;
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
plot(zn(1,:), Color='g')
title("x_{k}")
legend(["Estimated", "Actual", "Raw Sensor"])

subplot(2,2,2)
plot(xnn(3,:), '-o', Color='b')
hold on
plot(xn(3,:), Color='r')
plot(zn(2,:), Color='g')
title("y_{k}")
legend(["Estimated", "Actual", "Raw Sensor"])

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
% plot(prob,'-k')


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


function [sigmaPts] = GenerateSigmaPts(mean, P, nx, k)
    sigmaPts = zeros(nx,2*nx+1);
    sigmaPts(:,1) = mean;
    rho = nx + k;
    sqrtRhoP = sqrtm(rho*P);

    for i = 1:nx
        sigmaPts(:,i+1) = mean + sqrtRhoP(:,i);
    end

    for i = 1:nx
        sigmaPts(:,i+nx+1) = mean - sqrtRhoP(:,i);
    end
    
end
