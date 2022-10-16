function xhat_his = fun_GUAMP(A, Init, EstimIn, EstimOut)

%% Alogrithm parameter Initialization

[M, N] = size(A);
Iter_max = Init.Iter_max;
L = Init.L;
Iter_B_max = Init.Iter_B_max;
Iter_A_max = Init.Iter_A_max;
xhat_his = zeros(N*L,Iter_max);

[U,S,V] = svd(A,'econ');
R = rank(A);
U = U(:,1:R);
Q = S*V';
Q = Q(1:R,:);
Q_2 = Q.^2;
U_2 = U.^2;

%% UTGAMP Setting
tau_x = 1e4*ones(N,1);
xhat = zeros(N,1);
shat_A = zeros(R,1); % shat_UTAMP = zeros(min(M,N),1);
tau_b = 1e4*ones(R,1); % tau_b = 1e2*ones(min(M,N),1);
bhat = zeros(R,1); % bhat = zeros(min(M,N),1);
shat_B = zeros(M,1);
zvarToPvarMax = inf;
tau_p_A = Q_2*tau_x;
phat_A = Q*xhat-tau_p_A.*shat_A;
b_A_ext = phat_A;
v_A_ext = tau_p_A;

%% run UTGAMP
for iter = 1:Iter_max

    for iter_B = 1:Iter_B_max
        %% ModuleB:output linear step
        tau_p_B = U_2*tau_b; % 
%         tau_p_B = mean(tau_p_B)*ones(size(tau_p_B));
        phat_B = U*bhat - tau_p_B.*shat_B;
        if Init.B==1
            [zhat,zvar] = EstimOut.estim(phat_B,tau_p_B);
        else
            [zhat,zvar] = Init.EstimOut2bit(phat_B,tau_p_B);
        end
%         zvar = mean(zvar)*ones(size(zvar));
        % ModuleB:output nonlinear step
        shat_B = (1./tau_p_B).*(zhat-phat_B);
        tau_s_B = (1./tau_p_B).*(1-min(zvar./tau_p_B,zvarToPvarMax));

        % ModuleB:Input linear step
        tau_r_B = 1./(U_2'*tau_s_B);
        rhat_B = bhat+tau_r_B.*(U'*shat_B);

        % ModuleB:Input nonlinear step
        tau_b = tau_r_B.*v_A_ext./(tau_r_B+v_A_ext);
        tau_b(tau_b<0) = 1e6;
        bhat = tau_b.*(rhat_B./tau_r_B+b_A_ext./v_A_ext);
        % bhat = b_A_ext+v_A_ext.*(rhat_B-b_A_ext)./(tau_r_B+v_A_ext);
    end

    b_B_ext = rhat_B;
    v_B_ext = tau_r_B;

    %% Module A
    for iter_A = 1:Iter_A_max

        % ModuleA:output nonlinear step
        shat_A = (b_B_ext-phat_A)./(v_B_ext+tau_p_A);
        tau_s_A = 1./(v_B_ext+tau_p_A);
        % shat_A = (bhat-phat_UTAMP)./tau_p_UTAMP;
        % tau_s_A = (1-tau_b./tau_p_UTAMP)./tau_p_UTAMP;

        % ModuleA:Input linear step
        tau_r_A = 1./(Q_2'*tau_s_A);
        rhat_A = xhat+tau_r_A.*(Q'*shat_A);

        % ModuleA:Input nonlinear step
        [xhat,tau_x] = EstimIn.estim(rhat_A,tau_r_A);

        % ModuleA:output linear step
        tau_p_A = Q_2*tau_x;
        phat_A = Q*xhat-tau_p_A.*shat_A;
    end

    xhat_his(:,iter) = xhat(:);
    b_A_ext = phat_A;
    v_A_ext = tau_p_A;

end

end