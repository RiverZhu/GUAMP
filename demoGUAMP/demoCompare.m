clc
clear
close all
addpath('../main')
addpath('../classification')
addpath('../phase')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% simulation parameters
isCmplx = false; % must set =false
N = 512; % signal dimension
del = 4;  % measurement rate M/N [4.0]
M = round(del*N);
beta = 0.1; % sparsity rate K/N [0.1]
SNRdB = 30; % [40]
L = 1;
B_all = [1,2];
crou_all = [0,0.15];
plot_iter = 1;

% algorithmic parameters
maxIt = 100; % max iterations for VAMP
tol = min(1e-3,max(1e-6,10^(-SNRdB/10))); % stopping tolerance for VAMP
altUpdate = false; % alternate updates of x and z in VAMP?
damp = 1.0; % damping parameter for mean term
dampGam = 1.0; % damping parameter for precision term
% other defaults
fixed_K = true; % used fixed sparsity K=E{K}=round(rho*M)?
Afro2 = N; % squared Frobenius norm of matrix
xvar0 = 1; % prior variance of x elements
xmean1 = 0; % prior mean of non-zero x coefs
xvar1 = xvar0/beta; % prior variance of non-zero x coefs
wvar = (Afro2/M)*10^(-SNRdB/10)*beta*(abs(xmean1)^2 + xvar1); 

%% setup GVAMP
vampOpt = VampGlmOpt;
vampOpt.nitMax = maxIt;
vampOpt.tol = tol; 
vampOpt.damp = damp;
vampOpt.dampGam = dampGam;
vampOpt.dampConfig = [0,1,1,1,0,0, 0,0,0,0,0,1]; % from dampTest
vampOpt.verbose = true;
vampOpt.altUpdate = altUpdate;
vampOpt.silent = true; % suppress warnings about negative precisions

%% setup GAMP
% optGAMP = GampOpt('legacyOut',false,'uniformVariance',true,'adaptStepBethe',true,'step',0.2,'stepIncr',1.05,'stepWindow',1,'stepMax',0.5,'stepMin',0.02,'tol',tol/100,'nit',500,'xvar0',beta*(abs(xmean1)^2+xvar1));
optGAMP = GampOpt('legacyOut',false,'uniformVariance',true,'adaptStepBethe',false,'step',1,'tol',tol/100,'nit',100,'xvar0',beta*(abs(xmean1)^2+xvar1));

%%  setup GUAMP
Init.Iter_max = 100;
Init.L = L; 
Init.Iter_A_max = 4;
Init.Iter_B_max = 1;

%% iteration histroy
gampNit = optGAMP.nit;
gampNMSEdB_ = nan(L,gampNit);
gampNMSEdB_debiased_ = nan(L,gampNit);
gampNMSEdB_all = nan(length(B_all)*length(crou_all),gampNit);
gampNMSEdB_debiased_all = nan(length(B_all)*length(crou_all),gampNit);

vampNit = vampOpt.nitMax;
VampNMSEdB_ = nan(L,vampNit);
VampNMSEdB_debiased_ = nan(L,vampNit);
VampNMSEdB_all = nan(length(B_all)*length(crou_all),vampNit);
VampNMSEdB_debiased_all = nan(length(B_all)*length(crou_all),vampNit);

UTgampNit = Init.Iter_max;
UTgampNMSEdB_ = nan(L,UTgampNit);
UTgampNMSEdB_debiased_ = nan(L,UTgampNit);
UTgampNMSEdB_all = nan(length(B_all)*length(crou_all),UTgampNit);
UTgampNMSEdB_debiased_all = nan(length(B_all)*length(crou_all),UTgampNit);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Start here
tic
for B_idx = 1:length(B_all)
    B = B_all(B_idx);

    for crou_idx = 1:length(crou_all)
        crou = crou_all(crou_idx);

        % generate signal
        x = zeros(N,L);
        for l=1:L
            if fixed_K
                supp = randperm(N,round(beta*N));
            else
                supp = find(rand(N,1)<beta);
            end
            K = length(supp);
            x(supp,l) = xmean1 + sqrt(xvar1)*randn(K,1);
        end

        % generate linear transform
        for cr =1 :M
            for cc=1:M
                R(cr,cc)=crou.^abs(cr-cc);
            end
        end
        RL=sqrt(R);

        for cr =1 :N
            for cc=1:N
                R1(cr,cc)=crou.^abs(cr-cc);
            end
        end
        RR=sqrt(R1);
        A_mat = RL*randn(M,N)*RR;
        A_mat = A_mat/norm(A_mat,'fro')*sqrt(N); % Normalization
        mat = genMatSVD_UT(A_mat);

        % fxnHandles generate
        A = mat.fxnA; Ah = mat.fxnAh;
        U = mat.fxnU; Uh = mat.fxnUh;
        V = mat.fxnV; Vh = mat.fxnVh;
        d = mat.s.^2;

        % generate noise
        w = sqrt(wvar)*randn(M,L);

        % crease noisy observations
        z = A(x);
        if B == 1
            y = ((z+w)>0);
            EstimOut = ProbitEstimOut(y,0,wvar); % establish likelihood
            EstimOut2bit = [];

        else
            % 2bit quantization
            y_unq = z+w;
            Num_bins = 2^B;
            sig_var = Afro2*beta*(abs(xmean1)^2 + xvar1)/M;
            Delta_max = 3*sqrt(sig_var);
            Delta_min = -Delta_max;
            step_intval = (Delta_max - Delta_min)/(Num_bins);
            y_bin = floor((y_unq-Delta_min)/step_intval);
            y_bin(y_unq>=Delta_max) = Num_bins-1;
            y_bin(y_unq<=Delta_min) = 0;
            wvar_vec = repmat(wvar,M,1);
            EstimOut2bit = @(phat,pvar) GaussianMomentsComputation_MJH(y_bin, phat, pvar, Delta_min, Num_bins, step_intval, wvar_vec);
            EstimOut = [];
        end
        EstimIn = SparseScaEstim(AwgnEstimIn(xmean1,xvar1),beta); % establish input denoiser

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% initialize GVAMP
        debias = @(xhat,x) bsxfun(@times, xhat, sum(conj(xhat).*x,1)./sum(abs(xhat).^2,1)); % establish debiasing/disambiguation
        vampOpt.fxnErr1 = @(x1,z1) 10*log10( sum(abs(x1-x).^2,1)./sum(abs(x).^2,1) );
        vampOpt.fxnErr2 = @(x1,z1) 10*log10( sum(abs(x-debias(x1,x)).^2,1)./sum(abs(x).^2,1) );
        vampOpt.fxnErr3 = @(p1,gam1z,z1,eta1z,p2,gam2z,...
            r1,gam1x,x1,eta1x,r2,gam2x,...
            x2,z2,eta2x,eta2z) 10*log10( sum(abs(x-r1).^2,1)./sum(abs(x).^2,1) );
        vampOpt.Ah = Ah; vampOpt.d = d; vampOpt.N = N;
        vampOpt.U = U; vampOpt.Uh = Uh; vampOpt.V = V; vampOpt.Vh = Vh;

        zvar = beta*(abs(xmean1)^2+xvar1)*Afro2/M;
        zvarTest = norm(z,'fro')^2/numel(z); % for testing
        pvar = 1e0*zvar; % set pvar between 0 and zvar; note pvar=zvar gives phat=0
        phat = (1-pvar/zvar)*z ...
            + sqrt((1-pvar/zvar)*pvar/2)*(randn(M,L)+(1j^isCmplx)*randn(M,L));
        rvar = 1e2*xvar0; % set rvar between 0 and inf; note rvar=0 gives rhat=x
        rhat = x + sqrt(rvar/2)*(randn(N,L)+(1j^isCmplx)*randn(N,L));
        vampOpt.p1init = phat;
        vampOpt.r1init = rhat;
        vampOpt.gam1zinit = 1./pvar;
        vampOpt.gam1xinit = 1./rvar;
        vampOpt.B = B;
        vampOpt.EstimOut2bit = EstimOut2bit;

        %% run GVAMP
        [x1,vampEstFin] = VampGlmEst2(EstimIn,EstimOut,A,vampOpt);
        vampNMSEdB_ = vampEstFin.err1;
        vampNMSEdB_debiased_ = vampEstFin.err2;
        vampNit_tmp = vampEstFin.nit;

        vampNMSEdB_(:,vampNit_tmp+1:vampOpt.nitMax) = repmat(vampNMSEdB_(:,vampNit_tmp),1,vampOpt.nitMax-vampNit_tmp);
        vampNMSEdB_debiased_(:,vampNit_tmp+1:vampOpt.nitMax) = repmat(vampNMSEdB_debiased_(:,vampNit_tmp),1,vampOpt.nitMax-vampNit_tmp);

        %% initialize GAMP
        [xhat,xvar] = EstimIn.estim(rhat,rvar);
        optGAMP.xhat0 = xhat;
        optGAMP.xvar0 = xvar;
        optGAMP.B = B;
        optGAMP.EstimOut2bit = EstimOut2bit;
        %% run GAMP
        Agamp = FxnhandleLinTrans(M,N,A,Ah,Afro2/(M*N));
        [gampEstFin,optGampFin,gampEstHist] = gampEst(EstimIn,EstimOut,Agamp,optGAMP);

        %% Run GUAMP
        Init.EstimOut2bit = EstimOut2bit;
        Init.B = B;
        A_mat2 = A(eye(N));
        UTgampxhat_his = fun_GUAMP(A_mat2, Init, EstimIn, EstimOut);

        for l=1:L
            xhat_ = gampEstHist.xhat((l-1)*N+[1:N],:);
            gampNMSEdB_(l,:) = 10*log10(sum(abs(xhat_-x(:,l)*ones(1,gampNit)).^2,1)/norm(x(:,l))^2);
            gampNMSEdB_debiased_(l,:) = 10*log10(sum(abs( debias(xhat_,x(:,l)*ones(1,gampNit))-x(:,l)*ones(1,gampNit)).^2,1)/norm(x(:,l))^2);

            VampNMSEdB_(l,:) = vampNMSEdB_(l,:);
            VampNMSEdB_debiased_(l,:) = vampNMSEdB_debiased_(l,:);

            xhat_ = UTgampxhat_his((l-1)*N+[1:N],:);
            UTgampNMSEdB_(l,:) = 10*log10(sum(abs(xhat_-x(:,l)*ones(1,UTgampNit)).^2,1)/norm(x(:,l))^2);
            UTgampNMSEdB_debiased_(l,:) = 10*log10(sum(abs( debias(xhat_,x(:,l)*ones(1,UTgampNit))-x(:,l)*ones(1,UTgampNit)).^2,1)/norm(x(:,l))^2);
        end

        idx = ((B_idx-1)*length(crou_all)+crou_idx);
        gampNMSEdB_all(idx,:) = gampNMSEdB_;
        gampNMSEdB_debiased_all(idx,:) = gampNMSEdB_debiased_;
        VampNMSEdB_all(idx,:) = VampNMSEdB_;
        VampNMSEdB_debiased_all(idx,:) = VampNMSEdB_debiased_;
        UTgampNMSEdB_all(idx,:) = UTgampNMSEdB_;
        UTgampNMSEdB_debiased_all(idx,:) = UTgampNMSEdB_debiased_;

    end
end
toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot results
lw = 2;
fsz = 14;
msz = 2;

if plot_iter
    figure(1)
    subplot(2,1,1)
    plot(1:gampNit,gampNMSEdB_debiased_all(1,:),'-rd','LineWidth',lw,'MarkerSize',msz)
    hold on
    plot(1:vampNit,VampNMSEdB_debiased_all(1,:),'-xb','LineWidth',lw,'MarkerSize',msz)
    plot(1:UTgampNit,UTgampNMSEdB_debiased_all(1,:),'-ok','LineWidth',lw,'MarkerSize',msz)
    plot(1:gampNit,gampNMSEdB_all(length(crou_all)+1,:),'-.rd','LineWidth',lw,'MarkerSize',msz)
    plot(1:vampNit,VampNMSEdB_all(length(crou_all)+1,:),'-.xb','LineWidth',lw,'MarkerSize',msz)
    plot(1:UTgampNit,UTgampNMSEdB_all(length(crou_all)+1,:),'-.ok','LineWidth',lw,'MarkerSize',msz)
    hold off
    xlabel('Number of iterations','FontSize',fsz)
    ylabel('${\rm dNMSE}(\hat{\mathbf z})/{\rm NMSE}(\hat{\mathbf z})$','FontSize',fsz-1,'Interpreter','latex')
    legend('GAMP (1bit)','GVAMP (1bit)','GUAMP (1bit)','GAMP (2bit)','GVAMP (2bit)','GUAMP (2bit)')
    title('dNMSE/NMSE vs iter[p=0]','FontSize',fsz)

    subplot(2,1,2)
    plot(1:gampNit,gampNMSEdB_debiased_all(2,:),'-rd','LineWidth',lw,'MarkerSize',msz)
    hold on
    plot(1:vampNit,VampNMSEdB_debiased_all(2,:),'-xb','LineWidth',lw,'MarkerSize',msz)
    plot(1:UTgampNit,UTgampNMSEdB_debiased_all(2,:),'-ok','LineWidth',lw,'MarkerSize',msz)
    plot(1:gampNit,gampNMSEdB_all(length(crou_all)+2,:),'-.rd','LineWidth',lw,'MarkerSize',msz)
    plot(1:vampNit,VampNMSEdB_all(length(crou_all)+2,:),'-.xb','LineWidth',lw,'MarkerSize',msz)
    plot(1:UTgampNit,UTgampNMSEdB_all(length(crou_all)+2,:),'-.ok','LineWidth',lw,'MarkerSize',msz)
    hold off
    xlabel('Number of iterations','FontSize',fsz)
    ylabel('${\rm dNMSE}(\hat{\mathbf z})/{\rm NMSE}(\hat{\mathbf z})$','FontSize',fsz-1,'Interpreter','latex')
    title('dNMSE/NMSE vs iter[p=0.15]','FontSize',fsz)
end % plot_iter
