% Generate an MxN matrix with the 'economy' SVD form 
%    A = U*diag(s)*V' 
% where
%    s is non-zero and length min(M,N) ... "singular values"
%    U is size M x min(M,N) and obeys U'*U = I ... "left singular vectors"
%    V is size N x min(M,N) and obeys V'*V = I ... "right singular vectors"


function out = genMatSVD_UT(A)

[M,N] = size(A);
[Uhaar,S,Vhaar] = svd(A);
s = diag(S);
s = s(1:min(M,N));
d = s.^2;

% For UT-GAMP
% [U,S,V] = svd(A,'econ');
% R = rank(A);
% U = U(:,1:R);
% Q = S*V';
% Q = Q(1:R,:);
% Q_2 = Q.^2;
% U_2 = U.^2;

% opt.Afro2 = N;

%% generate left singular value matrix U
if M<=N
    U = Uhaar; % square
else % M>N
    Utall = A*(Vhaar*spdiags(1./sqrt(d),0,N,N)); % tall MxN
end

if M<=N
    fxnU = @(z) U*z;
    fxnUh = @(z) U'*z;
else % leverage M>N
    fxnU = @(z) Utall*z(1:N,:); % tall U
    fxnUh = @(z) speye(M,N)*(Utall'*z); % tall U
end

%% generate right singular value matrix V
if M<=N
    Vtall = A'*(Uhaar*spdiags(1./sqrt(d),0,M,M)); % tall NxM
else % M>N
    V = Vhaar; % square
end
if M<=N
    fxnV = @(x) Vtall*x(1:M,:);
    fxnVh = @(x) speye(N,M)*(Vtall'*x);
else % M>N
    fxnV = @(x) V*x;
    fxnVh = @(x) V'*x;
end

%% generate singular value vector/matrix S
if M<=N
    fxnS = @(x) bsxfun(@times,s,x(1:M,:)); % wide
    %fxnSh = @(z) [bsxfun(@times,s,z);zeros(N-M,size(z,2))]; % tall
    fxnSh = @(z) spdiags(s,0,N,M)*z; % tall
else
    fxnSh = @(z) bsxfun(@times,s,z(1:N,:)); % wide
    %fxnS = @(x) [bsxfun(@times,s,x);zeros(M-N,size(x,2))]; % tall
    fxnS = @(x) spdiags(s,0,M,N)*x; % tall
end

%% fill output structure
out.fxnA = @(x) fxnU(fxnS(fxnVh(x)));
out.fxnAh = @(z) fxnV(fxnSh(fxnUh(z)));
out.fxnU = fxnU;
out.fxnUh = fxnUh;
out.fxnS = fxnS;
out.fxnSh = fxnSh;
out.fxnV = fxnV;
out.fxnVh = fxnVh;
out.s = s;
