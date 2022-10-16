function [Zhat, Zvar] = GaussianMomentsComputation_MJH(Y, Phat,Pvar,yy_min, Quantize_bit,Quantize_stepsize, Var)

%     m = length(Y);
	Mean = 0;
    

    t_lower =  yy_min + Quantize_stepsize*Y;       % lower threshold
    t_upper =  yy_min + (Y+1)* Quantize_stepsize;    % upper threshold

    t_lower(Y == 0) = -inf;
    t_upper(Y == 2^(Quantize_bit)-1 ) = +inf;


    inv_sqrt_Pvar_plus_Var = 1./sqrt(Pvar + Var);
    z_lower = ( Phat - Mean - t_lower ) .* inv_sqrt_Pvar_plus_Var;
    z_upper = ( Phat - Mean - t_upper ) .* inv_sqrt_Pvar_plus_Var;

    % Now compute the probability P(Y == i)
    % Prob = normcdf(z_lower) - normcdf(z_upper);


    Prob = 0.5 * erfc( - (1/sqrt(2))*z_lower ) - 0.5 * erfc( - (1/sqrt(2))*z_upper );

    Ratio_lower = normpdf(z_lower)./Prob;
    Ratio_upper = normpdf(z_upper)./Prob;

    % Finally, compute E[Z(m,t) | Y(m,t)] = Zhat, and
    % var{Z(m,t) | Y(m,t)} = Zvar
    Zhat = Phat +  (Pvar .* inv_sqrt_Pvar_plus_Var) .* (Ratio_lower - Ratio_upper );

    Pvar2_over_Pvar_plus_Var = Pvar.^2 ./ (Pvar + Var); 
    Zvar = Pvar - Pvar2_over_Pvar_plus_Var .* Ratio_lower.*(z_lower + Ratio_lower)...
            - Pvar2_over_Pvar_plus_Var .* Ratio_upper.*(-z_upper + Ratio_upper)...
         + 2* Pvar2_over_Pvar_plus_Var .* Ratio_lower.* Ratio_upper;


    I = find(Prob <= 1e-8); % deal with the case with extremely small Prob
    if ~isempty(I)
    Zhat(I) = Phat(I) + (Pvar(I) ./ (Pvar(I) + Var(I) + 1/12 * (Quantize_stepsize)^2 )).*...
        ( Mean + 0.5*(t_lower(I) + t_upper(I)) - Phat(I));  %% I made a mistake in this line before!!!

    Zvar(I) = Pvar(I).^2 ./ (Pvar(I) + Var(I) + 1/12 * (Quantize_stepsize)^2 );
    end;

%     if any(Zvar(:)<=0 )
%     keyboard;
%     end;

    I1 = find(t_lower == -inf);
    I2 = find(t_upper == +inf);

    Ratio_upper(I1) = (2/sqrt(2*pi)) * (erfcx( z_upper(I1) / sqrt(2)).^(-1));
    Ratio_lower(I2) = (2/sqrt(2*pi)) * (erfcx(-z_lower(I2) / sqrt(2)).^(-1));

    Zhat(I1) = Phat(I1) - ( (Pvar(I1) ./ sqrt(Pvar(I1) + Var(I1))).* Ratio_upper(I1) );

    Zhat(I2) = Phat(I2) + ( (Pvar(I2) ./ sqrt(Pvar(I2) + Var(I2))).* Ratio_lower(I2) );

    Zvar(I1) = Pvar(I1) - (Pvar(I1).^2 ./ (Pvar(I1) + Var(I1))) .*...
    Ratio_upper(I1).*(-z_upper(I1) + Ratio_upper(I1));
    Zvar(I2) = Pvar(I2) - (Pvar(I2).^2 ./ (Pvar(I2) + Var(I2))) .*...
    Ratio_lower(I2).*(z_lower(I2) + Ratio_lower(I2));


    % For the cases other than 'random_QPSK'
    %                     Zhat = max(min(Zhat, 4 * 2^( Quantize_bit -1)* Quantize_stepsize), ...
    %                         - 4 * 2^( Quantize_bit -1)* Quantize_stepsize);

%     Zhat = Zhat.*sign(Phat);
%     Zvar = 2*Zvar;
    %                         1
%     if any(Zvar(:)<=0 | Zvar(:)== +inf | isnan(Zvar(:)))
%     keyboard;
%     end
end