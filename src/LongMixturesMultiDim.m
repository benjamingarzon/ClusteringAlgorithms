

function [logLikelihood, betas, Pki, pks, singularCov]=LongMixturesMultiDim(X, time, nClusters, order, plotting)
% Gaussian mixture modeling for polynomial trajectories
% Benji Garzón April 2012

% Pki: membership matrix, observations x clusters
% X: data observations x variables x timepoints
% time: observations x timepoints

eps=1e-6;
epsilon=1e-10;
oldLogLikelihood=-1e60;
singularCov = 0;
itermax = 300;

timeRange = linspace(min(time(:)), max(time(:)), 100);

% sigmas: variables x variables x clusters
% betas: order x variables x clusters

observations=size(X,1);
variables=size(X,2);
timepoints=size(X,3);

%initialize parameters
pks=ones(1,nClusters)/nClusters;
sigmas=repmat(cov( X(all(~isnan(X(:,:,1)),2),:,1)), [1 1 nClusters]);

%fit a polynomial to data of the first time point to get something to start
%with, and add a random perturbation

for v=1:variables
    xX=X(~isnan(X(:,v,1)),1,1); xt=time(~isnan(X(:,v,1)),1);
    p = polyfit(xt,xX,order);
    betas(:,v,:)=(ones(order+1,1,nClusters)+randn(order+1, 1, nClusters)/10).*repmat(p(end:-1:1)',[1, 1, nClusters]);
end

%iterate
for iter=1:itermax
    
    % E-step
    
    for obs=1:observations
        t=time(obs,:);
        
        y=squeeze(permute(X(obs,:,:),[2 3 1])); % variables x timepoints
        
        for clus2=1:nClusters
            den(clus2)=log(pks(clus2))+gDensity(y, betas(:,:,clus2), sigmas(:,:,clus2), t);
        end
        for clus=1:nClusters
            a=sum(exp(den-den(clus)));
            if isinf(a)
                Pki(obs,clus)=0;
            else
                Pki(obs,clus)=inv(a) ;
            end
        end
        q(obs) = sum(exp(den));
    end
    
    for clus=1:nClusters
        for v=1:variables
            curves(:,v,clus)=polyval(betas(end:-1:1,v,clus),timeRange)';
        end
    end
    
    logLikelihood=sum(log(q));
    
    if plotting
        figure(1)
        for v=1:variables
            subplot(1,variables,v)
            plot(timeRange, squeeze(curves(:,v,:)))
            xlabel('Age(years)')
        end
        
        figure(2)
        plot(iter,logLikelihood,'r.')
        xlabel('Log-Likelihood')
        ylabel('Iterations')
        hold on
    end
    
    % M-step: calculate betas and sigmas
    % betas
    
    for clus=1:nClusters
        
        tMat = zeros(order+1);
        B = zeros(order+1,variables);
        
        for obs=1:observations
            t=time(obs,:);
            
            y=squeeze(permute(X(obs,:,:),[2 3 1])); % variables x timepoints
            noNan=all(~isnan(y), 1);
            
            for timepoint=1:numel(noNan)
                if noNan(timepoint)
                    tVec=t(timepoint).^([ 1:order + 1 ] - 1);
                    tMat = tMat + Pki(obs,clus)*tVec'*tVec;
                    for v=1:variables
                        B(:,v) = B(:,v) + Pki(obs,clus)*tVec'* y(v,timepoint);
                    end
                end
                
            end
            
        end
        
        % solve polynomial for each variable
        for v=1:variables
            betas(:,v,clus)=linsolve(tMat, B(:,v) );
        end
        pks(clus)=mean(Pki(:,clus));
    end
    
    % sigmas
    
    for clus=1:nClusters
        num=0;
        den=0;
        for obs=1:observations
            
            t=time(obs,:);
            y=squeeze(permute(X(obs,:,:),[2 3 1])); % variables x timepoints
            noNan=all(~isnan(y), 1);
            
            for timepoint=1:numel(noNan)
                
                if noNan(timepoint)
                    tVec=t(timepoint).^([ 1:order + 1 ] - 1);
                    num = num + Pki(obs,clus)*(y(:,timepoint)-betas(:,:,clus)'*tVec')*(y(:,timepoint)-betas(:,:,clus)'*tVec')';
                    den = den + Pki(obs,clus);
                end
            end
        end        
        
        sigmas(:,:,clus) = num/den;
        
        if abs(det(sigmas(:,:,clus))) <epsilon;
            singularCov = 1;
        end
        
    end
    if singularCov
        display('Covariance matrix close to singular, restarting')
        break;
        
    end
    %    stop criterion
    e=abs(logLikelihood-oldLogLikelihood)/abs(oldLogLikelihood);
    
    display(['Iteration: ' num2str(iter) ': Log-likelihood ' num2str(logLikelihood) ])
    
    if (e < eps)
        display(['Convergence reached in ' num2str(iter) ' iterations'])
        break;
    end
    
    oldLogLikelihood = logLikelihood;
    
end

if (iter==itermax)
    display(['Max number of iterations reached.' ])
end

end

function d=gDensity(y, beta, sigma, t)

% y: variables x timepoints
% normal density with trajectory
variables=size(y,1);
timepoints=size(y,2);
noNan=all(~isnan(y), 1);
p=zeros(1,timepoints);
for timepoint=1:timepoints
    if noNan(timepoint)
        tVec=t(timepoint).^([ 1:size(beta,1) ] - 1);
        p(timepoint) = -0.5*(y(:,timepoint)-beta'*tVec')' * inv(sigma) *(y(:,timepoint)-beta'*tVec');
        p(timepoint) = p(timepoint) - log(sqrt( ( (2*pi)^variables ) * det(sigma) ));
    end
end

d=sum(p);

end