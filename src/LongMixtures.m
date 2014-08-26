

function [logLikelihood, betas, Pki]=LongMixtures(X, time, nClusters)
eps=1e-5;
oldLogLikelihood=-1e60;
order = 3; %polynomial order

itermax = 100;


timeRange = linspace(min(time(:)), max(time(:)), 100);
% U: membership matrix, observations x clusters
% X: data observations x variables x timepoints
% time: observations x timepoints
% by the moment assuming one variable
observations=size(X,1);
variables=size(X,2);
timepoints=size(X,3);


%initialize parameters
pks=ones(1,nClusters)/nClusters;
sigmas=ones(1,nClusters);
%betas=zeros(order+1, nClusters);
xX=X(~isnan(X(:,1,1)),1,1); xt=time(~isnan(X(:,1,1)),1);
p = polyfit(xt,xX,order);
betas=(ones(order+1,nClusters)+randn(order+1, nClusters)/5).*repmat(p(end:-1:1)',[1,nClusters]);

%iterate


for iter=1:itermax
    
    % E-step
    
    for clus=1:nClusters
        
        for obs=1:observations
            t=time(obs,:);
            
            y=X(obs,:);
            
            num=pks(clus)*gDensity(y, betas(:,clus), sigmas(clus), t);
            den = 0;
            
            
            for clus2=1:nClusters
                den=den+pks(clus2)*gDensity(y, betas(:,clus2), sigmas(clus2), t);
            end
            
            if den
                Pki(obs,clus)=num/den;
            else
                Pki(obs,clus)=0;
            end
            q(obs)=den;
            
        end
        
        
        
        
        curves(:,clus)=polyval(betas(end:-1:1,clus),timeRange)';
        
    end
    logLikelihood=sum(log(q));
    figure(1)
    
    subplot(2,1,1)
    plot(timeRange, curves)
    xlabel('Age(years)')
    subplot(2,1,2)
    plot(iter,logLikelihood,'ro')
    xlabel('Log-Likelihood')
    xlabel('Iterations')
    hold on
    
    % M-step: calculate betas and sigmas
    

    
    % betas
    
    for clus=1:nClusters
        
    tMat = zeros(order+1);
    B = zeros(order+1,1);
        
        for obs=1:observations
            t=time(obs,:);
            y=X(obs,:);
            
            for i=1:numel(y)
                if ~isnan(y(i))
                    tVec=t(i).^([ 1:order + 1 ] - 1);
                    tMat = tMat + Pki(obs,clus)*tVec'*tVec;
                    B = B + Pki(obs,clus)*tVec' * y(i);
                end
                
            end
                        
        end
        
        % solve
        betas(:,clus)=linsolve(tMat, B );
        pks(clus)=mean(Pki(:,clus));
        
    end
    
    % sigmas
    
    for clus=1:nClusters
        num=0;
        den=0;
        for obs=1:observations
            for timepoint=1:timepoints
                t=time(obs,:);
                y=X(obs,:);
                                
                for i=1:numel(y)
                    if ~isnan(y(i))
                        tVec=t(i).^([ 1:order + 1 ] - 1);
                        num = num + Pki(obs,clus)*(y(i)-tVec*betas(:,clus))^2;
                        den = den + Pki(obs,clus);
                    end
                end
            end
        end
        
        if den
        sigmas(clus)=sqrt(num/den);
        else 
            sigmas(clus)=0;
        end
    end
    
    
    
    %    stop criterion
    e=abs(logLikelihood-oldLogLikelihood)/abs(oldLogLikelihood);
    
    display(['Iteration: ' num2str(iter) ': Log-likelihood ' num2str(logLikelihood) ' Sigma: ' num2str(sigmas)])
    
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

%normal density with trajectory
for i=1:numel(y)
    if ~isnan(y(i))
        tVec=t(i).^([ 1:numel(beta) ] - 1);
        p(i) = exp(-0.5*(tVec*beta - y(i))^2)*inv(sqrt(2*pi)*sigma);
    end
    
end

d=prod(p(~isnan(y)));
end