

function [id, logLikelihood, Pki, pks, singularCov]=GM_MRF(z, mask, nClusters, plotting)
% Gaussian mixture modeling with MRF
% Benji Garzón July 2012

% Pki: membership matrix, observations x clusters
% vol: 4D multimodal image
vars=size(z,2);

eps=1e-6;
epsilon=1e-10;
oldLogLikelihood=-1e60;
singularCov = 0;
itermax = 100;

%timeRange = linspace(min(time(:)), max(time(:)), 100);

% sigmas: variables x variables x clusters
% mus: variables x clusters
% y: variables x observations

for j=1:size(z,2)
    vol=zeros(size(mask));
    vol(mask>0)=z(:,j);
    vols(:,:,:,j)=vol;
end


obs=size(z,1);

%start with kmeans sigma, mu
id=kmeans(z,nClusters);
y=z';
for clus=1:nClusters
    x=y(:,id==clus);
    
    mus(:,clus)=mean(x,2);
    sigmas(:,:,clus)=cov(x');
    
    pks(clus)=size(x,2)/obs;
end


%iterate
for iter=1:itermax
    iter
    % E-step
    
    for clus=1:nClusters
        mu=mus(:,clus);
        sigma=sigmas(:,:,clus);
        
        % rellenar
        Pki(:,clus)=exp(gDensity(y,mu,sigma) + log(pks(clus)))';
    end
    
    q=sum(Pki,2);
    
    Pki=Pki./repmat(q,[1 nClusters]);
    
    logLikelihood=sum(log(q));
    
    % M-step: calculate mus and sigmas
    
    for clus=1:nClusters
        
        p=sum(Pki(:,clus));
        
        mus(:,clus)=y*Pki(:,clus)/p;
        sigmas(:,:,clus)=zeros(vars,vars);
        
        for i=1:obs
            d=(y(:,i)-mus(:,clus));
            sigmas(:,:,clus)=sigmas(:,:,clus) + Pki(i,clus)*d*d';
        end
        
        sigmas(:,:,clus)=sigmas(:,:,clus)/p;
        
        if abs(det(sigmas(:,:,clus))) <epsilon;
            singularCov = 1;
        end
        
    end
    
    pks=mean(Pki);
    
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
    
    if plotting
        for i=1:obs
            
            [m,id(i)]=max(Pki(i,:));
            
        end
        figure(1)
        clf
        scatter(y(1,:),y(2,:),10,id,'filled');
        hold on
        for clus=1:nClusters
            plot(mus(1,clus),mus(2,clus),'x')
            pause(0.1)
        end
        drawnow
    end
end

if (iter==itermax)
    display(['Max number of iterations reached.' ])
end

for i=1:obs
    [m,id(i)]=max(Pki(i,:));
end

end

function p=gDensity(y, mu, sigma)

% y: variables x observations
% normal density
n=size(y,1);
obs=size(y,2);
for i=1:obs
    p(i) = -0.5*(y(:,i)-mu)' * inv(sigma) *(y(:,i)-mu);
end
p = p - log(sqrt( ( (2*pi)^n ) * det(sigma) ));

end