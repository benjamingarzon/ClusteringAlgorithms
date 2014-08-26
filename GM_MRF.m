

function [id, logLikelihood, Pki, pks, singularCov, mus, sigmas]=GM_MRF(z, mask, nClusters, plotting, iterMax)
% Gaussian mixture modeling with MRF
% Benji Garzón July 2012


% Pki: membership matrix, observations x clusters
vars=size(z,2);

eps=1e-6;
epsilon=1e-10;
oldLogLikelihood=-1e60;
singularCov = 0;


% sigmas: variables x variables x clusters
% mus: variables x clusters
% y: variables x observations

%for j=1:size(z,2)
%    vol=zeros(size(mask));
%    vol(mask>0)=z(:,j);
%    vols(:,:,:,j)=vol;
%end

%image=zeros(size(mask));

radius=2; diam=2*radius+1;
alpha=0.5;
G = ones(nClusters, nClusters)-eye(nClusters,nClusters);
G = alpha*G;
[cx,cy,cz] = ndgrid(-radius:radius, -radius:radius, -radius:radius);
% spherical
cx=cx(:); cy=cy(:); cz=cz(:); 
%remove center
p=~cx&~cy&~cz;
cx(p)=[];cy(p)=[];cz(p)=[];
p=(cx.^2+cy.^2+cz.^2 > radius^2);
cx(p)=[];cy(p)=[];cz(p)=[];

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

Pki=ones(obs, nClusters)/nClusters;


%iterate
for iter=1:iterMax
    display(['Iteration ' num2str(iter)])
       
    %regularization
    
     PkiVol = zeros([size(mask), nClusters]);
     U=zeros(size(Pki));

     for clus=1:nClusters
             vol=zeros(size(mask));
             vol(mask>0)=Pki(:,clus);
             PkiVol(:,:,:,clus)=vol;
     end
     n=1;
     for bx=1:size(mask,1)
         for by=1:size(mask,2)
         for bz=1:size(mask,3)
             if(mask(bx,by,bz))
                 for k=1:numel(cx);
                 aux(k,:) = PkiVol(bx+cx(k),by+cy(k),bz+cz(k),:);
                 end
                 % does the zero matter
                 U(n,:)=sum(aux*G);
                 n=n+1;
             end
         end
         end
     end
    
    
    % E-step
    
    for clus=1:nClusters
        mu=mus(:,clus);
        sigma=sigmas(:,:,clus);
        
        Pki(:,clus)=exp(gDensity(y,mu,sigma)' - U(:,clus))';
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
    
    
for i=1:obs
    [m,id(i)]=max(Pki(i,:));
end
    
    if plotting

        figure(9)
        subplot(2,1,1)
        hold off
        scatter(y(1,:),y(2,:),10,id,'filled');
        hold on
        for clus=1:nClusters
            plot(mus(1,clus),mus(2,clus),'x')
            pause(0.1)
        end
        subplot(2,1,2)
        
        plot(iter,logLikelihood,'x')
        hold on
        
        drawnow
    end
end

if (iter==iterMax)
    display(['Max number of iterations reached.' ])
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