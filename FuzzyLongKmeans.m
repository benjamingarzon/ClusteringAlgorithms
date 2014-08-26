

function [U,centroids]=FuzzyLongKmeans(X, nClusters, m)
eps=1e-5;
% U: membership matrix, observations x clusters
% X: data observations x variables x timepoints
% m: exponent for u
observations=size(X,1);
variables=size(X,2);
timepoints=size(X,3);

%initialize U
U=zeros(observations, nClusters);
% random
r=randi(nClusters,observations,1);
for i=1:observations
    U(i,r(i))=1;
end
LastU=U;
%iterate

itermax = 100;

for iter=1: itermax
    %calculate centroids
    
    centroids=zeros(nClusters, variables, timepoints);
    
    for time=1:timepoints
        
        %select those that do not contain nans
        
        Xt=X(:,:,time);
        noNan=all(~isnan(Xt),2);
        Xt=Xt(noNan,:);
        Ut=U(noNan,:).^m;
        
        %use mean
        if 1
            centroids(:,:,time)=Ut'*Xt./repmat(sum(Ut)',1, variables);
        else
            %use median
            for cluster=1:nClusters
                centroids(cluster,:,time)=weightedMedian(Xt,Ut(:,cluster)/sum(Ut(:,cluster)));
            end
        end
    end
    
    %update membership matrix
    
    
    for i=1:observations
        
        y = reshape(X(i,:,:),variables, timepoints );
        %calculate distance from the observation to each cluster
        for cluster=1:nClusters
            
            centroid = reshape(centroids(cluster,:,:),variables, timepoints);
            d(cluster)=LongDist(y,centroid);
        end
        
        dm=d.^(2/(m-1))+1e-14;
        aux = zeros(1,nClusters);
        
        U(i,:)= 1./( dm*sum(1./dm) );
        
    end
    
    
    
    %stop criterion
    e=norm(U-LastU)/norm(U);
    
    display(['cost function ' num2str(e) ])
    
    if (e<eps)
        
        display(['Convergence reached in ' num2str(iter) ' iterations'])
        break;
    end
    
    LastU=U;
    
end


if (iter==itermax)
    display(['Max number of iterations reached.' ])
end

end

function dist=LongDist(x,y)
%calculate distance between longitudinal vectors with Gower adjustment
% x,y: vars x timepoints

wx=all(~isnan(x),1);
wy=all(~isnan(y),1);

w=wy.*wx;

%assuming euclidean distance on the variables, could be made mahalanobis
S  = sum ((x - y ).^2,1 );

dist = sqrt( sum(S(w>0))/sum(w) );

end