function [err] = octaveNewton
   
   A = [[1 0];[0 1];[-1,0];[0,-1]];
   b = [1 1 1 1]';
   c = [1 0.1]';

   mu1 = 10;
   mu2 = 0.0001;
   mu =  (mu1+mu2)/2;

   [m,n] = size(A);

   x = zeros(n,1);
   s = b;
   y = [2*c(1) 2*c(2) c(1) c(2)]';


   [xa,sa,ya] = calcPathElement(A,b,c,mu1,x,s,y)
   [xb,sb,yb] = calcPathElement(A,b,c,mu2,x,s,y);
	lmu = [0:0.05:0.5  1:0.5:10];
	lXvec = [];
	for i=lmu
   	[xl,sl,yl] = calcPathElement(A,b,c,i,x,s,y);
		lXvec = [lXvec xl]; 
	end
	%lXvec = [lXvec xb] 
	thet = 0.05:0.025:0.95;
	for i = thet
   	[xn,sn,yn,mun,err,xVec] = calcNewtElement(A,b,c,mu,(xa+xb)/2,(sa+sb)/2,(ya+yb)/2,xa,xb,mu1,mu2,i)
   	hold on
		plot(xVec(1,:),xVec(2,:), 'b','linewidth',1.5)
		hold off	
	end
   
	
	hold on
   plot(lXvec(1,:),lXvec(2,:), 'k','linewidth',1)
   plot(xVec(1,:),xVec(2,:), 'b','linewidth',1.5)
	
	%plot(err(:,1))
   %plot(err(:,2))
   %plot(err(:,2))
   %plot(err(:,3))
   %ylim([-1,1])
   hold off
   
   out = [xa,xb,xn]'

   px = out(:,1)
   py = out(:,2)

   hold on
   plot(px,py,'r.','MarkerSize',20)
   axis('equal')
	hold off

endfunction

function [] = divideMuInterval(A,b,c,mu,x,s,y,x1,x2,mu1,mu2,theta,tol)


function [x,s,y,mu,err,xVec] = calcNewtElement(A,b,c,mu,x,s,y,x1,x2,mu1,mu2,theta)
   [m,n] = size(A);

   tol = 10^(-8);
   er = norm([(y.*s)-mu])
	%theta = 0.9;

	%x = (1-theta)*x1+theta*x2;   

   i = 0;
	xVec = [];
   while er >= tol
   %for j = 1:1000
      i += 1;
      M = [[A'*diag(y)*diag(1./s)*A, -A'*diag(1./s)*ones(m,1)];[(x2-x1)', 0]];
      U = [[-mu*A'*diag(1./s)*ones(m,1) + c];[(theta*(norm(x2-x1)^2))-(x2-x1)'*(x-x1)]]
      
      dz = M\U;
      dx = dz(1:end-1);
      dmu = dz(end);

      ds = -A*dx;
      %ds = b-A*(x+dx)-s;

      dy = diag(1./s)*((ones(m,1)*(mu-dmu))-(diag(y)*(s+ds)));


      negIndS = find(ds < 0);
		if isempty(negIndS)
		    alpha = 1;
		else
		    alpha = min(1, 0.9 * min(-s(negIndS) ./ ds(negIndS)));
		end

      negIndY = find(dy < 0);
		if isempty(negIndY)
		    beta = 1;
		else
		    beta = min(1, 0.9 * min(-y(negIndY) ./ dy(negIndY)));
		end

      if (mu-abs(dmu) < min(mu1,mu2))
         gamma = 0.1*min(abs(mu-mu1),abs(mu-mu2))/dmu;
      elseif (mu+abs(dmu) > max(mu1,mu2))
         gamma = 0.1*min(abs(mu-mu1),abs(mu-mu2))/dmu;
      else
         gamma = 1; 
      endif
      
      %alpha = 10^(-2);
      %beta = 10^(-2);
      %gamma = 10^(-2);

      x = x+alpha*dx;
      s = s+alpha*ds;
      y = y+beta*dy;
      mu = mu-gamma*dmu;
      er = norm([(y.*s)-mu]);
      

		xVec = [xVec x];
      err(i,1) = er;
      %err(i,2) = mu;
      %err(i,3) = norm(A'*dy);
      %err(i,4) = norm(gamma);
   end
endfunction


function [x,s,y] = calcPathElement(A,b,c,mu,x,s,y)
   [m,n] = size(A);

   tol = 10^(-8);
   er = norm([(y.*s)-mu]);
   while (er >= tol)
      dx = (A'*diag(y)*diag(1./s)*A)\(c-mu*A'*(1./s));
      ds = -A*dx;
      dy = mu./s - y - diag(y./s)*ds;

      negIndS = find(ds < 0);
		if isempty(negIndS)
		    alpha = 1;
		else
		    alpha = min(1, 0.9 * min(-s(negIndS) ./ ds(negIndS)));
		endif
      
      negIndY = find(dy < 0);
		if isempty(negIndY)
		    beta = 1;
		else
		    beta = min(1, 0.9 * min(-y(negIndY) ./ dy(negIndY)));
		endif

      dely = A'*dy;

      x = x+alpha*dx;
      s = s+alpha*ds;
      y = y+beta*dy;
      er = norm([[(y.*s)-mu];[A'*dy]]);
      %er = norm([(y.*s)-mu]);

   endwhile
endfunction
